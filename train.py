import copy
import gc
from collections import OrderedDict
from typing import Tuple

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PretrainedConfig

from distillation_meta import MetaPatientDistillation
from evaluate import task_eval
from initialize import get_config, load_models, load_tokenizer
from pre_processing import get_data_loaders
from utils import get_optimizer_and_scheduler, get_order, save_teacher_student_models


def run_full_training(config_path):
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize with relevant parameters
    config = get_config(config_path)

    # Load initial models
    teacher, student, teacher_config, student_config = load_models(config, device)
    print("Loaded models")

    # Get tokenizer
    tokenizer = load_tokenizer(config)

    # Get Train and Dev DataLoaders
    train_dataloader, quiz_dataloader, val_dataloader = get_data_loaders(config, tokenizer)
    print("Loaded data")

    # Run training
    print("Beginning meta distil training")
    final_teacher, final_student = train(config,
                                         teacher,
                                         student,
                                         teacher_config,
                                         student_config,
                                         train_dataloader,
                                         quiz_dataloader,
                                         val_dataloader,
                                         device)

    # Save models
    save_teacher_student_models(config, teacher, student, tokenizer)
    print("Models saved")

    # Get model metrics
    metrics, _, _ = task_eval(final_student, val_dataloader, config['task'], device)
    print(f"Task ({config['task']})  Metrics: {metrics}")

    return final_teacher, final_student, metrics


def train(config: dict,
          teacher: Module,
          student: Module,
          teacher_config: PretrainedConfig,
          student_config: PretrainedConfig,
          train_dataloader: DataLoader,
          quiz_dataloader: DataLoader,
          val_dataloader: DataLoader,
          device: str) -> Tuple[Module, Module]:
    """
    Train student and teacher models on the training set, evaluate on the validation set, and return the best models.

    Args:
        config (dict): Configuration dictionary.
        teacher (Module): Teacher model.
        student (Module): Student model.
        teacher_config (PretrainedConfig): Teacher config from transformers library,
        student_config (PretrainedConfig: Student config from transformers library,
        train_dataloader (DataLoader): DataLoader for the training set.
        quiz_dataloader (DataLoader): DataLoader for the quiz set.
        val_dataloader (DataLoader): DataLoader for the validation set.
        device (string): training device e.g cpu, cuda

    Returns:
        Tuple of the final teacher and student models
    """

    # Forward methods
    meta_distil_forward = MetaPatientDistillation(teacher_config, student_config)
    order = get_order(teacher_config, student_config)

    # Set up optimizer for student and teacher models
    total_steps = config['num_epochs'] * len(train_dataloader)
    t_optimizer, t_scheduler = get_optimizer_and_scheduler(teacher.named_parameters(), total_steps, config, True)
    s_optimizer, s_scheduler = get_optimizer_and_scheduler(student.named_parameters(), total_steps, config)

    quiz_loss = 0.0
    s_prime_total_avg_loss, s_prime_train_avg_loss, s_prime_soft_avg_loss, s_prime_distill_avg_loss = 0.0, 0.0, 0.0, 0.0
    student_total_avg_loss, student_train_avg_loss, student_soft_avg_loss, student_distill_avg_loss = 0.0, 0.0, 0.0, 0.0

    train_results, train_loss, _ = task_eval(student, train_dataloader, config['task'], device)
    results, val_loss, _ = task_eval(student, val_dataloader, config['task'], device)

    print(f"Epoch: {0} (no training), "
          f"Student Train Loss: {train_loss}, "
          f"Student Val Loss: {val_loss}, "
          f"Task Metrics: {results}")

    teacher.zero_grad()
    student.zero_grad()

    # Training loop
    for epoch in range(config['num_epochs']):
        # Train student and teacher models on the training set

        for step, batch in enumerate(tqdm(train_dataloader, desc='Train')):
            # Step 2: Sample batch of training data x ~ D
            student_weights = OrderedDict((name, param) for (name, param) in student.named_parameters())
            student_backup_state_dict = copy.deepcopy(student.state_dict())
            s_optimizer_backup_state_dict = copy.deepcopy(s_optimizer.state_dict())

            student.train(), teacher.eval()

            batch = tuple(data.to(device) for data in batch)
            input_ids, attention_mask, token_type_ids, labels = batch[0], batch[1], batch[2], batch[3]

            # Step 4: Update θ_S' with x and θ_T: θ_S' <- θ_S' - λ∇θ_S' LS(x;θ_S;θ_T)
            s_prime_train_loss, s_prime_soft_loss, s_prime_pkd_loss = meta_distil_forward(
                t_model=teacher,
                s_model=student if step == 0 else student_weights,
                order=order,
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
                config=config,
                teacher_grad=True)

            s_prime_loss = config['alpha'] * s_prime_soft_loss \
                           + (1 - config['alpha']) * s_prime_train_loss \
                           + config['beta'] * s_prime_pkd_loss

            grads = torch.autograd.grad(s_prime_loss,
                                        student.parameters() if step == 0 else student_weights.values(),
                                        create_graph=True,
                                        retain_graph=True)

            student_weights = OrderedDict(
                (name, param - config['learning_rate_student'] * grad)
                for ((name, param), grad) in zip(student_weights.items(), grads)
            )

            s_prime_total_avg_loss += s_prime_loss.item()
            s_prime_train_avg_loss += s_prime_train_loss.item()
            s_prime_soft_avg_loss += s_prime_soft_loss.item()
            s_prime_distill_avg_loss += s_prime_pkd_loss.item()

            # Calculate s_prime_loss and t_grads
            s_prime_quiz_loss = torch.tensor(0, dtype=s_prime_loss.dtype, device=device)
            quiz_batch_num = 0

            teacher.train()
            for step, q_batch in enumerate(quiz_dataloader):
                # if (step + 1) % 1 == 0:
                #     print(f"Processing quiz batch: {step + 1}")
                q_batch = tuple(t.to(device) for t in q_batch)
                q_input_ids, q_attention_mask, q_token_type_ids, q_labels = q_batch[:4]

                s_prime_step_loss = meta_distil_forward.s_prime_forward(
                    s_prime=student_weights,
                    input_ids=q_input_ids,
                    token_type_ids=q_token_type_ids,
                    attention_mask=q_attention_mask,
                    labels=q_labels
                )

                s_prime_quiz_loss += s_prime_step_loss
                quiz_batch_num += 1

            s_prime_quiz_loss /= quiz_batch_num
            t_grads = torch.autograd.grad(s_prime_quiz_loss, teacher.parameters())

            for p, gr in zip(teacher.parameters(), t_grads):
                p.grad = gr

            torch.nn.utils.clip_grad_norm_(teacher.parameters(), config['max_grad_norm'])

            quiz_loss += s_prime_quiz_loss.item()

            t_optimizer.step()
            t_scheduler.step()

            # Manual zero_grad
            for p in teacher.parameters():
                p.grad = None

            for p in student.parameters():
                p.grad = None

            del t_grads, grads, student_weights
            # del grads
            # del student_weights

            # Step 7: Update original θ_S with x and the updated θ_T: θ_S <- θ_S - λ∇θ_S LS(x;θ_S;θ_T)
            student.load_state_dict(student_backup_state_dict)
            s_optimizer.load_state_dict(s_optimizer_backup_state_dict)
            del student_backup_state_dict, s_optimizer_backup_state_dict

            student.train(), teacher.eval()

            student_train_loss, student_soft_loss, student_pkd_loss = meta_distil_forward(
                t_model=teacher,
                s_model=student,
                order=order,
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
                config=config,
                teacher_grad=False
            )

            student_loss = config['alpha'] * student_soft_loss \
                           + (1 - config['alpha']) * student_train_loss \
                           + config['beta'] * student_pkd_loss

            student_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), config['max_grad_norm'])

            s_optimizer.step()
            s_scheduler.step()
            s_optimizer.zero_grad()

            student_total_avg_loss += student_loss.item()
            student_train_avg_loss += student_train_loss.item()
            student_soft_avg_loss += student_soft_loss.item()
            student_distill_avg_loss += student_pkd_loss.item()

            gc.collect()
            torch.cuda.empty_cache()

        # Evaluate student model on the validation set
        results, val_loss, _ = task_eval(student, val_dataloader, config['task'], device)

        print(f"Epoch: {epoch + 1}, "
              f"Student Train Loss: {student_total_avg_loss / len(train_dataloader)}, "
              f"Student Val Loss: {val_loss}, "
              f"Task Metrics: {results}")

    return teacher, student


if __name__ == '__main__':
    config_path = 'example_config.json'
    run_full_training(config_path)
