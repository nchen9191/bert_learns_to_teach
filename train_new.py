from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import DataLoader

from evaluate import task_eval
from initialize import get_config, load_models, load_tokenizer
from pre_processing import get_data_loaders


def run_full_training(config_path):
    # Initialize with relevant parameters
    config = get_config(config_path)

    # Load initial models
    teacher, student = load_models(config)

    # Get tokenizer
    tokenizer = load_tokenizer(config)

    # Get Train and Dev DataLoaders
    train_dataloader, quiz_dataloader, val_dataloader = get_data_loaders(config, tokenizer)

    # Run training
    final_teacher, final_student, train_loss, val_loss = train(config,
                                                               teacher,
                                                               student,
                                                               train_dataloader,
                                                               val_dataloader)

    # Get model metrics
    metrics = task_eval(final_student, val_dataloader, config['task'])

    return final_teacher, final_student, metrics

def train(config: dict,
          teacher: Module,
          student: Module,
          train_dataloader: DataLoader,
          quiz_dataloader: DataLoader,
          val_dataloader: DataLoader) -> Tuple[Module, Module, float, float]:
    """
    Train student and teacher models on the training set, evaluate on the validation set, and return the best models.

    Args:
        config (dict): Configuration dictionary.
        teacher (Module): Teacher model.
        student (Module): Student model.
        train_dataloader (DataLoader): DataLoader for the training set.
        quiz_dataloader (DataLoader): DataLoader for the quiz set.
        val_dataloader (DataLoader): DataLoader for the validation set.

    Returns:
        Tuple of the final teacher and student models and their corresponding validation loss.
    """
    # Set up optimizer for student and teacher models
    optimizer_S = torch.optim.Adam(student.parameters(), lr=config['learning_rate_student'])
    optimizer_T = torch.optim.Adam(teacher.parameters(), lr=config['learning_rate_teacher'])

    # Set up loss function for student and teacher models
    loss_fn_S = F.cross_entropy
    loss_fn_T = F.kl_div

    # Initialize best validation loss to infinity
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(config['num_epochs']):
        # Train student and teacher models on the training set
        for step, batch in enumerate(train_dataloader):
            # Step 2: Sample batch of training data x ~ D
            inputs, labels = batch
            inputs = inputs.to(config['device'])
            labels = labels.to(config['device'])

            # Step 3: Copy student parameter θ_S to student θ_S'
            student_prime = student.clone()

            # Step 4: Update θ_S' with x and θ_T: θ_S' <- θ_S' - λ∇θ_S' LS(x;θ_S;θ_T)
            optimizer_S.zero_grad()
            logits_S = student(inputs)
            loss_S = loss_fn_S(logits_S, labels)
            loss_S.backward()
            optimizer_S.step()

            # Step 5: Sample a batch of quiz data q ~ Q
            inputs_q, labels_q = next(iter(quiz_dataloader))
            inputs_q = inputs_q.to(config['device'])
            labels_q = labels_q.to(config['device'])

            # Step 6: Update θ_T with q and θ_S': θ_T <- θ_T - μ∇θ_T LT(q,θ_S'(θ_T))
            optimizer_T.zero_grad()
            logits_S_prime = student_prime(inputs_q)
            logits_T = teacher(inputs_q)
            loss_T = loss_fn_T(F.log_softmax(logits_S_prime / config['temperature'], dim=1),
                               F.softmax(logits_T / config['temperature'], dim=1),
                               reduction='batchmean')
            loss_T.backward()
            optimizer_T.step()

            # Step 7: Update original θ_S with x and the updated θ_T: θ_S <- θ_S - λ∇θ_S LS(x;θ_S;θ_T)
            optimizer_S.zero_grad()
            logits_S = student(inputs)
            logits_T = teacher(inputs)
            loss_S = loss_fn_S(logits_S, labels) + config['lambda'] * loss_fn_T(F.log_softmax(logits_S / config['temperature'], dim=1), F.softmax(logits_T / config['temperature'], dim=1), reduction='batchmean')
            loss_S.backward()
            optimizer_S.step()

            # Print loss every n steps
            if step % config['print_every'] == 0:
                print(f"Epoch [{epoch}/{config['num_epochs']}] Step [{step}/{len(train_dataloader)}]:"
                      f" Loss_S: {loss_S.item():.4f}, Loss_T: {loss_T.item():.4f}")

        # Evaluate student model on the validation set
        val_loss, _ = task_eval(config, student, val_dataloader)

        # Save best models based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_teacher = teacher
            best_student = student

    return best_teacher, best_student,loss_S, best_val_loss



