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
          val_dataloader: DataLoader) -> Tuple[Module, Module, float, float]:
    device = config['device']
    task_loss_fn = config['task_loss_fn']
    alpha = config['alpha']
    beta = config['beta']
    lambda_S = config['lambda_S']
    lambda_T = config['lambda_T']
    num_iterations = config['num_iterations']
    print_every = config['print_every']
    eval_every = config['eval_every']
    best_loss = float('inf')
    best_student = None

    for iteration in range(1, num_iterations+1):
        # Sample batch of training data
        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, token_type_ids, labels = batch

            # Copy student parameter to student'
            student_copy = student.copy()

            # Update student' with x and teacher
            student_copy.train()
            teacher.eval()
            teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask, 
                                      token_type_ids=token_type_ids)
            loss = task_loss_fn(student_copy(input_ids=input_ids, attention_mask=attention_mask, 
                                             token_type_ids=token_type_ids), labels)
            student_copy.zero_grad()
            loss.backward()
            student_copy_optim = torch.optim.AdamW(student_copy.parameters(), lr=3e-5)
            student_copy_optim.step()

            # Update teacher with q and student'
            for quiz_batch in quiz_dataloader:
                quiz_batch = tuple(t.to(device) for t in quiz_batch)
                quiz_input_ids, quiz_attention_mask, quiz_token_type_ids, quiz_labels = quiz_batch

                student_copy.eval()
                student_outputs = student_copy(input_ids=quiz_input_ids, attention_mask=quiz_attention_mask, 
                                               token_type_ids=quiz_token_type_ids)

                teacher.train()
                teacher_outputs = teacher(quiz_input_ids, quiz_attention_mask, quiz_token_type_ids)
                distillation_loss = F.mse_loss(student_outputs, teacher_outputs)
                teacher.zero_grad()
                distillation_loss.backward()
                teacher_optim = torch.optim.AdamW(teacher.parameters(), lr=3e-5)
                teacher_optim.step()

                break  # Only one batch of quiz data is used in each iteration

            # Update original student with x and updated teacher
            student.train()
            teacher.eval()
            teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask, 
                                      token_type_ids=token_type_ids)
            loss = task_loss_fn(student(input_ids=input_ids, attention_mask=attention_mask, 
                                         token_type_ids=token_type_ids), labels)
            distillation_loss = F.mse_loss(student(input_ids=input_ids, attention_mask=attention_mask, 
                                                    token_type_ids=token_type_ids), teacher_outputs)
            loss = alpha * distillation_loss + beta * loss
            student.zero_grad()
            loss.backward()
            student_optim = torch.optim.AdamW(student.parameters(), lr=3e-5)
            student_optim.step()

        # Evaluate the student model
        if iteration % eval_every == 0:
            student.eval()
            with torch.no_grad():
                # Compute validation loss and other metrics
                val_loss, val_metrics = task_eval(student, val_dataloader, config)
            print('Iteration: {}/{} | Train Loss: {:.6f} | Val Loss: {:.6f}'.format(iteration, num_iterations, loss.item(), val_loss))
            # Check if the current student model is the best so far
            if val_loss < best_loss:
                best_loss = val_loss
                best_student = student
                # Print training loss every few iterations
        if iteration % print_every == 0:
            print('Iteration: {}/{} | Train Loss: {:.6f} | Best Val Loss: {:.6f}'.format(iteration, num_iterations, loss.item(), best_loss))

    # Use the best student model to get the best teacher model
    best_teacher = teacher.copy()
    best_teacher.load_state_dict(best_student.state_dict())


    return best_student,best_teacher,loss,val_loss


