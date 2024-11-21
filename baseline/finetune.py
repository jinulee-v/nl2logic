import os
import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from transformers import AdamW
from dataset import MallsDataset
from model import T5Model
from datasets import load_dataset, DatasetDict
from utils import save_model
import argparse


def train(model, dataloader, optimizer, device, print_every=100):
    model.train()
    scaler = GradScaler()  # Initialize a gradient scaler for mixed precision
    progress_bar = tqdm(dataloader, total=len(dataloader), desc="Training", unit="batch")


    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)


        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Scales the loss, calls backward() on scaled loss to create scaled gradients
        scaler.scale(loss).backward()
        # Unscales gradients and updates optimizer
        scaler.step(optimizer)
        # Updates the scale for next iteration
        scaler.update()
        progress_bar.set_postfix({"Training Loss": loss.cpu().item()})


def finetune(train_dataset, model_name='t5-base', epochs=3, batch_size=4, lr=5e-5, max_length=512, save_dir='fine_tuned_t5-base_checkpoints'):
    device = torch.device('cuda')

    # Initialize T5 model and tokenizer
    t5_model = T5Model(model_name)
    model = t5_model.get_model().to(device)
    tokenizer = t5_model.get_tokenizer()

    # Load dataset
    dataset = MallsDataset(train_dataset, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train(model, dataloader, optimizer, device)
        # Save the fine-tuned model
        model_save_path = os.path.join(save_dir, f'fine_tuned_{model_name}_epoch_{epoch}')
        os.makedirs(save_dir, exist_ok=True)
        save_model(model, tokenizer, model_save_path)
        print(f"Model saved at '{model_save_path}'")


def save_dataset_as_json(data, json_file):
    """Save the entire dataset to a JSON file."""
    with open(json_file, 'w') as f:
        json.dump(data, f)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune T5 model with MALLS dataset.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for the optimizer.')
    parser.add_argument('--max_length', type=int, default=258, help='Maximum sequence length for input.')
    parser.add_argument('--save_dir', type=str, default='fine_tuned_t5-base_checkpoints', help='Directory to save fine-tuned model.')

    args = parser.parse_args()

    # Directory to store the dataset
    data_dir = './malls'

    if not os.path.exists(data_dir):
        print("Loading MALLS dataset from Hugging Face...")
        
        # Load the MALLS dataset from Hugging Face
        malls_dataset = load_dataset('yuan-yang/MALLS-v0')

        # Split the test set into validation and test sets (80/20 split)
        test_split = malls_dataset['test']
        test_dataset = test_split.train_test_split(test_size=0.2)
        
        # Combine the splits into a DatasetDict
        dataset_dict = DatasetDict({
            'train': malls_dataset['train'],
            'validation': test_dataset['train'],
            'test': test_dataset['test']
        })
        
        # Save dataset locally in a folder called 'malls'
        os.makedirs(data_dir, exist_ok=True)
        dataset_dict['train'].to_json(os.path.join(data_dir, 'train.json'))
        dataset_dict['validation'].to_json(os.path.join(data_dir, 'validation.json'))
        dataset_dict['test'].to_json(os.path.join(data_dir, 'test.json'))
        print(f"Dataset saved locally in '{data_dir}'.")
    else:
        print(f"Loading dataset from '{data_dir}'.")

    # Load the dataset using Hugging Face datasets library
    train_dataset = load_dataset('json', data_files=os.path.join(data_dir, 'train.json'), split='train')

    # Display dataset details and examples
    print(f"Dataset loaded. Number of examples: {len(train_dataset)}")
    print("Here are a few examples from the dataset:")

    for i in range(3):
        example = train_dataset[i]
        print(f"Example {i+1}:")
        print(f"  Natural Language (NL): {example['NL']}")
        print(f"  FOL: {example['FOL']}")

    # Fine-tune the model using the loaded dataset
    print("\nStarting fine-tuning process...")
    finetune(
        train_dataset,
        model_name='t5-base',
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
