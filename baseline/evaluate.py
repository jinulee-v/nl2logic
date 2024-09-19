import torch
import os
import random
from torch.utils.data import DataLoader
from dataset import MallsDataset
from tqdm import tqdm
from datasets import load_dataset
from utils import load_model_and_tokenizer, compute_accuracy
import argparse


def evaluate_model(model, dataloader, tokenizer, device):
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    all_inputs = []
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Decode original inputs and store them
            inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            all_inputs.extend(inputs)

            # Forward pass (to get loss)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss  # Loss for the current batch
            progress_bar.set_postfix({"Validation Loss": loss.cpu().item()})
            total_loss += loss.item()  # Accumulate loss

            # Generate predictions
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512)

            # Decode the predictions and labels
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_predictions.extend(predictions)
            all_labels.extend(labels)

    avg_loss = total_loss / len(dataloader)  # Calculate the average loss over all batches
    return all_inputs, all_predictions, all_labels, avg_loss


def main():
    parser = argparse.ArgumentParser(description='Evaluate a fine-tuned T5 model on the MALLS dataset.')
    parser.add_argument('--model_dir', type=str, default='fine_tuned_t5-base_checkpoints/fine_tuned_t5-base_epoch_3', help='Directory where the fine-tuned model is saved.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation.')

    args = parser.parse_args()

    device = torch.device('cuda')

    # Load the tokenizer and model
    model, tokenizer = load_model_and_tokenizer('t5-base', args.model_dir)

    # Load the evaluation dataset
    data_dir = './malls'
    eval_dataset = load_dataset('json', data_files=os.path.join(data_dir, 'validation.json'), split='train')
    
    # Prepare dataset for evaluation
    eval_data = MallsDataset(eval_dataset, tokenizer)
    eval_dataloader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False)

    # Evaluate the model
    print("Starting evaluation...")
    inputs, predictions, labels, avg_loss = evaluate_model(model, eval_dataloader, tokenizer, device)

    # Compute evaluation metrics (accuracy in this case)
    accuracy = compute_accuracy(predictions, labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Evaluation Loss: {avg_loss:.4f}")

    print("\nRandom 10 predictions, inputs, and labels:")
    random_indices = random.sample(range(len(predictions)), 10)

    for i, idx in enumerate(random_indices):
        print(f"Example {i+1}:")
        print(f"Input: {inputs[idx]}")
        print(f"Prediction: {predictions[idx]}")
        print(f"Label: {labels[idx]}")
    print()


if __name__ == "__main__":
    main()
