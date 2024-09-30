import torch
import os
import random
from torch.utils.data import DataLoader
from dataset import MallsDataset
from tqdm import tqdm
from datasets import load_dataset
from utils import load_model_and_tokenizer, compute_accuracy
import argparse
import json


def generate_model(model, dataloader, tokenizer, device, args):
    model.eval()
    model.to(device)
    
    all_predictions = []
    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")

    with torch.no_grad():
        for batch in progress_bar:
            inputs = tokenizer(batch["nl"], return_tensors="pt", padding=True, truncation=True)

            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # Set the number of return sequences based on the inference mode
            num_return_sequences = 1
            if args.beam_size > 1:
                num_return_sequences = args.beam_size
            elif args.sample:
                num_return_sequences = args.sample_size

            # Generate predictions
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=args.beam_size, # Beam Search
                num_beam_groups=args.beam_size if args.dbs else 1, # Diverse Beam Search
                do_sample=args.sample, # Sampling
                temperature=args.temperature, # Temperature for sampling
                num_return_sequences=num_return_sequences, # Return >1 sequence for beam search and sampling
                max_length=512 # Maximum length of the generated sequence
            )

            # Decode the predictions and labels
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Resize predictions to [batch_size, num_return_sequences]
            resized_predictions = []
            for i in range(len(predictions)):
                if i % num_return_sequences == 0:
                    resized_predictions.append([])
                resized_predictions[-1].append(predictions[i])

            all_predictions.extend(resized_predictions)

    return all_predictions


def main():
    parser = argparse.ArgumentParser(description='Run T5 inference on a dataset.')
    parser.add_argument('--model_dir', type=str, default='model', help='Directory where the fine-tuned model is saved.')
    parser.add_argument('--dataset', type=str, choices=['entailmentbank', 'enwn', 'eqasc', 'folio', 'prontoqa'], help='Dataset name to evaluate on.')
    parser.add_argument('--data_dir', type=str, default='../data', help='Directory to save outputs.')
    parser.add_argument('--output_dir', type=str, default='../results/baseline', help='Directory to save outputs.')
    parser.add_argument('--experiment_name', type=str, default=None, help='Directory to save outputs.')

    parser.add_argument('--split', type=str, choices=['train', 'validation', 'test'], help='Split to evaluate on.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference.')

    # Decoding settings: defaults to greedy decoding
    parser.add_argument('--beam_size', type=int, default=1, help='Beam size for inference.')
    parser.add_argument('--dbs', action='store_true', help='Use Diverse Beam Search.')
    parser.add_argument('--sample', action="store_true", help='If set, use sampling instead of beam search.')
    parser.add_argument('--sample_size', type=int, default=1, help='Number of samples to return for sampling.')
    parser.add_argument('--temperature', type=float, default=0, help='Temperature for inference.')
    
    args = parser.parse_args()

    if args.experiment_name is None:
        if args.sample:
            args.experiment_name = f"sample_size{args.sample_size}_temp{args.temperature}"
        elif args.dbs:
            args.experiment_name = f"dbs_size{args.beam_size}"
        else:
            args.experiment_name = f"beam_size{args.beam_size}"
    # Create output directory if it doesn't exist
    if not os.path.exists(f"{args.output_dir}/{args.experiment_name}"):
        os.makedirs(f"{args.output_dir}/{args.experiment_name}")

    device = torch.device('cuda')

    # Load the tokenizer and model
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer('t5-base', args.model_dir)
    model.to(device)

    # Load the dataset to generate from: {dataset}_{split}_sentences.jsonl
    print("Loading dataset...")
    data_dir = f'{args.data_dir}/{args.dataset}_{args.split}_sentences.jsonl'
    eval_dataset = load_dataset('json', data_files=data_dir)["train"] # It is always train split, dunno why
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluate the model
    print("Starting generation...")
    predictions = generate_model(model, eval_dataloader, tokenizer, device, args)
    print(predictions)

    with open(f'{args.output_dir}/{args.experiment_name}/{args.dataset}_{args.split}_sentences.jsonl', 'w') as f:
        for eval_datum, prediction in zip(eval_dataset, predictions):
            datum = eval_datum.copy()
            datum["prediction"] = prediction
            f.write(json.dumps(datum) + "\n")


if __name__ == "__main__":
    main()
