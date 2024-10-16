# utils.py
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

import nltk
read_expr = nltk.sem.logic.Expression.fromstring
nltk.Prover9._binary_location = "../LADR-2009-11A/bin"

def save_model(model, tokenizer, path):
    """
    Save the model and tokenizer to the specified path.
    """
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def load_model_and_tokenizer(model_name, model_dir=None):
    """Load the fine-tuned model and tokenizer."""
    if model_dir:
        # Load fine-tuned model and tokenizer from the specified directory
        model = T5ForConditionalGeneration.from_pretrained(model_dir)
        tokenizer = T5Tokenizer.from_pretrained(model_dir)
    else:
        # Load the base pre-trained model and tokenizer
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)

    return model, tokenizer


def set_seed(seed):
    """
    Set seed for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_accuracy(predictions, labels, fasteval=True):
    """Compute simple accuracy based on exact match between prediction and label."""
    correct = 0
    total = len(predictions)
    
    for pred, label in zip(predictions, labels):
        if fasteval:
            if pred.strip() == label.strip():
                correct += 1
        else:
            try:
                pred = read_expr(pred)
                label = read_expr(label)
                if pred.equiv(label):
                    correct += 1
            except:
                pass
    
    accuracy = correct / total
    return accuracy