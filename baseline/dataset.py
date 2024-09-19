# dataset.py
from torch.utils.data import Dataset
from nltk.sem import logic
import re

def preprocess_fol(fol_expression):
    """
    Replace special symbols in FOL expressions with the specific boolean, equality,
    and binding operators, ensuring proper formatting for NLTK logic.
    """
    # Replace boolean operators
    fol_expression = fol_expression.replace('∧', '&')  # conjunction
    fol_expression = fol_expression.replace('∨', '|')  # disjunction
    fol_expression = fol_expression.replace('→', '->') # implication
    fol_expression = fol_expression.replace('↔', '<->') # equivalence
    fol_expression = fol_expression.replace('¬', '-')  # negation

    # Replace equality predicates
    fol_expression = fol_expression.replace('=', '=')  # equality remains the same
    fol_expression = fol_expression.replace('≠', '!=')  # inequality

    # Replace binding operators (quantifiers)
    fol_expression = fol_expression.replace('λ', '\\')  # lambda operator

    # Replace ∃ (existential quantifier) followed by a variable
    fol_expression = re.sub(r'∃([a-zA-Z])', r'exists \1.', fol_expression)
    
    # Replace ∀ (universal quantifier) followed by a variable
    fol_expression = re.sub(r'∀([a-zA-Z])', r'all \1.', fol_expression)

    return fol_expression

class MallsDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.data, self.num_removed = self.load_malls_data(dataset)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_malls_data(self, dataset):
        """
        Load MALLS data and convert the FOL part to NLTK inference format.
        Removes any entries that contain unsupported names.
        """
        converted_data = []
        num_removed = 0

        for entry in dataset:
            fol_expression = entry["FOL"]
            natural_language = entry["NL"]

            # Remove entries with unsupported names
            # TODO: Convert the exclusive or symbol ⊕ by A ⊕ B ≡ (A ∨ B) ∧ ¬(A ∧ B)
            if '⊕' in fol_expression or '.' in fol_expression or '-' in fol_expression\
                or 'still life' in fol_expression:
                num_removed += 1
                continue  # Skip this entry
            
            # Pre-process FOL expression to replace special symbols
            fol_expression = preprocess_fol(fol_expression)

            # Convert FOL to NLTK logic format
            try:
                nltk_fol = logic.Expression.fromstring(fol_expression)
            except logic.LogicalExpressionException as e:
                print(f"Failed to parse FOL: {fol_expression}")
                raise e

            converted_data.append({
                "input": natural_language,
                "output": str(nltk_fol)
            })
        
        print('number removed: ', num_removed)
        return converted_data, num_removed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_entry = self.data[idx]
        inputs = self.tokenizer(
            data_entry['input'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        labels = self.tokenizer(
            data_entry['output'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = inputs.input_ids.squeeze()
        attention_mask = inputs.attention_mask.squeeze()
        labels_ids = labels.input_ids.squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_ids
        }
