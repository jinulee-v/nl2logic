import json
import re
from utils.metrics import entailment_preserving_rate_corpus

def convert_fol_expression(unicode_fol: str) -> str:
    # Mapping of Unicode symbols to the desired format
    replacements = {
        "\u2200": "all ",      # Universal quantifier
        "\u2203": "exists ",   # Existential quantifier
        "\u2192": "->",       # Implication
        "\u2227": "&",        # Conjunction
        "\u2228": "|",        # Disjunction
        "\u00ac": "-",        # Negation
        "\u2194": "<->",      # Biconditional
    }

    # Replace Unicode symbols with target format equivalents
    for unicode_symbol, target_symbol in replacements.items():
        unicode_fol = unicode_fol.replace(unicode_symbol, target_symbol)
    
    # Add a period after quantifiers followed by a single variable
    unicode_fol = re.sub(r"(all|exists) (\w)", r"\1 \2.", unicode_fol)

    # Ensure spaces after periods in the quantifier sections
    unicode_fol = re.sub(r"\.(\w)", r". \1", unicode_fol)
    
    return unicode_fol


sentences = []
sentence_dict = {}
dataset = "entailmentbank_validation"
# dataset = "eqasc_test"
# dataset = "esnli_test"

# Load sentences with FOL
for i in range(16):
    file_path = f'data/logicllama_outputs/{dataset}_sentences_{i}.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
        for sentence_data in data:
            sentence_id = sentence_data["id"]
            prediction = sentence_data["naive_translate_pred"].split("\n")[-1]
            converted_prediction = convert_fol_expression(prediction)
            
            if sentence_id not in sentence_dict:
                # Add a new entry for this ID
                sentence_dict[sentence_id] = {
                    "id": sentence_id,
                    "nl": sentence_data["NL"],
                    "prediction": [converted_prediction]
                }
            else:
                # Append prediction to the existing entry
                sentence_dict[sentence_id]["prediction"].append(converted_prediction)

sentences = list(sentence_dict.values())

# Load entailment chains
with open(f'data/{dataset}_chains.jsonl', 'r') as f:
    chains = []
    for line in f:
        chain_data = json.loads(line)
        chains.append({
            "premises": chain_data["premises"],
            "conclusion": chain_data["conclusion"],
            "label": chain_data["label"]
        })

# Calculate entailment preserving rate
entailment_rate = entailment_preserving_rate_corpus(sentences, chains, tqdm=True)[0]
print("Entailment Preserving Rate:", entailment_rate)
