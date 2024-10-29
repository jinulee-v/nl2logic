import json
from utils.metrics import entailment_preserving_rate_corpus

# Load sentences with FOL
with open('baseline/entailmentbank_validation_sentences_with_fol_batch.jsonl', 'r') as f:
    sentences = []
    for line in f:
        sentence_data = json.loads(line)
        if not 'prediction' in sentence_data:
            continue
        sentences.append({
            "id": sentence_data["id"],
            "nl": sentence_data["nl"],
            "prediction": sentence_data["prediction"]
        })

# Load entailment chains
with open('baseline/entailmentbank_validation_chains.jsonl', 'r') as f:
    chains = []
    for line in f:
        chain_data = json.loads(line)
        chains.append({
            "premises": chain_data["premises"],
            "conclusion": chain_data["conclusion"],
            "label": chain_data["label"]
        })

# Calculate entailment preserving rate
entailment_rate = entailment_preserving_rate_corpus(sentences, chains)[0]
print("Entailment Preserving Rate:", entailment_rate)
