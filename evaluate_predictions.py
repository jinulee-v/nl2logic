import argparse
import json

import sys; sys.path.append('.')
from utils.metrics import single_step_accuracy_corpus
from utils.prover import read_expr

def main(args):
    sentences = []
    chains = []
    with open(args.sentence_data, "r") as f:
        for line in f:
            sentences.append(json.loads(line))
            # DEBUG
            sentences[-1]["prediction"] = sentences[-1]["prediction"]
    with open(args.chain_data, "r") as f:
        for line in f:
            chains.append(json.loads(line))

    f1, confusion_matrix, predictions = single_step_accuracy_corpus(sentences, chains, tqdm=True)
    print(f"F1 score: {f1}")
    print(f"Confusion matrix: {confusion_matrix}")
    print(json.dumps(predictions[0], indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_prefix", type=str, help="Prefix of the run files. If provided, sentence_data and chain_data are ignored.")
    parser.add_argument("--sentence_data", type=str, default=None, help="Path to the run sentences file.")
    parser.add_argument("--chain_data", type=str, default=None, help="Path to the run chains file.")

    args = parser.parse_args()

    if args.data_prefix is None:
        assert args.sentence_data is not None
        assert args.chain_data is not None
    else:
        args.sentence_data = f"{args.data_prefix}_sentences.jsonl"
        args.chain_data = f"{args.data_prefix}_chains.jsonl"

    main(args)
        
        