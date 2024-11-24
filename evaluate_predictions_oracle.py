import argparse
import json

import sys; sys.path.append('.')
from utils.metrics_oracle import entailment_preserving_rate_oracle
from utils.prover import read_expr

def main(args):
    sentences = []
    chains = []
    with open(args.sentence_data, "r") as f:
        for line in f:
            sentences.append(json.loads(line))
            # DEBUG
            # sentences[-1]["prediction"] = [sentences[-1]["prediction"][0]]
    with open(args.chain_data, "r") as f:
        for line in f:
            chains.append(json.loads(line))
        # DEBUG
        # chains = chains[:30]

    if args.output_graph:
        graph_filename = args.sentence_data.replace("_sentences.jsonl", "_entailment_graph.png")
    else:
        graph_filename = None
    f1, confusion_matrix, predictions = entailment_preserving_rate_oracle(sentences, chains, tqdm=True, graph_filename=graph_filename)
    print(f"F1 score: {f1}")
    print(f"Confusion matrix: {confusion_matrix}")

    # Store the results
    with open(args.data_prefix + "_entailment_preserving_rate_eval_meta_oracle.json", "w") as f:
        json.dump({
            "f1": f1,
            "confusion_matrix": confusion_matrix
        }, f, ensure_ascii=False, indent=4)
    
    with open(args.data_prefix + "_entailment_preserving_rate_eval_oracle.jsonl", "w") as f:
        for p in predictions:
            for q in p:
                del q["normalized_prediction"]
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_prefix", type=str, help="Prefix of the run files. If provided, sentence_data and chain_data are ignored.")
    parser.add_argument("--sentence_data", type=str, default=None, help="Path to the run sentences file.")
    parser.add_argument("--chain_data", type=str, default=None, help="Path to the run chains file.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to the output file. Defaults to sentence_data, but suffix `_sentence` changed to `_entailment_preserving_rate_eval`")
    parser.add_argument("--output_graph", action="store_true", help="If true, output graph file.")

    args = parser.parse_args()

    if args.data_prefix is None:
        assert args.sentence_data is not None
        assert args.chain_data is not None
        args.data_prefix = args.sentence_data.replace("_sentences.jsonl", "")
    else:
        args.sentence_data = f"{args.data_prefix}_sentences.jsonl"
        args.chain_data = f"{args.data_prefix}_chains.jsonl"
    if args.output_file is None:
        args.output_file = args.sentence_data.replace("_sentences.jsonl", "_entailment_preserving_rate_eval.jsonl")

    main(args)
        
        