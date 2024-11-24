import argparse
import json

import sys; sys.path.append('.')
from utils.metrics import entailment_preserving_rate_corpus
from utils.prover import read_expr

def main(args):
    sentences = []
    chains = []
    with open(args.sentence_data, "r") as f:
        for line in f:
            sentences.append(json.loads(line))
            # DEBUG
            sentences[-1]["prediction"] = [sentences[-1]["prediction"][0]]
    with open(args.chain_data, "r") as f:
        for line in f:
            chains.append(json.loads(line))

    if args.output_graph:
        graph_filename = args.sentence_data.replace("_sentences.jsonl", "_entailment_graph.png")
    else:
        graph_filename = None
    f1, confusion_matrix, predictions = entailment_preserving_rate_corpus(sentences, chains, tqdm=True, graph_filename=graph_filename)
    print(f"F1 score: {f1}")
    print(f"Confusion matrix: {confusion_matrix}")
    cnt = 0; tot = 0
    connected_sentences = set() # set of sentence ids that has at least one prediction that is connected
    for sentence in predictions:
        for pred in sentence:
            tot += 1
            if pred["score"] > 0:
                cnt += 1
                connected_sentences.add(pred["id"])
                # print(f"{cnt:>4}:", q["prediction"], " => connected in", q["score"], "chains")
    print("Connected sentences:", len(connected_sentences))
    print("Total sentences:", len(sentences))
    print("Connected sentence ratio:", len(connected_sentences) / len(sentences))
    print("Connected predictions:", cnt)
    print("Total predictions:", tot)
    print("Connected prediction ratio:", cnt / tot)

    # Store the results
    with open(args.output_prefix + "_entailment_preserving_rate_eval_top1.json", "w") as f:
        json.dump({
            "f1": f1,
            "confusion_matrix": confusion_matrix,
            "connectivity": {
                "total": len(sentences),
                "connected": len(connected_sentences),
                "ratio": len(connected_sentences) / len(sentences)
            }
        }, f, ensure_ascii=False, indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_prefix", type=str, help="Prefix of the run files. If provided, sentence_data and chain_data are ignored.")
    parser.add_argument("--sentence_data", type=str, default=None, help="Path to the run sentences file.")
    parser.add_argument("--chain_data", type=str, default=None, help="Path to the run chains file.")
    parser.add_argument("--output_prefix", type=str, default=None, help="Path to the output file. Defaults to sentence_data with suffix `_sentences.jsonl` removed")
    parser.add_argument("--output_graph", action="store_true", help="If true, output graph file.")

    args = parser.parse_args()

    if args.data_prefix is None:
        assert args.sentence_data is not None
        assert args.chain_data is not None
        args.data_prefix = args.sentence_data.replace("_sentences.jsonl", "")
    else:
        args.sentence_data = f"{args.data_prefix}_sentences.jsonl"
        args.chain_data = f"{args.data_prefix}_chains.jsonl"
    if args.output_prefix is None:
        args.output_prefix = args.sentence_data.replace("_sentences.jsonl", "")

    main(args)
        
        