from typing import *

from itertools import product
from .prover import prove, read_expr, LogicalExpressionException, VampireFatalException
from .prover.utils import normalize_predictions, predicates
from .graph import entailment_graph
from tqdm import tqdm as _tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

import os
EPR_EVAL_N_WORKERS = os.environ.get("EPR_EVAL_N_WORKERS", "1")

def evaluate_single_pair(prem_conc, chain):
    prems = prem_conc[:-1]
    prems_fol = [s["normalized_prediction"] for s in prems]
    conc = prem_conc[-1]
    conc_fol = conc["normalized_prediction"]

    try:
        label, proof = prove(prems_fol, conc_fol, return_proof=True)
    except VampireFatalException as e:
        print("ERROR", e)
        return None, 0, 0
    except TimeoutError:
        return None, 0, 0

    # Update the score of the prediction if the pair gets correct answer
    if proof is None:
        # label is neutral and model got it right
        return (prems, conc), 0, 0
    if label == chain["label"]:
        # Check for sprious patterns:
        # 1. If all chain's premises appear in the proof,
        if proof.count("[input]") != len(chain["premises"]) + 1:
            # DEBUG
            # print("============")
            # for prem in premises_in_proof:
            #     print(prem["id"], prem["normalized_prediction"])
            # print(conclusion_fol)
            # print("============\n")
            return None, 0, 0

        # 2. If the conc includes a predicate that does not appear in the prem,
        #    We discard the whole (prems, conc) tuple
        # conc_predicates = predicates(read_expr(conc_fol))
        conc_predicates = predicates(conc_fol)
        for p in prems_fol:
            # conc_predicates = conc_predicates.difference(predicates(read_expr(p)))
            conc_predicates = conc_predicates.difference(predicates(p))
        if len(conc_predicates) > 0:
            return None, 0, 0

        # If passed validity check, store the pair so we can update it at the end
        if label == "entailment":
            return (prems, conc), 1, 0
        elif label == "contradiction":
            return (prems, conc), 0, 1
    return None, 0, 0

def entailment_preserving_rate_corpus(sentences: List[Dict[str, str]], chains: List[Dict[str, Union[str, List[str]]]], tqdm: bool = False, graph_filename: str = None) -> Tuple[float, Dict[str, Dict[str, int]], List[Dict[str, str]]]:
    """_summary_

    :param sentences: [{"id": "___", "fol": "___"}]
    :type sentences: List[Dict[str, str]]
    :param chains: [{"premises": [], "conclusion": "___", label: "entailment"}]
    :type chains: List[Dict[str, Union[str, List[str]]]]
    :return: micro-F1, confusion-matrix, predictions-dict
    """
    # create sentence dict
    sentences_dict = {}
    predictions_dict = {}
    for s in _tqdm(sentences) if tqdm else sentences:
        sentences_dict[s["id"]] = s
        # Deduplicate and leave only the syntactically valid FOLs.
        predictions, normalized_predictions, predictions_score = normalize_predictions(s["prediction"])
        predictions_dict[s["id"]] = [
            {
                "id": s["id"],
                "prediction": p,
                "normalized_prediction": np,
                "score": sc
            } for p, np, sc in zip(predictions, normalized_predictions, predictions_score)
        ]
    
    # evaluate per example -> accuracy and confusion matrix
    labels = ["entailment", "contradiction", "neutral"]
    confusion_matrix = {
        gold: {
            predict: 0 for predict in labels
        } for gold in labels
    }

    for chain in _tqdm(chains) if tqdm else chains: # wrap tqdm if instructed
        gold_label = chain["label"] # gold label from the dataset
        # if gold_label == "neutral":
        #     continue

        premises = []
        for prem_id in chain["premises"]:
            # Ignore invalid premises
            valid_ps = [p for p in predictions_dict[prem_id] if p["score"] >= 0]
            premises.append(valid_ps)
        conclusion = [c for c in predictions_dict[chain["conclusion"]] if c["score"] >= 0]

        # Run prover
        execution_results = []
        if EPR_EVAL_N_WORKERS != "1":
            with ThreadPoolExecutor(max_workers=EPR_EVAL_N_WORKERS) as executor:
                futures = [
                    executor.submit(evaluate_single_pair, prem_conc, chain)
                    for prem_conc in product(*premises, conclusion)
                ]
                for future in as_completed(futures):
                    execution_results.append(future.result())
        else:
            for prem_conc in product(*premises, conclusion):
                execution_results.append(evaluate_single_pair(prem_conc, chain))

        # Update scores
        entailment_cnt = 0
        contradiction_cnt = 0
        for prem_conc, ent, cont in execution_results:
            if prem_conc is not None and (ent or cont):
                # Finally we can say that pair is correct
                prems, conc = prem_conc
                for prem in prems:
                    prem["score"] += 1
                conc["score"] += 1
            # Sum up for voting
            entailment_cnt += ent
            contradiction_cnt += cont

        predict_label = "neutral" # overall prediction, default to neutral
        if entailment_cnt > contradiction_cnt and contradiction_cnt >= 0:
            predict_label = "entailment"
        elif contradiction_cnt > entailment_cnt and entailment_cnt >= 0:
            predict_label = "contradiction"
        confusion_matrix[gold_label][predict_label] += 1
        chain["correct"] = int(gold_label == predict_label)
    
    # Calculate (micro) F1 score
    correct = 0
    wrong = 0
    for gold in labels:
        for predict in labels:
            if gold == predict:
                correct += confusion_matrix[gold][predict]
            else:
                wrong += confusion_matrix[gold][predict]
    
    predictions_scored = list(predictions_dict.values())
    # predictions_scored.sort(key=lambda x: x[0]["id"])

    # Generate entailment graph
    if graph_filename is not None:
        entailment_graph(chains, output_file=graph_filename)

    return correct / (correct + wrong), confusion_matrix, predictions_scored


if __name__ == "__main__":
    sentences = [
        {"id": "entailmentbank_train_0", "nl": "", "prediction": ["all x.(Leo(x) -> Constellation(x))"]},
        {"id": "entailmentbank_train_1", "nl": "", "prediction": ["all x.(Constellation(x) -> ContainsStars(x))"]},
        {"id": "entailmentbank_train_2", "nl": "", "prediction": ["all x.(Leo(x) -> (Constellation(x) & ContainsStars(x)))"]},
        {"id": "entailmentbank_train_3", "nl": "", "prediction": ["all x.(Leo(x) -> ConstellationWithStars(x))"]},
    ]
    chains = [
        {"premises": ["entailmentbank_train_0", "entailmentbank_train_1"], "conclusion": "entailmentbank_train_1", "label": "entailment"}, # only need one premise
        {"premises": ["entailmentbank_train_0", "entailmentbank_train_1"], "conclusion": "entailmentbank_train_2", "label": "entailment"}, # correct
        {"premises": ["entailmentbank_train_0", "entailmentbank_train_1"], "conclusion": "entailmentbank_train_3", "label": "entailment"}, # cannot entail
    ]

    print(entailment_preserving_rate_corpus(sentences, chains)[0]) # 0.33