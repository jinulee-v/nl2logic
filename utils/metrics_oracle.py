from typing import *

from itertools import product
from .prover import prove, read_expr, LogicalExpressionException, VampireFatalException
from .prover.utils import normalize_predictions, predicates
from .graph import entailment_graph
from tqdm import tqdm as _tqdm
import subprocess

from concurrent.futures import ThreadPoolExecutor, as_completed

import os
EPR_EVAL_N_WORKERS = os.environ.get("EPR_EVAL_N_WORKERS", "1")
CLINGO_PATH = os.environ.get("CLINGO_PATH", "clingo")

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

def entailment_preserving_rate_oracle(sentences: List[Dict[str, str]], chains: List[Dict[str, Union[str, List[str]]]], tqdm: bool = False, graph_filename: str = None) -> Tuple[float, Dict[str, Dict[str, int]], List[Dict[str, str]]]:
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
    for s in _tqdm(sentences):
        sentences_dict[s["id"]] = s
        # Deduplicate and leave only the syntactically valid FOLs.
        predictions, normalized_predictions, predictions_score = normalize_predictions(s["prediction"])
        predictions_dict[s["id"]] = [
            {
                "id": s["id"],
                "output_id": i,
                "prediction": p,
                "normalized_prediction": np,
                "score": sc
            } for i, (p, np, sc) in enumerate(zip(predictions, normalized_predictions, predictions_score))
        ]
    
    # evaluate per example -> accuracy and confusion matrix
    labels = ["entailment", "contradiction", "neutral"]
    confusion_matrix = {
        gold: {
            predict: 0 for predict in labels
        } for gold in labels
    }

    successful_pairs = []

    for i, chain in enumerate(_tqdm(chains)) if tqdm else chains: # wrap tqdm if instructed
        gold_label = chain["label"] # gold label from the dataset
        chain["id"] = i
        if gold_label == "neutral":
            continue

        premises = []
        for prem_id in chain["premises"]:
            # Ignore invalid premises
            valid_ps = [p for p in predictions_dict[prem_id] if p["score"] >= 0]
            premises.append(valid_ps)
        conclusion = [c for c in predictions_dict[chain["conclusion"]] if c["score"] >= 0]

        # Run prover
        execution_results = []
        for prem_conc in product(*premises, conclusion):
            execution_results.append(evaluate_single_pair(prem_conc, chain))

        # Update scores
        for prem_conc, ent, cont in execution_results:
            if prem_conc is not None:
                # Finally we can say that pair is correct
                prems, conc = prem_conc
                for prem in prems:
                    prem["score"] += 1
                conc["score"] += 1
            # Sum up for voting
            if ent and gold_label == "entailment":
                successful_pairs.append((prem_conc, chain))
            elif cont and gold_label == "contradiction":
                successful_pairs.append((prem_conc, chain))

    # Construct clingo input
    """
% Define the range of a and b atoms
a(1, 1..3).
a(2, 1..2).
a(3, 1..3).
b(1..2).

% Each group of a variables can only have one true value
1 { select(a(S, N)) : a(S, N)} 1 :- a(S, X).

% Define conditions for b1 and b2
true(b(1)) :- select(a(1, 1)), select(a(2, 1)).
true(b(1)) :- select(a(1, 3)), select(a(2, 2)).
true(b(2)) :- select(a(2, 1)), select(a(3, 2)).

% Maximize the number of true b values
#maximize { 1, X : true(b(X)) }.

#show select/1.
#show true/1.
    """
    asp_sents = set()
    asp_chains = set()
    asp_chains_success = set()
    for prem_conc, chain in successful_pairs:
        if prem_conc is None:
            continue
        prem, conc = prem_conc
        for sent in list(prem) + [conc]:
            asp_sents.add(f'a("{sent["id"]}", {sent["output_id"]}).\n')
        asp_chains.add(f'b({chain["id"]}).\n')
        conds = " , ".join([f"select(a(\"{sent['id']}\", {sent['output_id']}))" for sent in list(prem) + [conc]])
        asp_chains_success.add(f'true(b({chain["id"]})) :- {conds}.\n')
        
    clingo_input = ""
    clingo_input += "".join(asp_sents) + "\n"
    clingo_input += "".join(asp_chains) + "\n"
    clingo_input += "".join(asp_chains_success) + "\n"
    clingo_input += """
% only select one from each sentences.
1 { select(a(S, N)) : a(S, N)} 1 :- a(S, X).

% Maximize number of true b's.
#maximize { 1, X : true(b(X)) }.
#show select/1.
#show true/1.
"""
    print(clingo_input)
    # print(clingo_input); exit()
    process = subprocess.Popen(
        [CLINGO_PATH, '--quiet=1', '--opt-mode=opt', "--restart-on-model", "--time-limit=600"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    clingo_output, _ = process.communicate(input=clingo_input)
    clingo_output = [sent for sent in clingo_output.split("\n") if sent.startswith("true")][0]
    clingo_output = clingo_output.split(" ")
    clingo_output = [x.replace("true(", "").replace("select(", "")[:-1] for x in clingo_output] # remove true(XXX) to XXX
    # print(clingo_output); exit()

    # Update scores
    correct = 0
    total = len(chains)
    for x in clingo_output:
        # print(x)
        if x.startswith("a("):
            # selected translations
            x = x.replace("a(", "").replace(")", "").split(",")
            predictions_dict[x[0].replace('"', '')][int(x[1])]["selected"] = True
        else:
            # selected chains
            correct += 1
    
    predictions_scored = list(predictions_dict.values())
    predictions_scored.sort(key=lambda x: x[0]["id"])

    # # Generate entailment graph
    # if graph_filename is not None:
    #     entailment_graph(chains, output_file=graph_filename)

    return correct / (total), None, predictions_scored



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