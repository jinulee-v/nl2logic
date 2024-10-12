from typing import *

from itertools import product
from .prover import prove, read_expr, LogicalExpressionException, Prover9FatalException
from .prover.utils import normalize_predictions, predicates
from tqdm import tqdm as _tqdm
import re

import unicodedata


def single_step_accuracy_corpus(sentences: List[Dict[str, str]], chains: List[Dict[str, Union[str, List[str]]]], tqdm: bool = False):
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

    for chain in _tqdm(chains[:3]) if tqdm else chains: # wrap tqdm if instructed
        print(chain)
        gold_label = chain["label"] # gold label from the dataset
        if gold_label == "neutral":
            continue

        premises = []
        for prem_id in chain["premises"]:
            # Ignore invalid premises
            valid_ps = [p for p in predictions_dict[prem_id] if p["score"] >= 0]
            premises.append(valid_ps)
        conclusion = [c for c in predictions_dict[chain["conclusion"]] if c["score"] >= 0]

        # Run prover
        entailment_cnt = 0
        contradiction_cnt = 0
        successful_pairs = [] # stores deep reference to each prem/conc dicts, so that it only updates at the end
        for prem_conc in product(*premises, conclusion):
            prems = prem_conc[:-1]
            prems_fol = [s["normalized_prediction"] for s in prems]
            conc = prem_conc[-1]
            conc_fol = conc["normalized_prediction"]

            try:
                label, proof = prove(prems_fol, conc_fol, return_proof=True)
            except Prover9FatalException as e:
                continue
            except TimeoutError:
                continue
        
            # Update the score of the prediction if the pair gets correct answer
            if label == gold_label:
                # Check for sprious patterns:
                # 1. If all chain's premises appear in the proof,
                # if set([p["id"] for p in premises_in_proof]) == set(chain["premises"]):
                    # DEBUG
                    # print("============")
                    # for prem in premises_in_proof:
                    #     print(prem["id"], prem["normalized_prediction"])
                    # print(conclusion_fol)
                    # print("============\n")

                # 2. If the conc includes a predicate that does not appear in the prem,
                #    We discard the whole (prems, conc) tuple
                conc_predicates = predicates(read_expr(conc_fol))
                for p in prems_fol:
                    conc_predicates = conc_predicates.difference(predicates(read_expr(p)))
                if len(conc_predicates) > 0:
                    successful_pairs = []
                    break

                # If passed validity check, store the pair so we can update it at the end
                successful_pairs.append((prems, conc))
                if label == "entailment":
                    entailment_cnt += 1
                elif label == "contradiction":
                    contradiction_cnt += 1

        # Update scores
        for prems, conc in successful_pairs:
            # Finally we can say that pair is correct
            for prem in prems:
                prem["score"] += 1
            conc["score"] += 1
            # Sum up for voting
            if label == "entailment":
                entailment_cnt += 1
            elif label == "contradiction":
                contradiction_cnt += 1

        predict_label = "neutral" # overall prediction, default to neutral
        print(entailment_cnt, contradiction_cnt)
        if entailment_cnt > contradiction_cnt and contradiction_cnt >= 0:
            predict_label = "entailment"
        elif contradiction_cnt > entailment_cnt and entailment_cnt >= 0:
            predict_label = "contradiction"
    
        confusion_matrix[gold_label][predict_label] += 1
    
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
    return correct / (correct + wrong), confusion_matrix, predictions_scored


if __name__ == "__main__":
    sentences = [
        {"id": "folio_train_9", "nl": "Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.", "prediction": ["(IsCzech(Miroslav) & IsChoralConductor(Miroslav) & SpecializesIn(Miroslav,Renaissance) & SpecializesIn(Miroslav,Baroque))"]},
        {"id": "folio_train_10", "nl": "Any choral conductor is a musician.", "prediction": ["all x.(IsChoralConductor(x) -> IsMusician(x))"]},
        {"id": "folio_train_11", "nl": "Some musicians love music.", "prediction": ["exists x.(IsMusician(x) -> Loves(x,Music))"]},
        {"id": "folio_train_12", "nl": "Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.", "prediction": ["(IsBook(MethodOfStudyingGregorianChant) & IsAuthorOf(Miroslav,MethodOfStudyingGregorianChant) & PublishedInYear(MethodOfStudyingGregorianChant,Year1946))"]},
        {"id": "folio_train_13", "nl": "Miroslav Venhoda loved music.", "prediction": ["Loves(Miroslav,Music)"]},
        {"id": "folio_train_14", "nl": "A Czech person wrote a book in 1946.", "prediction": ["exists x.(IsCzech(x) & exists y.(IsBook(y) & IsAuthorOf(x,y) & PublishedInYear(y,Year1946)))"]},
        {"id": "folio_train_15", "nl": "No choral conductor specialized in the performance of Renaissance.", "prediction": ["-exists x.(IsChoralConductor(x) & SpecializesIn(x,Renaissance))"]}
    ]
    chains = [
        {"premises": ["folio_train_9", "folio_train_10", "folio_train_11", "folio_train_12"], "conclusion": "folio_train_13", "label": "neutral"},
        {"premises": ["folio_train_9", "folio_train_10", "folio_train_11", "folio_train_12"], "conclusion": "folio_train_14", "label": "entailment"},
        {"premises": ["folio_train_9", "folio_train_10", "folio_train_11", "folio_train_12"], "conclusion": "folio_train_15", "label": "contradiction"},
        {"premises": ["folio_train_9", "folio_train_10", "folio_train_11", "folio_train_12"], "conclusion": "folio_train_15", "label": "entailment"} # wrong
    ]

    print(single_step_accuracy_corpus(sentences, chains)[0]) # 0.75