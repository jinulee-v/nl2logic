from typing import *

from itertools import product
from .prover import prove, read_expr, LogicalExpressionException, Prover9FatalException
from .prover._rename import rename_predicates

from tqdm import tqdm as _tqdm

import unicodedata

def single_step_accuracy_problem(premises_fol: List[str], conclusion_fol: str, label: Literal["entailment", "contradiction", "neutral"]) -> float:
    return 1 if (prove(premises_fol, conclusion_fol) == label) else 0

def normalize_predictions(predictions: List[str]) -> Tuple[List[str], List[int]]:
    # Deduplicate predictions, but preserve order
    dedup = set()
    predictions = [p for p in predictions if not (p in dedup or dedup.add(p))]

    new_predictions = []
    score = []
    for fol in predictions:
        # Remove accents from the FOL
        fol = unicodedata.normalize('NFKD', fol).encode('ascii', 'ignore').decode('ascii')

        # Syntax check
        try:
            fol_expr = read_expr(fol) # Syntax check
        except LogicalExpressionException as e:
            # Syntax error
            new_predictions.append(fol)
            score.append(-1) # invalid
            continue
            
        # Rename all predicates in fol_expr so that each predicates have arity included in its name
        # e.g. P(x) -> P_1(x), Q(x,y) -> Q_2(x,y)
        fol_expr = rename_predicates(fol_expr)

        new_predictions.append(str(fol_expr))
        score.append(0) # valid
    assert len(new_predictions) == len(score)
    return new_predictions, score # Leave only the valid FOLs

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
    for s in sentences:
        sentences_dict[s["id"]] = s
        # Deduplicate and leave only the syntactically valid FOLs.
        predictions, predictions_score = normalize_predictions(s["prediction"])
        # add prediction/score dict
        predictions_dict[s["id"]] = [
            {
                "id": s["id"],
                "prediction": p,
                "score": sc
            } for p, sc in zip(predictions, predictions_score)
        ]
    
    # evaluate per example -> accuracy and confusion matrix
    labels = ["entailment", "contradiction", "neutral"]
    confusion_matrix = {
        gold: {
            predict: 0 for predict in labels
        } for gold in labels
    }

    for c in _tqdm(chains) if tqdm else chains: # wrap tqdm if instructed
        gold_label = c["label"] # gold label from the dataset
        premises = []
        for prem_id in c["premises"]:
            # Ignore invalid premises
            premises.append([p for p in predictions_dict[prem_id] if p["score"] >= 0])

        prem_conc_pairs = product(*premises, predictions_dict[c["conclusion"]])

        entailment_cnt = 0
        contradiction_cnt = 0

        for pair in prem_conc_pairs:
            # Ignore invalid conclusions
            if pair[-1]["score"] < 0:
                continue

            # Run prover
            premises_fol, conclusion_fol = [p["prediction"] for p in pair[:-1]], pair[-1]["prediction"]

            try:
                label = prove(premises_fol, conclusion_fol)
            except Prover9FatalException as e:
                continue
            except TimeoutError:
                continue

            # Sum up for voting
            if label == "entailment":
                entailment_cnt += 1
            elif label == "contradiction":
                contradiction_cnt += 1
            
            # Update the score of the prediction if the pair gets correct answer
            if label == gold_label:
                for p in pair:
                    p["score"] += 1

        predict_label = "neutral" # overall prediction, default to neutral
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
    predictions_scored.sort(key=lambda x: x[0]["id"])
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