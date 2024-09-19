from typing import *

from .prover import prove

def single_step_accuracy_problem(premises_fol: List[str], conclusion_fol: str, label: Literal["entailment", "contradiction", "neutral"]) -> float:
    return 1 if (prove(premises_fol, conclusion_fol) == label) else 0

def single_step_accuracy_corpus(sentences: List[Dict[str, str]], chains: List[Dict[str, Union[str, List[str]]]]) -> Tuple[float, Dict[str, Dict[str, int]]]:
    """_summary_

    :param sentences: [{"id": "___", "fol": "___"}]
    :type sentences: List[Dict[str, str]]
    :param chains: [{"premises": [], "conclusion": "___", label: "entailment"}]
    :type chains: List[Dict[str, Union[str, List[str]]]]
    :return: micro-F1, confusion-matrix
    :rtype: Tuple[float, float, float, Dict[str, Dict[str, int]]]
    """
    # create sentence dict
    sentences_dict = {}
    for s in sentences:
        sentences_dict[s["id"]] = s["fol"]
    
    # evaluate per example -> accuracy and confusion matrix
    labels = ["entailment", "contradiction", "neutral"]
    confusion_matrix = {
        gold: {
            predict: 0 for predict in labels
        } for gold in labels
    }
    for c in chains:
        premises_fol = [sentences_dict[p] for p in c["premises"]]
        conclusion_fol = sentences_dict[c["conclusion"]]
        gold_label = c["label"]
        predict_label = prove(premises_fol, conclusion_fol)
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
    
    return correct / (correct + wrong), confusion_matrix


if __name__ == "__main__":
    sentences = [
        {"id": "folio_train_9", "nl": "Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.", "fol": "(IsCzech(Miroslav) & IsChoralConductor(Miroslav) & SpecializesIn(Miroslav,Renaissance) & SpecializesIn(Miroslav,Baroque))"},
        {"id": "folio_train_10", "nl": "Any choral conductor is a musician.", "fol": "all x.(IsChoralConductor(x) -> IsMusician(x))"},
        {"id": "folio_train_11", "nl": "Some musicians love music.", "fol": "exists x.(IsMusician(x) -> Loves(x,Music))"},
        {"id": "folio_train_12", "nl": "Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.", "fol": "(IsBook(MethodOfStudyingGregorianChant) & IsAuthorOf(Miroslav,MethodOfStudyingGregorianChant) & PublishedInYear(MethodOfStudyingGregorianChant,Year1946))"},
        {"id": "folio_train_13", "nl": "Miroslav Venhoda loved music.", "fol": "Loves(Miroslav,Music)"},
        {"id": "folio_train_14", "nl": "A Czech person wrote a book in 1946.", "fol": "exists x.(IsCzech(x) & exists y.(IsBook(y) & IsAuthorOf(x,y) & PublishedInYear(y,Year1946)))"},
        {"id": "folio_train_15", "nl": "No choral conductor specialized in the performance of Renaissance.", "fol": "-exists x.(IsChoralConductor(x) & SpecializesIn(x,Renaissance))"}
    ]
    chains = [
        {"premises": ["folio_train_9", "folio_train_10", "folio_train_11", "folio_train_12"], "conclusion": "folio_train_13", "label": "neutral"},
        {"premises": ["folio_train_9", "folio_train_10", "folio_train_11", "folio_train_12"], "conclusion": "folio_train_14", "label": "entailment"},
        {"premises": ["folio_train_9", "folio_train_10", "folio_train_11", "folio_train_12"], "conclusion": "folio_train_15", "label": "contradiction"},
        {"premises": ["folio_train_9", "folio_train_10", "folio_train_11", "folio_train_12"], "conclusion": "folio_train_15", "label": "entailment"} # wrong
    ]

    print(single_step_accuracy_corpus(sentences, chains)[0]) # 0.75