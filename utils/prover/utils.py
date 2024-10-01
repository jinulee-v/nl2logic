from typing import *
import unicodedata

from nltk.sem.logic import Expression, LogicalExpressionException
read_expr = Expression.fromstring

from ._rename import rename_predicates

def normalize_predictions(predictions: List[str]) -> Tuple[List[str], List[int]]:
    # Deduplicate predictions, but preserve order
    dedup = set()
    predictions = [p for p in predictions if not (p in dedup or dedup.add(p))]

    normalized_predictions = []
    score = []
    for fol in predictions:
        # Remove accents from the FOL
        fol = unicodedata.normalize('NFKD', fol).encode('ascii', 'ignore').decode('ascii')

        # Syntax check
        try:
            fol_expr = read_expr(fol) # Syntax check
        except LogicalExpressionException as e:
            # Syntax error
            normalized_predictions.append(fol)
            score.append(-1) # invalid
            continue
            
        # Rename all predicates in fol_expr so that each predicates have arity included in its name
        # e.g. P(x) -> P_1(x), Q(x,y) -> Q_2(x,y)
        fol_expr = rename_predicates(fol_expr)

        normalized_predictions.append(str(fol_expr))
        score.append(0) # valid
    assert len(normalized_predictions) == len(score)
    return predictions, normalized_predictions, score # Leave only the valid FOLs