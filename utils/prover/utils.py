from typing import *
import unicodedata

from nltk.sem.logic import Expression, LogicalExpressionException
read_expr = Expression.fromstring

from ._rename import rename_predicates

from nltk.sem.logic import *

def predicates(expression) -> Set[str]:
    # print(expression, type(expression))
    if isinstance(expression, ApplicationExpression):
        # ApplicationExpression is curried form; (f(x))(y)
        # We assume that functions are not nested, i.e. f(g(y)) is not allowed
        e = expression
        arity = 0
        while hasattr(e, 'function') and e.function:
            # find innermost function
            arity += 1
            e = e.function
        return set(e.variable.name)
    elif isinstance(expression, BinaryExpression):
        return predicates(expression.first).union(predicates(expression.second))
    elif isinstance(expression, list):
        # Recursively apply to list elements (like in conjunctions, etc.)
        result = set()
        for subexp in expression:
            result = result.union(predicates(subexp))
        return result
    else:
        # For non-variable expressions (Quantifiers, etc.), recurse down
        if hasattr(expression, 'term'):
            return predicates(expression.term)
        else:
            return set()
            raise ValueError(f"Invalid expression type: {type(expression)}")

def normalize_predictions(predictions: List[str]) -> Tuple[List[Expression], List[int]]:
    # Deduplicate predictions, but preserve order
    dedup = set()
    predictions = [p for p in predictions if not (p in dedup or dedup.add(p))]

    normalized_predictions = []
    score = []
    for fol in predictions:
        # Syntax check
        try:
            # Remove accents from the FOL
            fol = unicodedata.normalize('NFKD', fol).encode('ascii', 'ignore').decode('ascii')
            fol_expr = read_expr(fol) # Syntax check in NLTK side
        except (LogicalExpressionException, TypeError):
            # Syntax error
            normalized_predictions.append(None)
            score.append(-1) # invalid
            continue
            
        # Rename all predicates in fol_expr so that each predicates have arity included in its name
        # e.g. P(x) -> P_1(x), Q(x,y) -> Q_2(x,y)
        fol_expr = rename_predicates(fol_expr)

        normalized_predictions.append(fol_expr)
        score.append(0) # valid
    assert len(normalized_predictions) == len(score)
    return predictions, normalized_predictions, score # Leave only the valid FOLs