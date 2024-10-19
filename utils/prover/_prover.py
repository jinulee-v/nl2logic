from typing import *
import logging

import nltk
from nltk.sem.logic import Expression, NegatedExpression, LogicalExpressionException
from timeoutcontext import timeout

from .vampire import run_vampire

read_expr = Expression.fromstring

def prove(premises, conclusion, return_proof=False) -> Literal["entailment", "contradict", "neutral"]:
    """Run Vampire to prove the conclusion (or its hard negation) given the premises.

    :return: _description_
    :rtype: _type_
    """
    # Convert premises, conclusion to symbols
    p_list = []
    symbols = set()
    for p in premises:
        # p = read_expr(p)
        p_list.append(p)
        symbols.update(p.predicates())
    # c = read_expr(conclusion)
    c = conclusion
    if len(set(c.predicates()).difference(symbols)) != 0:
        # If conclusion symbols are not completely included in the premises, then it is neutral
        return "neutral", None

    truth_value, proof = run_vampire(p_list, c)
    if truth_value:
        logging.debug(proof)
        if return_proof:
            return "entailment", proof
        return "entailment"
    else:
        # neg_c = read_expr("-(" + conclusion + ")")
        neg_c = NegatedExpression(c)
        negation_true, proof = run_vampire(p_list, neg_c)
        if negation_true:
            logging.debug(proof)
            if return_proof:
                return "contradiction", proof
            return "contradiction"
        else:
            if return_proof:
                return "neutral", None
            return "neutral"

def equiv(s1: str, s2: str):
    new_expr = read_expr(f"({s1}) <-> ({s2})")
    
    truth_value, _ = run_vampire([], new_expr)

    return truth_value