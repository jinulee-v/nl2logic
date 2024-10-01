from typing import *
import logging

import nltk
from nltk.sem import logic
from nltk.sem import Expression, LogicalExpressionException
from nltk.inference.prover9 import Prover9FatalException
from timeoutcontext import timeout

import subprocess

nltk.Prover9._binary_location = "LADR-2009-11A/bin"

logic._counter._value = 0
read_expr = Expression.fromstring

def prove(premises, conclusion) -> Literal["entailment", "contradict", "neutral"]:
    """Run Prover9 to prove the conclusion (or its hard negation) given the premises.

    :return: _description_
    :rtype: _type_
    """
    # Convert premises, conclusion to symbols
    p_list = []
    symbols = set()
    for p in premises:
        p = read_expr(p)
        p_list.append(p)
        symbols.update(p.predicates())
    c = read_expr(conclusion)
    if len(set(c.predicates()).difference(symbols)) != 0:
        # If conclusion symbols are not completely included in the premises, then it is neutral
        return "neutral"

    with timeout(10):
        command = nltk.Prover9Command(c, p_list)
        truth_value = command.prove()
        # print(command._prover.prover9_input(c, p_list))
    if truth_value:
        logging.info(command.proof())
        return "entailment"
    else:
        neg_c = read_expr("-(" + conclusion + ")")
        with timeout(10):
            command = nltk.Prover9Command(neg_c, p_list)
            negation_true = command.prove()
        if negation_true:
            logging.info(command.proof())
            return "contradiction"
        else:
            return "neutral"


def convert_nltk_format(premises, conclusion) -> Tuple[List[str], str]:
    """
    """
    # Check syntax and parse
    parser = logic.LogicParser()
    p_list = []
    for p in premises:
        p_list.append(str(parser.parse(p)))
    c = str(parser.parse(conclusion))

    return p_list, c
    
