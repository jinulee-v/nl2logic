from typing import *
import subprocess
import os

from nltk.sem.logic import Expression
from .nltk_to_tptp import _convert_to_tptp

class VampireFatalException(RuntimeError):
    def __init__(self, msg):
        super(VampireFatalException, self).__init__(msg)

VAMPIRE_PATH = os.environ.get("VAMPIRE_PATH")
if not VAMPIRE_PATH:
    VAMPIRE_PATH = "./vampire" # default

def run_vampire(premises: List[Expression], conclusion: Expression):
    input_str = ""

    for p in premises:
        input_str += f"fof(sos, axiom, {_convert_to_tptp(p)}).\n"
    input_str += f"fof(goals, conjecture, {_convert_to_tptp(conclusion)}).\n"

    process = subprocess.Popen([VAMPIRE_PATH], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    proof, _ = process.communicate(input_str) # ignore stdout

    if "Error on line" in proof:
        print(input_str)
        raise VampireFatalException("\n" + proof)
    return "Termination reason: Refutation" in proof, proof

if __name__ == "__main__":
    read_expr = Expression.fromstring
    print(run_vampire(
        [
            read_expr("all x (P(x) -> Q(x))"),
            read_expr("all x (Q(x) -> R(x))")
        ],
        read_expr("all x (P(x) -> (Q(x) & R(x)))")
    )[0])
    print(run_vampire(
        [],
        read_expr("(F(c) & G(c)) <-> (G(c) & F(c))")
    )[0])
    print(run_vampire(
        [
            read_expr("all x.(Leo(x) -> Constellation(x))"),
            read_expr("all x.(Constellation(x) -> ContainsStars(x))"),
        ],
        read_expr("all x.(Leo(x) -> (Constellation(x) & ContainsStars(x)))")
    )[0])