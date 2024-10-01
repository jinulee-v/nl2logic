from nltk.sem.logic import *

def rename_predicates(expression):
    # print(expression, type(expression))
    if isinstance(expression, ApplicationExpression):
        # ApplicationExpression is curried form; (f(x))(y)
        # We assume that functions are not nested, i.e. f(g(y)) is not allowed
        e = expression
        arity = 0
        while hasattr(e, 'function') and e.function:
            arity += 1
            e = e.function
        # Rename the predicate with its arity
        # e is a FunctionVariableExpression
        e.variable = Variable(f"{e.variable.name}_{arity}")
        return expression
    elif isinstance(expression, VariableBinderExpression):
        return expression.__class__(expression.variable, rename_predicates(expression.term))
    elif isinstance(expression, BinaryExpression):
        return expression.__class__(rename_predicates(expression.first), rename_predicates(expression.second))
    elif isinstance(expression, Expression):
        # For non-variable expressions (Quantifiers, etc.), recurse down
        if hasattr(expression, 'term'):
            return expression.__class__(rename_predicates(expression.term))
        else:
            return expression
    elif isinstance(expression, list):
        # Recursively apply to list elements (like in conjunctions, etc.)
        return [rename_predicates(subexp) for subexp in expression]
    else:
        raise ValueError(f"Invalid expression type: {type(expression)}")