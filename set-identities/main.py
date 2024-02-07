from functools import wraps
from typing import Callable

from mip import BINARY, LinExpr, Model, Var, maximize, xsum

Expr = Var | LinExpr


def var_linexpr(x: Expr) -> LinExpr:
    """Convert a variable to a linear expression."""
    if isinstance(x, Var):
        return LinExpr([x], [1])  # type: ignore
    elif isinstance(x, LinExpr):
        return x
    else:
        raise ValueError(f"Invalid type {type(x)}.")


def transfer_vars(model: Model, other: Model | None) -> Model:
    """Transfer variables from another model to this model."""
    if other is None or other == model:
        return model
    for var in other.vars:
        # assume variables already present are identical
        if model.var_by_name(var.name) is None:
            model.add_var(
                name=var.name, lb=var.lb, ub=var.ub, var_type=var.var_type
            )
    return model


def transfer_expr(model: Model, expr: Expr) -> Expr:
    """Transfer a variable or linear expression from another model."""
    if expr.model == model:
        return expr
    transfer_vars(model, expr.model)
    if isinstance(expr, Var):
        var = model.var_by_name(expr.name)
        assert var is not None, "Variable not found."
        return var
    elif isinstance(expr, LinExpr):
        new_expr = LinExpr(const=expr.const, sense=expr.sense)
        for var, coeff in expr.expr.items():
            var = model.var_by_name(var.name)
            assert var is not None, "Variable not found."
            new_expr.add_var(var, coeff=coeff)
        return new_expr
    else:
        raise ValueError(f"Invalid type {type(expr)}.")


def transfer_constrs(model: Model, other: Model | None) -> Model:
    """Transfer constraints from another model to this model."""
    if other is None or other == model:
        return model
    for constr in other.constrs:
        # assume constraints already present are identical
        if model.constr_by_name(constr.name) is None:
            model.add_constr(
                transfer_expr(model, constr.expr),  # type: ignore
                name=constr.name,
            )
    return model


def op(f: Callable[[Expr, Expr], Expr]) -> Callable[[Expr, Expr], Expr]:
    """Decorator for convenient transfers."""

    @wraps(f)
    def wrapper(left: Expr, right: Expr) -> Expr:
        """Transfer the right expression before performing f."""
        assert left.model is not None, "Missing model."
        transfer_constrs(left.model, right.model)
        return f(left, transfer_expr(left.model, right))

    return wrapper


def cmp(f: Callable[[Expr, Expr], bool]) -> Callable[[Expr, Expr], bool]:
    """Boolean wrapper for op."""
    return op(f)  # type: ignore


def var_product(x: Var, y: Var) -> Expr:
    """Linearizes the product between two binary variables."""
    assert x.var_type == y.var_type == BINARY, "Variables not binary."
    assert x.model == y.model, "Variables belong to different models."
    # product of binary variable with itself is itself
    if x.name == y.name:
        return x
    # take advantage of commutativity, etc.
    name = "|".join(sorted(set(x.name.split("|") + y.name.split("|"))))
    model = x.model
    z = model.var_by_name(name)
    # relatively clean way of doing this
    if z is None:
        z = model.add_var(name=name, var_type=BINARY)
        model.add_constr(z <= x + y, name=f"{name}_1")
        model.add_constr(x + y <= 2 * z, name=f"{name}_2")  # type: ignore
    return x + y - z


def linexpr_product(left: LinExpr, right: LinExpr) -> LinExpr:
    """Linearizes a product term-by-term."""
    assert left.model == right.model, "Expressions belong to different models."
    return (
        xsum(
            lcoeff * rcoeff * var_product(lvar, rvar)
            for lvar, lcoeff in left.expr.items()
            for rvar, rcoeff in right.expr.items()
        )
        + left.const * right
        + right.const * left
        - left.const * right.const
    )


@op
def product(left: Expr, right: Expr) -> LinExpr:
    """Linearizes a product of variables or linear expressions."""
    return linexpr_product(var_linexpr(left), var_linexpr(right))


# indicator calculus


@op
def intersect(A: Expr, B: Expr) -> Expr:
    """Intersection."""
    return product(A, B)


@op
def union(A: Expr, B: Expr) -> Expr:
    """Union."""
    return A + B - intersect(A, B)


def complement(A: Expr) -> Expr:
    """Complement."""
    return 1 - A  # type: ignore


@op
def difference(A: Expr, B: Expr) -> Expr:
    """Difference."""
    return intersect(A, complement(B))


@op
def symm_diff(A: Expr, B: Expr) -> Expr:
    """Symmetric difference."""
    return union(A, B) - intersect(A, B)


@cmp
def subset(A: Expr, B: Expr) -> bool:
    """Subset."""
    # check A <= B, or A - B <= 0 by maximizing A - B
    model = A.model
    assert model is not None, "Missing model."
    model.objective = maximize(A - B)
    model.verbose = 0
    model.optimize()
    x = model.objective.x
    assert x is not None, "Missing objective value."
    return x <= 0


@cmp
def superset(A: Expr, B: Expr) -> bool:
    """Superset."""
    return subset(B, A)


@cmp
def equality(A: Expr, B: Expr) -> bool:
    """Equality."""
    return subset(A, B) and superset(A, B)


class Set:
    """A variable representing a set expression."""

    def __init__(self, name: str = "X", expr: Expr | None = None) -> None:
        """Initialize the set with the given name."""
        self.expr = (
            Model().add_var(name=name, var_type=BINARY)
            if expr is None
            else expr
        )

    def __str__(self) -> str:
        """String representation."""
        return str(self.expr)

    def __add__(self, other):
        """Alias for union."""
        return Set(expr=union(self.expr, other.expr))

    def __sub__(self, other):
        """Difference."""
        return Set(expr=difference(self.expr, other.expr))

    def __mul__(self, other):
        """Alias for intersection."""
        return Set(expr=intersect(self.expr, other.expr))

    def __and__(self, other):
        """Intersection."""
        return Set(expr=intersect(self.expr, other.expr))

    def __xor__(self, other):
        """Symmetric difference."""
        return Set(expr=symm_diff(self.expr, other.expr))

    def __or__(self, other):
        """Union."""
        return Set(expr=union(self.expr, other.expr))

    def __neg__(self):
        """Alias for complement."""
        return Set(expr=complement(self.expr))

    def __invert__(self):
        """Complement."""
        return Set(expr=complement(self.expr))

    def __le__(self, other) -> bool:
        """Subset."""
        return subset(self.expr, other.expr)

    def __eq__(self, other) -> bool:
        """Equality."""
        return equality(self.expr, other.expr)

    def __ge__(self, other) -> bool:
        """Superset."""
        return superset(self.expr, other.expr)


if __name__ == "__main__":
    A, B, C, D, E, F = map(Set, "ABCDEF")
    # miscellaneous
    print(A | B == (A - B) | (B - A) | (A & B))
    # De Morgan's
    print(~(A | B) == ~A & ~B)
    print(~(A & B) == ~A | ~B)
    # Hahn–Kolmogorov
    print((A | B) ^ (E | F) <= (A ^ E) | (B ^ F))
    print((A & B) ^ (E & F) <= (A ^ E) | (B ^ F))
