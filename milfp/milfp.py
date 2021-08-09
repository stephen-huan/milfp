from typing import Union
from numbers import Real
from functools import lru_cache
import mip

LinExpr, Var = mip.LinExpr, mip.Var
Linear = Union[LinExpr, "milfp.NonLinExpr"]
Other = Union["milfp.Var", Real, "milfp.NonLinExpr"]
NodeValue = Union["milfp.Node", Other]

# TODO: caching, difficult because  __eq__ (==) is overwritten
# caching can break if the model/variables are mutated...
# TODO: take advantage of commutivity in caching (x1*x2 == x2*x1)
# and for canceling (x1*x2*x1 = x1*x1*x2 = x1*x2)

def quick_bound(sense: str, g: LinExpr) -> Real:
    """ Finds a quick bound on a linear expression with linear logic. """
    k = sense == mip.MAXIMIZE
    x = 0
    for var, coef in g.expr.items():
        x += coef*(var.lb, var.ub)[k ^ (coef < 0)]
    return x + g.const

def bound(model: mip.Model, sense: str, g: LinExpr) -> Real:
    """ Finds a bound on a linear expression with relaxation. """
    relaxed = model.copy()
    relaxed.objective, relaxed.sense = g, sense
    # noisy operation --- hide output?
    status = relaxed.optimize(relax=True)
    return relaxed.objective_value

def binary_var_product(x: Var, y: Var) -> LinExpr:
    """ Linearizes the product between two binary variables. """
    assert x.var_type == y.var_type == mip.BINARY, "variables not binary"
    if x.equals(y):
        # product of binary variable with itself is just itself
        pass
        # return x
    # multiple ways of doing this but this is probably the cleanest
    model = x.model
    z = model.add_var(name=f"__({x.name})*({y.name})",
                      var_type=mip.BINARY)
    model += z <= x + y
    model += x + y <= 2*z
    return x + y - z

def var_glover_product(x: Var, g: Union[Var, LinExpr]) -> Union[Var, LinExpr]:
    """ Linearizes a product between a discrete variable and an
        arbitrary linear expression with Glover's linearization. """
    assert x.var_type in {mip.BINARY, mip.INTEGER}, "variable not discrete"
    g = to_LinExpr(g)
    model = x.model
    if x.var_type == mip.INTEGER:
        # decompose into sum of binary variables
        if abs(x.ub) == mip.INF or abs(x.lb) == mip.INF:
            raise TypeError(f"Can't linearize {x.name}, infinite range")
        ys = [model.add_var(name=f"__{x.name}_{i}", var_type=mip.BINARY)
              for i in range(round(x.ub - x.lb))]
        model += mip.xsum(ys) <= 1
        return glover_product(LinExpr(ys, range(1, len(ys) + 1), x.lb), g)
    # binary variable
    else:
        # find lower and upper bound of g with relaxation
        L, U = quick_bound(mip.MINIMIZE, g), quick_bound(mip.MAXIMIZE, g)
        # L, U = bound(model, mip.MINIMIZE, g), bound(model, mip.MAXIMIZE, g)
        # apply Glover's linearization
        y = model.add_var(lb=L, ub=U, var_type=mip.CONTINUOUS)
        model += L*x <= y
        model += y <= U*x
        model += g - U*(1 - x) <= y
        model += y <= g - L*(1 - x)
        return y

def term_product(left: LinExpr, right: LinExpr) -> LinExpr:
    """ Linearizes a product by computing the term-by-term product. """
    return mip.xsum(lcoef*rcoef*binary_var_product(lvar, rvar)
                    for lvar, lcoef in left.expr.items()
                    for rvar, rcoef in right.expr.items()) + \
        left.const*right + right.const*left - left.const*right.const

def glover_product(left: LinExpr, right: LinExpr) -> LinExpr:
    """ Linearizes a product by reducing the left side term-by-term. """
    return mip.xsum(coef*var_glover_product(var, right)
                    for var, coef in left.expr.items()) + left.const*right

def product(left: NodeValue, right: NodeValue, glover: bool=True) -> NodeValue:
    """ Computes the product between left and right. """
    # we assume the left side is entirely discrete
    left, right = to_LinExpr(left), to_LinExpr(right)
    return (glover_product if glover else term_product)(left, right)

def is_discrete(x: NodeValue) -> bool:
    """ Returns whether the object can be trivially linearized. """
    if isinstance(x, Var):
        return x.var_type in {mip.BINARY, mip.INTEGER}
    if isinstance(x, NonLinExpr):
        return all(map(is_discrete, x.expr))
    if isinstance(x, Node):
        return x.is_discrete
    # real number
    return True

def value(x: NodeValue) -> Union[Var, LinExpr, Real]:
    """ Converts a Node or NonLinExpr into a proper mip variable. """
    if isinstance(x, Node):
        return x.linearize()
    if type(x) == NonLinExpr:
        return x.linear_form
    return x

def get_model(x: NodeValue) -> Union[mip.Model, None]:
    """ Returns the model of x, if it exists. """
    return x.model if hasattr(x, "model") else None


class Node():
    """
    A node in the arithmetic syntax tree.

    A node's value is an arithmetic operator (__add__, __sub__, etc.). A node
    has exactly two children, either Node, NonLinExpr, Var, or Real objects.
    """

    def __init__(self, operator: str, left: NodeValue, right: NodeValue):
        self.ops = {
            "__add__": "+",
            "__sub__": "-",
            "__mul__": "*",
            "__truediv__": "/"
        }
        assert operator in self.ops, f"unsupported operation {operator}"
        self.operator, self.left, self.right = operator, left, right
        # computed properties
        left_model, right_model = get_model(self.left), get_model(self.right)
        self.model = left_model if right_model is None else right_model
        self.is_discrete = is_discrete(self.left) and is_discrete(self.right)

    def __str__(self) -> str:
        """ Displays the tree in-line, recursively. """
        op = self.ops[self.operator]
        s = "({}){}({})" if op in "*/" else "{} {} {}"
        return s.format(self.left, op, self.right)

    def linearize(self, discrete: bool=True) -> mip.LinExpr:
        """ Reduces the nonlinear expression into a linear expression. """
        l, r = self.left, self.right
        # linear operations
        if self.operator in ["__add__", "__sub__"] or \
                isinstance(l, Real) or isinstance(r, Real):
            l, r, op = (l, r, self.operator) if isinstance(r, Real) else \
                       (r, l, "__r" + self.operator[2:])
            return getattr(value(l), op)(value(r))
        # otherwise, determine which side is discrete/continuous
        num_discrete = is_discrete(l) + is_discrete(r)
        # can't linearize product of two continous variables
        if num_discrete == 0:
            raise TypeError(f"can't linearize expression {self}: \
product of two continuous variables")
        # only choice is to apply Glover's to the continuous side 
        elif num_discrete == 1:
            l, r = (l, r) if is_discrete(l) else (r, l)
            return value(product(value(l), value(r), glover=True))
        # choose side by minimizing number of terms
        else:
            return value(product(value(l), value(r), glover=False))


class NonLinExpr():
    """
    An extension of mip.LinExpr for automatic linearization.

    It overwrites multiplication and division by recording all
    arithmetic operations performed on it in a binary tree,
    which is then decomposed into a linear expression on demand.
    """

    def __init__(self, *args, **kwargs):
        self.root = LinExpr(*args, **kwargs)
        self.__linear_form = self.root

    def __str__(self) -> str:
        return str(self.root)

    def add(self, operator: str, other: Other) -> "milfp.NonLinExpr":
        """ Implements the operator by adding a node to the tree. """
        other = other.root if type(other) == NonLinExpr else other
        # handle right operators
        if operator[2] == "r":
            operator = "__" + operator[3:]
            self.root, other = other, self.root
        # soft copy, we need a new object, but we can recycle node pointers
        result = NonLinExpr()
        result.root = Node(operator, self.root, other)
        result.__linear_form = None
        return result

    @property
    def linear_form(self) -> mip.LinExpr:
        """
        Reduces the nonlinear expression into a linear expression.

        Wraps the Node's linearize function with additional caching.
        """
        if self.__linear_form is None:
            self.__linear_form = value(self.root.linearize())
        return self.__linear_form

    def compare(self, comparision: str, other: Other) -> mip.LinExpr:
        """ Uses the derived linear expression for comparisions. """
        return getattr(self.linear_form, comparision)(other)

    # in Python, __getattr__ is called only when AttributeError would happen
    # this introduces hard to debug errors because it ignores AttributeError
    def __getattr__(self, name: str):
        """ Uses the derived linear expression for attribute lookups. """
        return getattr(self.linear_form, name)

# TODO: power x**k
# TODO: unitary negative -x
# all arithmetic operations are simply lazily added to the tree 
for func in ["__add__", "__radd__", "__sub__", "__rsub__",
             "__mul__", "__rmul__", "__truediv__", "__rtruediv__"]:
    setattr(NonLinExpr, func,
            (lambda name:
                lambda self, other: NonLinExpr.add(self, name, other)
            )(func))

# wrap functions by deriving from the final linear form
# other functions will be called on the resulting LinExpr object
for func in ["__eq__", "__le__", "__ge__"]:
    setattr(NonLinExpr, func,
            (lambda name:
                lambda self, other: NonLinExpr.compare(self, name, other)
            )(func))


class LinExpr(mip.LinExpr, NonLinExpr):
    """ The "base" class mip.LinExpr modified to stick within LinExpr. """

    def copy(self) -> "mip.LinExpr":
        copy = LinExpr()
        copy.__const = self.__const
        copy.__expr = self.__expr.copy()
        copy.__sense = self.__sense
        return copy

    def __getattr__(self, name: str):
        raise AttributeError


class Var(mip.Var):
    """
    An extension of mip.Var for automatic linearization.

    In particular, it overwrites multiplication and division.
    """

    def equals(self, other: "milfp.Var") -> bool:
        return hash(self) == hash(other)

    def __mul__(self, other: Other) -> Other:
        # use default multiplication if Real
        return super().__mul__(other) if isinstance(other, Real) else \
            to_LinExpr(self, NonLinExpr)*other


class Model(mip.Model):
    """ An extension of mip.Model for solving nonlinear programs. """

    def set_objective(self, objective):
        if type(objective) == NonLinExpr:
            objective = objective.linear_form
        mip.Model.objective.fset(self, objective)

    objective = property(mip.Model.objective.fget, set_objective)


def to_LinExpr(x: Union[Var, Linear], cls: Linear=LinExpr) -> Linear:
    """ Converts a variable into a linear expression. """
    return cls([x], [1]) if isinstance(x, Var) else x

