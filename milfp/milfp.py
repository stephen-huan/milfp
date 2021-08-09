from typing import List, Optional, Dict, Union, Tuple
from numbers import Real
from functools import lru_cache
import mip

# TODO: power x**k

class LinExpr(mip.LinExpr):
    """ An extension of mip.LinExpr for automatic linearization.
    In particular, it overwrites multiplication and division. """

    @property
    def is_discrete(self) -> bool:
        """ Returns whether all the variables are integer type. """
        return all(var.var_type != mip.CONTINUOUS for var in self.expr)

    @property
    def num_terms(self) -> int:
        """ Counts the number of terms relevant when linearizing a product. """
        return sum(var.ub - var.lb for var in self.expr)

    def __mul__(self, other: Union["milfp.Var", Real, "milfp.LinExpr"]) -> Union[Real, "milfp.LinExpr"]:
        if not isinstance(other, Real):
            if isinstance(other, Var):
                other = LinExpr([other], [1])
            num_discrete = self.is_discrete + other.is_discrete
            if num_discrete == 0:
                raise TypeError(f"Both expressions include continuous terms")
            elif num_discrete == 1:
                left, right = (self, other) if self.is_discrete else (other, self)
                return factorize(left, right)
            else:
                # could use either, pick by minimizing terms
                left, right = (self, other) if self.num_terms < other.num_terms else (other, self)
                return brute_force_product(left, right)
                # return factorize(left, right)
        # Real, use default multiplication
        else:
            return super().__mul__(other)

class Var(mip.Var):
    """ An extension of mip.Var for automatic linearization.
    In particular, it overwrites multiplication and division. """

    def equals(self, other: "milfp.Var") -> bool:
        return False
        return hash(self) == hash(other)

    def __mul__(self, other: Union["milfp.Var", Real, LinExpr]) -> Union["milfp.Var", Real, LinExpr]:
        if not isinstance(other, Real):
            if isinstance(other, Var) and self.var_type == other.var_type == mip.BINARY:
                if self.equals(other):
                    # product of binary variable with itself is just itself
                    return self
                # multiple ways of doing this but this is probably the cleanest
                model = self.model
                z = model.add_var(name=f"__{self.name}*{other.name}",
                                  var_type=mip.BINARY)
                model += z <= self + other
                model += self + other <= 2*z
                return self + other - z
            else:
                return LinExpr([self], [1])*other
        # Real, use default multiplication
        else:
            return super().__mul__(other)

# overwrite mip classes with new classes
mip.Var = mip.entities.Var = Var
mip.LinExpr = mip.entities.LinExpr = LinExpr

# TODO: caching, difficult because  __eq__ (==) is overwritten
# caching can break if the model/variables are mutated...

def quick_bound(sense: str, g: LinExpr) -> Real:
    """ Finds a quick bound on g with linear logic. """
    k = sense == mip.MAXIMIZE
    x = 0
    for var, coef in g.expr.items():
        x += coef*(var.lb, var.ub)[k ^ (coef < 0)]
    return x + g.const

def bound(model: mip.Model, sense: str, g: LinExpr) -> Real:
    """ Finds a bound on g with relaxation. """
    relaxed = model.copy()
    relaxed.objective, relaxed.sense = g, sense
    status = relaxed.optimize(relax=True) # noisy operation --- hide output?
    return relaxed.objective_value

def linearize_product(x: Var, g: LinExpr) -> Union[LinExpr, Var]:
    """ Linearizes a product between a discrete variable
        and an arbitrary linear expression. """
    assert x.var_type in {mip.BINARY, mip.INTEGER}, "variable not discrete"
    if isinstance(g, Var):
        g = LinExpr([g], [1])
    model = x.model
    if x.var_type == mip.INTEGER:
        # decompose into sum of binary variables
        if abs(x.ub) == mip.INF or abs(x.lb) == mip.INF:
            raise TypeError(f"Can't linearize {x.name}, infinite range")
        ys = [model.add_var(name=f"__{x.name}_{i}", var_type=mip.BINARY)
              for i in range(round(x.ub - x.lb))]
        model += mip.xsum(ys) <= 1
        return mip.xsum((i + 1)*linearize_product(ys[i], g)
                        for i in range(len(ys))) + x.lb*g
    else: # binary variable
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

def brute_force_product(left: LinExpr, right: LinExpr) -> LinExpr:
    """ Lineraizes a product by computing the term-by-term product. """
    return mip.xsum(lcoef*rcoef*(lvar*rvar)
                    for lvar, lcoef in left.expr.items()
                    for rvar, rcoef in right.expr.items()) + \
        left.const*right + right.const*left - left.const*right.const

def factorize(left: LinExpr, right: LinExpr) -> LinExpr:
    """ Linearizes a product by reducing the left side term-by-term. """
    return mip.xsum(coef*linearize_product(var, right)
                    for var, coef in left.expr.items()) + left.const*right

if __name__ == "__main__":
    m = mip.Model(sense=mip.MAXIMIZE, solver_name=mip.CBC)
    x = m.add_var(name="x", var_type=mip.BINARY)
    y = m.add_var(name="y", var_type=mip.BINARY)
    m += x*y == 0
    # m.objective = x + 2*y
    m.objective = (x + 2*y)*(x + y*y)

    status = m.optimize()
    print(m.objective.x)
    print(x.x, y.x)

