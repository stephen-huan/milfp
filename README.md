# Python MILFP (Mixed-Integer Linear Fractional Programming) Library

A simple extension of [Python-MIP](https://github.com/coin-or/python-mip)
to support linear fractional programming as well as linear programming.

Uses the [reformulation-linearization](
https://optimization.mccormick.northwestern.edu/index.php/Mixed-integer_linear_fractional_programming_(MILFP))
method to convert linear fractional programs to linear programs,
specifically using the [Charnes-Cooper transformation](
http://lpsolve.sourceforge.net/5.1/ratio.htm)
and [Glover's linearization](
https://glossary.informs.org/ver2/mpgwiki/index.php/Glovers_linearization)
to account for integer variables
(see [_Comparisons and enhancement strategies for linearizing mixed
0-1 quadratic programs_](https://doi.org/10.1016/j.disopt.2004.03.006)
for a simple explanation of why this works).

Although this method of solving MILFPs is simple and relies on Python-MIP's
ability to solve MILPs quickly (rather than re-inventing the wheel), it
comes as the cost of making the problem size substantially bigger. Perhaps
the parametric algorithm may be faster for certain problems.

## Technical Notes

### Denominators

It is assumed that denominators of a fraction are positive. If the denominators
are negative, then negate the denominator and negate the numerator. However,
the denominator of each fraction must be either strictly positive or strictly
negative for interpretable results. If the denominator can switch signs, then
it cannot be represented as a linear program. If the denominator can be zero,
then that forces the numerator to be zero.

### Non-binary Integer Variables

Currently `milfp` only works for binary (0/1) integer variables
because of Glover's linearization. Note that it is possible to
encode an integer variable between [0, M] into M binary variables:
```
Suppose we have the variable x3 that takes integer values between [0, 3].
Encode x3 into 4 binary variables y0, y1, y2, y3 and enforce the condition that 
y0 + y1 + y2 + y3 = 1 (i.e. exactly one must be active) 
and replace x3 with (0*y0 + 1*y1 + 2*y2 + 3*y3)
(i.e. it can take a value between [0, 3]). 
Clearly y0 is useless so it can be dropped if the condition is changed to
y1 + y2 + y3 <= 1 (i.e. allow that none of them are on which is the 0 case)
```
Of course the same trick can be used for negative values as well,
but it is probably simpler to shift a variable between [-N, M] to
[0, N + M] and then subtract N from the value of the variable.

## Examples

See the examples in [stephen-huan/probabilistic-rolling](
https://github.com/stephen-huan/probabilistic-rolling)
for the use of MILFP to solve practical problems.

