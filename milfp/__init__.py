from mip import *
from milfp.milfp import Model, NonLinExpr, Var

# overwrite mip variables
mip.Var = mip.entities.Var = Var
mip.LinExpr = mip.entities.LinExpr = LinExpr = NonLinExpr
# mip.xsum = mip.model.xsum = xsum

name = "milfp"

