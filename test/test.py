import milfp as mip

m = mip.Model(sense=mip.MAXIMIZE, solver_name=mip.CBC)
x = m.add_var(name="x", var_type=mip.BINARY)
y = m.add_var(name="y", var_type=mip.BINARY)
m += x*y == 0
# m.objective = x + 2*y
m.objective = x*(x + 2*y)
# m.objective = (x + 2*y)*(x + y*y)

status = m.optimize()
print(m.objective.x)
print(x.x, y.x)

