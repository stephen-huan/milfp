import milfp as mip

m = mip.Model(sense=mip.MAXIMIZE, solver_name=mip.CBC)
x = m.add_var(name="x", var_type=mip.BINARY)
y = m.add_var(name="y", var_type=mip.BINARY)
z = m.add_var(name="z", var_type=mip.CONTINUOUS)

m += z <= 5*y
m += x*z == 0
m.objective = x*(x + 2*y) + z

status = m.optimize()
print(m.objective.x)
print(x.x, y.x, z.x)

