from meenpy.numerics import Equation as E, System as S

eqn1 = E("y = m*x + b")

print(eqn1)
print(eqn1.residual({
    "y": 1,
    "m": 1,
    "x": 1,
    "b": 0
}))
print(eqn1.solve({
    "m": 1,
    "x": 1,
    "b": -1
}))

eqn2 = E("y = n*x")
system = S([eqn1, eqn2])

print(system)
print(system.variables)
print(system.solve({
    "m": 1,
    "n": -1,
    "b": -2
}, {
    "y": 1,
    "x": 1
}))