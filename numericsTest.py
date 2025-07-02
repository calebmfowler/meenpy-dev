from meenpy.numerics import Equation, System
from sympy import symbols as symbolize

eqn1 = Equation("y = m*x + b")

print(eqn1)

residual = eqn1.residual({
    "y": 1,
    "m": 1,
    "x": 1,
    "b": 0
})
print(residual)

sol = eqn1.solve({
    "m": 1,
    "x": 1,
    "b": -1
})
print(sol)

eqn2 = Equation("y = n*x")
system = System([eqn1, eqn2])

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