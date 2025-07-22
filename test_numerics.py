from meenpy.numerics import Equation as E, System as S
# from meenpy.numerics.utils import *
from sympy import symbols as symb, sin, cos, pi

y, m, x, b = symb("y, m, x, b")
line = E(y, m * x + b)

print(line)
print(line.residual({
    y: 1,
    m: 1,
    x: 1,
    b: 0
}))
print(line.solve({
    y: 0,
    m: 1,
    b: -1
}))

sinusoid = E(y, sin(pi * x))
system = S([line, sinusoid])

print(system)
print(system.symbols)
print(system.solve({
    y: 0,
    m: 1,
    b: -1
}, {
    x: 0
}))

