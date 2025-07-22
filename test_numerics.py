from meenpy.numerics import Equation as Eqn, System as Sys
from meenpy.numerics.utils import *

print("Eqn Test")
y, m, x, b = symb("y, m, x, b")
line = Eqn(y, m * x + b)
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

print("Sys Test")
sinusoid = Eqn(y, sin(pi * x))
system = Sys([line, sinusoid])
print(system)
print(system.symbols)
print(system.residual({
    y: 0,
    m: 1,
    b: -1,
    x: 1
}))
print(system.solve({
    y: 0,
    m: 1,
    b: -1
}, {
    x: 0
}))
print(system.solve({
    y: 0,
    m: 1,
    b: -1
}))

print("Matrix Eqn Test")
fx, fy, fz, m, ax, ay, az = symb('fx, fy, fz, m, ax, ay, az')
F = Mat([fx, fy, fz]).T
A = Mat([ax, ay, az]).T
newtons_second_law = Eqn(F, m * A)
print(newtons_second_law)
print(newtons_second_law.residual({
    fx: 1,
    fy: 1,
    fz: 1,
    m: 1,
    ax: 1,
    ay: 1,
    az: 1
}))
print(newtons_second_law.solve({
    fx: 1,
    fy: 1,
    fz: 1,
    m: 1
}))

# Ref https://docs.sympy.org/latest/guides/solving/solve-numerically.html for refactoring
print("Matrix Sys Test")
f = symb('f')
force_magnitude = Eqn(f, sqrt(fx**2 + fy**2 + fz**2))
push_system = Sys([newtons_second_law, force_magnitude])
print(push_system)
print(push_system.symbols)
print(push_system.residual({
    fx: 1,
    fy: 1,
    fz: 1,
    m: 1,
    ax: 1,
    ay: 1,
    az: 1,
    f: sqrt(3)
}))
print(push_system.solve({
    fx: 1,
    fy: 1,
    fz: 1,
    m: 1,
    ax: 1,
    ay: 1,
    az: 1
}))