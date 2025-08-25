import unittest
from meenpy.numerics import Equation as Eqn, System as Sys
from meenpy.numerics.utils import *
import numpy as np

class MEENPyTest(unittest.TestCase):
    def test_scalar_eqn_and_sys(self):
        return

        y, m, x, b = symb("y, m, x, b")
        line = Eqn(y, m * x + b)
        self.assertAlmostEqual(line.solve({y: 0, m: 1, b: -1}), [1])

        sinusoid = Eqn(y, sin(pi * x))
        system = Sys([line, sinusoid])

        solution_value = np.float64(system.solve({y: 0, m: 1, b: -1}, {x: 0}).get(x))
        self.assertAlmostEqual(solution_value, np.float64(1))

    def test_vector_eqn_and_sys(self):
        fx, fy, fz, m, ax, ay, az = symb('fx, fy, fz, m, ax, ay, az')
        F = Mat([fx, fy, fz]).T
        A = Mat([ax, ay, az]).T
        newtons_second_law = Eqn(F, m * A)

        self.assertAlmostEqual(newtons_second_law.solve({fx: 1, fy: 1, fz: 1, m: 1}), dict({ax: 1, ay: 1, az: 1}))

        f = symb('f')
        magnitude = Eqn(f, sqrt(sum([component**2 for component in [fx, fy, fz]])))
        system = Sys([newtons_second_law, magnitude])
        print(system)
        system.solve({
            fx: 1,
            fy: 1,
            fz: 1,
            m: 1,
            ax: 1,
            ay: 1,
            az: 1
        })
        
        pass
    pass

if __name__ == '__main__':
    unittest.main()