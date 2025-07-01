from sympy.parsing.sympy_parser import parse_expr
from sympy import sympify
from sympy import symbols

class Eqn:
    def __init__(self, eqn_str):
        eqn_str = str(eqn_str).strip()
        letters = [chr(i) for i in list(range(65, 91)) + list(range(97, 123))] # A-Z, a-z
        if any(c1 in letters and c2 in letters for c1, c2 in zip(eqn_str[:-1], eqn_str[1:])):
            raise Exception("Invalid equation input, single letter variables required")
        lhs_str, rhs_str = eqn_str.split("=")
        variable_dict = {letter: symbols(letter) for letter in letters}
        self.lhs = sympify(lhs_str, variable_dict)
        self.rhs = sympify(rhs_str, variable_dict)

    def __str__(self):
        return self.lhs.__str__() + " = " + self.rhs.__str__()
