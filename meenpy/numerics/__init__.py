from pandas import read_csv
from scipy.optimize import fsolve
from sympy import sympify
from sympy import symbols as vary
from sympy.solvers.solvers import solve as sympy_solve


class Equation:
    def __init__(self, eqn_str):
        eqn_str = eqn_str.strip()

        standard_symbols = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
        custom_symbols = set(read_csv("meenpy/symbols.csv")["variable"].values)
        all_variables = {symbol : vary(symbol) for symbol in custom_symbols | set(standard_symbols)}

        lhs_str, rhs_str = eqn_str.split("=")
        self.lhs = sympify(lhs_str, all_variables)
        self.rhs = sympify(rhs_str, all_variables)
        self.variables = self.lhs.free_symbols | self.rhs.free_symbols

        return

    def __str__(self):
        return self.lhs.__str__() + " = " + self.rhs.__str__()
    
    def residual(self, symbol_subs):
        symbols, values = symbol_subs.keys(), symbol_subs.values()
        variable_subs = {vary(symbol) : value for symbol, value in zip(symbols, values)}

        return self.lhs.subs(variable_subs) - self.rhs.subs(variable_subs)
    
    def solve(self, symbol_subs):
        subbed_residual = self.residual(symbol_subs)
        
        return sympy_solve(subbed_residual)

class System:
    def __init__(self, eqn_list):
        self.eqn_list = eqn_list
        self.variables = set().union(*[eqn.variables for eqn in self.eqn_list])
        
        return
    
    def __str__(self):
        return "\n".join([eqn.__str__() for eqn in self.eqn_list])
    
    def residual(self, symbol_subs, symbol_guesses=None):
        return [eqn.residual(symbol_subs) for eqn in self.eqn_list]
    
    def solve(self, symbol_subs, symbol_guesses=None):
        subbed_residual_list = self.residual(symbol_subs)
        unknown_variables = list(set().union(*[subbed_residual.free_symbols for subbed_residual in subbed_residual_list]))

        eqn_cnt, unknown_cnt = len(subbed_residual_list), len(unknown_variables)
        if eqn_cnt > unknown_cnt:
            raise Exception("Cannot solve, system is overspecified (eqn_cnt > unknown_cnt)")
        
        elif eqn_cnt < unknown_cnt:
            raise Exception("Cannot solve, system is underspecified (eqn_cnt < unknown_cnt)")
        
        else:
            guesses = [0] * unknown_cnt
            if symbol_guesses:
                guess_map = {vary(symbol) : guess for symbol, guess in zip(symbol_guesses.keys(), symbol_guesses.values())}
                guesses = list(guess_map.get(variable) for variable in unknown_variables)

            unknown_subs = lambda solution : {variable : value for variable, value in zip(unknown_variables, solution)}
            solution_residual = lambda solution : [subbed_eqn.subs(unknown_subs(solution)) for subbed_eqn in subbed_residual_list]
            solution = fsolve(solution_residual, guesses)

            return "\n".join([f"{unknown} = {value}" for unknown, value in zip(unknown_variables, solution)])