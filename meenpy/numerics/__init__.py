from scipy.optimize import fsolve
from sympy import sympify, Expr, Basic
from sympy.solvers.solvers import solve as sympy_solve
from numpy import array as nparr

class Equation:
    def __init__(self, lhs, rhs):
        self.lhs: Basic = sympify(lhs)
        self.rhs: Basic = sympify(rhs)
        self.symbols = self.lhs.free_symbols | self.rhs.free_symbols

        return

    def __str__(self):
        return self.lhs.__str__() + ' = ' + self.rhs.__str__()
    
    def residual(self, subs: dict, type: str = 'differential'):
        if type == 'differential':
            return sympify(self.lhs.subs(subs)) - sympify(self.rhs.subs(subs))
        elif type == 'rational':
            return sympify(self.lhs.subs(subs)) / sympify(self.rhs.subs(subs)) - 1
        else:
            raise Exception(f'Invalid residual type "{type}"')
    
    def solve(self, subs, type: str = 'differential'):
        subbed_residual = self.residual(subs, type)
        
        return sympy_solve(subbed_residual)


class System:
    def __init__(self, eqn_list: list[Equation]):
        self.eqn_list = eqn_list
        self.symbols = set().union(*[eqn.symbols for eqn in self.eqn_list])
        
        return
    
    def __str__(self):
        return '| ' + '\n| '.join([eqn.__str__() for eqn in self.eqn_list])
    
    def residual(self, subs, types: list[str] = []):
        if types == []:
            types = ['differential'] * len(self.eqn_list)
        
        return [eqn.residual(subs, type) for eqn, type in zip(self.eqn_list, types)]
    
    def solve(self, subs, guess_dict: dict = {}):
        subbed_residual: list[Basic] = self.residual(subs)
        unknowns = list(set().union(*[element.free_symbols for element in subbed_residual]))
        eqn_cnt, unknown_cnt = len(subbed_residual), len(unknowns)

        if eqn_cnt > unknown_cnt + 1:
            raise Exception('Cannot solve, system is overspecified (eqn_cnt > unknown_cnt)')
        
        elif eqn_cnt < unknown_cnt + 1:
            raise Exception('Cannot solve, system is underspecified (eqn_cnt < unknown_cnt)')
        
        else:
            guess_vect = [0] * unknown_cnt
            if guess_dict != {}:
                guess_vect = [guess_dict.get(variable) for variable in unknowns]

            unknown_subs = lambda solution: {unknown: solution_element for unknown, solution_element in zip(unknowns, solution)}
            solution_residual = lambda solution: [subbed_residual_element.subs(unknown_subs(solution)) for subbed_residual_element in subbed_residual][0 : unknown_cnt]

            solution = fsolve(solution_residual, guess_vect)

            return '\n'.join([f'{unknown} = {value}' for unknown, value in zip(unknowns, solution)])

