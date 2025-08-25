from scipy.optimize import fsolve
from sympy import sympify, lambdify, Basic, Number, Float, Matrix
from sympy.solvers.solvers import solve as sympy_solve
from numpy import array as nparr, ndarray, float64 as npfloat, concatenate
from collections.abc import Iterable
from inspect import signature 

class Equation:
    def __init__(self, lhs, rhs) -> None:
        self.lhs: Basic = sympify(lhs)
        self.rhs: Basic = sympify(rhs)
        self.symbols = self.lhs.free_symbols | self.rhs.free_symbols

        return

    def __str__(self) -> str:
        return self.lhs.__str__() + ' = ' + self.rhs.__str__()
    
    def residual(self, subs: dict, numerical: bool = False, type: str = 'differential') -> Basic:
        if not numerical and type == 'differential':
            return sympify(self.lhs.subs(subs)) - sympify(self.rhs.subs(subs))
        elif not numerical and type == 'rational':
            return sympify(self.lhs.subs(subs)) / sympify(self.rhs.subs(subs)) - 1 # Gonna need to branch for matrix values Eqns to invert & subtract identity
        elif numerical:
            symbolic_residual = self.residual(subs, type=type)

            if isinstance(symbolic_residual, Number):
                return Float(self.residual(subs, type=type))
            else:
                raise ValueError(f'Insufficient subs for numerical residual evaluation\nEquation =\n{self.__str__()}\nsubs =\n{subs}')
        else:
            raise ValueError(f'Invalid residual type "{type}"')
    
    def solve(self, subs: dict, residual_type: str = 'differential'):
        subbed_residual = self.residual(subs, numerical=False, type=residual_type)
        
        return sympy_solve(subbed_residual)


class System:
    def __init__(self, eqn_list: list[Equation]) -> None:
        self.eqn_list = eqn_list
        self.symbols = set().union(*[eqn.symbols for eqn in self.eqn_list])
        
        return
    
    def __str__(self) -> str:
        return '| ' + '\n| '.join([eqn.__str__() for eqn in self.eqn_list])
    
    def residual(self, subs: dict, numerical: bool = False, types: list[str] = []) -> list[Basic]:
        if types == []:
            types = ['differential'] * len(self.eqn_list)
        
        residual = []
        for eqn, type in zip(self.eqn_list, types):
            try:
                eqn_residual = eqn.residual(subs, numerical=numerical, type=type)
            except ValueError as eqn_exception:
                raise ValueError(f'Failure to calculate system residual\nSystemt =\n{self.__str__()}\n{eqn_exception.__str__()}')
            
            residual.append(eqn_residual)

        return residual
    
    def solve(self, subs: dict, guess_dict: dict = {}, residual_types: list[str] = []): # -> dict[Basic, npfloat]
        subbed_residual: list[Basic] = self.residual(subs, types=residual_types)
        eqn_unknown_sets: list[list[Basic]] = [list(element.free_symbols) for element in subbed_residual]
        eqn_dims = [len(subbed_eqn) if isinstance(subbed_eqn, Matrix) else 1 for subbed_eqn in subbed_residual]

        func_residual = [lambdify(eqn_unknowns, subbed_eqn) for eqn_unknowns, subbed_eqn in zip(eqn_unknown_sets, subbed_residual)]
        
'''
        guess_vect = [0] * unknown_cnt
        if guess_dict != {}:
            guess_vect = [guess_dict.get(variable) for variable in unknowns]

        unknown_subs = lambda solution: dict(zip(unknowns, solution))
        def solution_residual_func(solution):
            return [element.subs(unknown_subs(solution)) for element in subbed_residual][0 : unknown_cnt]

        fsolve_result = fsolve(solution_residual_func, guess_vect)
        print(f'fsolve_result\n{fsolve_result}\n')
        solution_vect = nparr(fsolve_result, dtype=npfloat)
        print(f'solution_vect\n{solution_vect}\n')
        solution_dict = dict(zip(unknowns, solution_vect))
        print(f'solution_dict\n{solution_dict}\n')
        return solution_dict
    '''
'''
        if eqn_cnt > unknown_cnt + 1:
            raise ValueError('Cannot solve, system is overspecified (eqn_cnt > unknown_cnt)')
        elif eqn_cnt < unknown_cnt + 1:
            raise ValueError('Cannot solve, system is underspecified (eqn_cnt < unknown_cnt)')
        else:
'''
'''
            if isinstance(eqn_residual, Iterable):
                residual.append(eqn_residual)
            else:
                residual.append([eqn_residual])

        return [element for subresidual in residual for element in subresidual]
'''