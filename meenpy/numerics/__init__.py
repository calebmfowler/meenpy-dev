from scipy.optimize import fsolve
from sympy import sympify, lambdify, Basic, Expr, Matrix
from numpy import array as nparr, ndarray, float64 as npfloat, concatenate, prod, sum
from inspect import signature
from typing import Callable, get_type_hints

usernum = int | float | npfloat

class Equation:
    def __init__(self, lhs, rhs) -> None:
        self.init_lhs_rhs(lhs, rhs)
        self.init_shape_size()
        self.init_free_symbols()
    
    def init_lhs_rhs(self, lhs, rhs) -> None:
        self.lhs = sympify(lhs)
        self.rhs = sympify(rhs)
    
    def init_shape_size(self) -> None:
        self.shape: tuple[int, int] = (-1, -1)
        self.size: int = -1
    
    def init_free_symbols(self) -> None:
        self.free_symbols: set[Basic] = set()

    def residual(self, subs: dict[Basic, usernum], residual_type: str = "differential") -> Expr | Matrix:
        return Expr()
    
    def lambda_residual(self, subs: dict[Basic, usernum], residual_type: str = "differential") -> tuple[Callable, list[Basic]]:
        return (lambda *args, **kwargs: None, [])
    
    def solve(self, subs: dict[Basic, usernum], residual_type: str = "differential") -> dict[Basic, npfloat]:
        return {}

    def __str__(self) -> str:
        return self.lhs.__str__() + ' = ' + self.rhs.__str__()


class ScalarEquation(Equation):
    def init_lhs_rhs(self, lhs, rhs) -> None:
        self.lhs: Expr = sympify(lhs)
        self.rhs: Expr = sympify(rhs)

    def init_shape_size(self) -> None:
        self.shape: tuple[int, int] = (1, 1)
        self.size = 1
    
    def init_free_symbols(self) -> None:
        self.free_symbols: set[Basic] = self.lhs.free_symbols | self.rhs.free_symbols
    
    def residual(self, subs: dict[Basic, usernum], residual_type: str = "differential") -> Expr:
        if residual_type == "differential":
            return sympify(self.lhs.subs(subs)) - sympify(self.rhs.subs(subs))
        elif residual_type == "left_rational":
            return sympify(self.lhs.subs(subs)) / sympify(self.rhs.subs(subs)) - 1
        elif residual_type == "right_rational":
            return sympify(self.rhs.subs(subs)) / sympify(self.lhs.subs(subs)) - 1
        else:
            raise ValueError(f"Invalid residual_type = '{residual_type}'")
    
    def lambda_residual(self, subs: dict[Basic, usernum], residual_type: str = "differential") -> tuple[Callable[[ndarray], usernum], list[Basic]]:
        residual = self.residual(subs, residual_type)
        residual_free_symbols_list = list(residual.free_symbols)
        lambda_residual_func: Callable[[tuple], usernum] = lambdify(residual_free_symbols_list, residual)

        def unpack_wrapper(func: Callable[[tuple], usernum]):
            return lambda arr: func(*arr)

        return unpack_wrapper(lambda_residual_func), residual_free_symbols_list
        
    def solve(self, subs: dict[Basic, usernum], residual_type: str = "differential", guess: usernum = 1) -> dict[Basic, npfloat]:
        residual_lambda, residual_free_symbols_list = self.lambda_residual(subs, residual_type)

        exception_output = f"\
                ScalarEquation:\n    {self.__str__().replace('\n', '\n    ')}\n\
                subs:\n    {subs.__str__().replace('\n', '\n    ')}\n\
                residual_free_symbols_list:\n    {residual_free_symbols_list.__str__().replace('\n', '\n    ')}"

        if len(residual_free_symbols_list) != 1:
            raise ValueError(f"Insufficient subs to solve scalar equation\n{exception_output}")
        
        try:
            solution = fsolve(residual_lambda, guess)
        except Exception as e:
            raise Exception(f"{e}\nUnable to solve ScalarEquation\n{exception_output}")

        return {residual_free_symbols_list[0] : npfloat(solution[0])}


class MatrixEquation(Equation):
    def init_lhs_rhs(self, lhs, rhs) -> None:
        self.lhs: Matrix = sympify(lhs)
        self.rhs: Matrix = sympify(rhs)
    
    def init_shape_size(self) -> None:
        lhs_shape: tuple[int, int] = self.lhs.shape
        rhs_shape: tuple[int, int] = self.rhs.shape
        lhs_rows, lhs_cols = lhs_shape
        rhs_rows, rhs_cols = rhs_shape

        exception_output = f"\
                lhs:\n    {str(self.lhs.__str__()).replace('\n', '\n    ')}\n\
                rhs:\n    {str(self.rhs.__str__()).replace('\n', '\n    ')}\n\
                lhs.shape: {self.lhs.shape}\n\
                rhs.shape: {self.rhs.shape}"

        if lhs_rows == 0 or lhs_cols == 0 or rhs_rows == 0 or rhs_cols == 0:
            raise ValueError(f"Given expression shapes include a zero width dimension\n{exception_output}")
        
        if self.lhs.shape != self.rhs.shape:
            raise ValueError(f"Given matrices have unequal shapes\n{exception_output}")
        
        self.shape: tuple[int, int] = self.lhs.shape
        self.size: int = int(prod(self.shape))

    def init_free_symbols(self) -> None:
        self.free_symbols: set[Basic] = self.lhs.free_symbols | self.rhs.free_symbols

    def residual(self, subs: dict[Basic, usernum], residual_type: str = "differential") -> Matrix:
        if residual_type == "differential":
            return sympify(self.lhs.subs(subs)) - sympify(self.rhs.subs(subs))
        else:
            raise ValueError(f"Invalid residual_type = '{residual_type}'")

    def lambda_residual(self, subs: dict[Basic, usernum], residual_type: str = "differential") -> tuple[Callable[[ndarray], ndarray], list[Basic]]:
        residual = self.residual(subs, residual_type)
        residual_free_symbols_list: list[Basic] = list(residual.free_symbols)
        shaped_lambda_residual_func: Callable[[tuple], ndarray] = lambdify(residual_free_symbols_list, residual)

        def unpack_ravel_wrapper(func: Callable[[tuple], ndarray]) -> Callable[[ndarray], ndarray]:
            return lambda args: nparr(func(*args)).ravel()

        return unpack_ravel_wrapper(shaped_lambda_residual_func), residual_free_symbols_list

    def solve(self, subs: dict[Basic, usernum], residual_type: str = "differential", guess_dict: dict[Basic, usernum] = {}) -> dict[Basic, npfloat]:
        lambda_residual_func, residual_free_symbols_list = self.lambda_residual(subs, residual_type)

        exception_output = f"\
                MatrixEquation:\n    {self.__str__().replace('\n', '\n    ')}\n\
                subs:\n    {subs.__str__().replace('\n', '\n    ')}\n\
                residual_free_symbols_list:\n    {residual_free_symbols_list.__str__().replace('\n', '\n    ')}"

        if len(residual_free_symbols_list) > self.size:
            raise ValueError(f"Insufficient subs to solve matrix equation\n{exception_output}")
        
        guess_vect = [guess_dict.get(free_symbol) if free_symbol in guess_dict.keys() else 1 for free_symbol in residual_free_symbols_list]
        
        try:
            solution = nparr(fsolve(lambda_residual_func, guess_vect), dtype=npfloat)
        except Exception as e:
            raise Exception(f"{e}\nUnable to solve MatrixEquation\n{exception_output}")

        return dict(zip(residual_free_symbols_list, solution))


class System:
    def __init__(self, eqn_list: list[Equation]) -> None:
        self.eqn_list = eqn_list
        self.free_symbols = set().union(*[eqn.free_symbols for eqn in self.eqn_list])
        self.size = sum([eqn.size for eqn in self.eqn_list])
    
    def __str__(self) -> str:
        return "| " + "\n| ".join([eqn.__str__().replace('\n', '\n| ') for eqn in self.eqn_list])
    
    def add_eqn(self, eqn: Equation) -> None:
        if eqn not in self.eqn_list:
            self.eqn_list.append(eqn)
        else:
            raise ValueError(f"Equation to be added is already in the System\nEquation: {eqn}")

    def remove_eqn(self, eqn: Equation) -> None:
        if eqn in self.eqn_list:
            self.eqn_list.remove(eqn)
        else:
            raise ValueError(f"Equation to be removed is not in System\nEquation: {eqn}")
    
    def lambda_residual(self, subs: dict[Basic, usernum], residual_types: list[str] = []) -> tuple[Callable[[ndarray], ndarray], list[Basic]]:
        if residual_types == []:
            residual_types = ["differential"] * len(self.eqn_list)

        lambda_residual_list: list[tuple[Callable[[ndarray], usernum | ndarray], list[Basic]]] = [eqn.lambda_residual(subs, residual_type) for eqn, residual_type in zip(self.eqn_list, residual_types)]
        func_list = [lambda_residual[0] for lambda_residual in lambda_residual_list]
        farg_list = [lambda_residual[1] for lambda_residual in lambda_residual_list]
        arg_list: list[Basic] = list(set().union(*[fargs for fargs in farg_list]))
        arg_index_map = {arg: i for i, arg in enumerate(arg_list)}
        farg_indices_list = [[arg_index_map[arg] for arg in farg] for farg in farg_list]

        def concatenate_wrapper(func_list: list[Callable[[ndarray], usernum | ndarray]], farg_list: list[list[Basic]]) -> Callable[[ndarray], ndarray]:
            return lambda args: concatenate([
                func(args[farg_indices_list]) if get_type_hints(func).get('return') == ndarray\
                else nparr([func(args[farg_indices_list])]).ravel()\
                for func, farg_indices_list in zip(func_list, farg_indices_list)
            ])

        return concatenate_wrapper(func_list, farg_list), arg_list

    def solve(self, subs: dict[Basic, usernum], residual_types: list[str] = [], guess_dict: dict[Basic, usernum] = {})-> dict[Basic, npfloat]:
        if residual_types == []:
            residual_types = ["differential"] * len(self.eqn_list)

        lambda_residual_func, residual_free_symbols_list = self.lambda_residual(subs, residual_types)
    
        exception_output = f"\
                System:\n    {self.__str__().replace('\n', '    ')}\n\
                subs:\n    {subs.__str__().replace('\n', '    ')}\n\
                residual_free_symbols_list:\n    {residual_free_symbols_list.__str__().replace('\n', '    ')}"

        if len(residual_free_symbols_list) > self.size:
            raise ValueError(f"Insufficient subs to solve matrix equation\n{exception_output}")
        
        guess_vect = [guess_dict.get(free_symbol) if free_symbol in guess_dict.keys() else 1 for free_symbol in residual_free_symbols_list]
               
        solution = nparr(fsolve(lambda_residual_func, guess_vect), dtype=npfloat)

        return dict(zip(residual_free_symbols_list, solution))        

