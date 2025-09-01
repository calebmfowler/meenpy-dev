from scipy.optimize import fsolve
from sympy import sympify, lambdify, Basic, Expr, Matrix, Identity
from numpy import array as nparr, ndarray, float64 as npfloat, concatenate, prod, sum
from typing import Callable, get_type_hints
from pandas import Series, DataFrame, concat

usernum = int | float | npfloat

class Equation:
    def __init__(self, lhs, rhs, residual_type = "differential") -> None:
        self.init_lhs_rhs(lhs, rhs)
        self.init_shape_size()
        self.init_free_symbols()
        self.init_residual_type(residual_type)
       
    def init_lhs_rhs(self, lhs, rhs) -> None:
        self.lhs = sympify(lhs)
        self.rhs = sympify(rhs)
    
    def init_shape_size(self) -> None:
        self.shape: tuple[int, int] = (-1, -1)
        self.size: int = -1

    def init_free_symbols(self) -> None:
        self.free_symbols: set[Basic] = set()
    
    def init_residual_type(self, residual_type):
        self.residual_type = residual_type

    def get_subbed(self, subs: dict[Basic, usernum] = {}) -> "Equation":
        return self

    def get_residual(self, subs: dict[Basic, usernum] = {}) -> Expr | Matrix:
        return Expr()
    
    def get_lambda_residual(self, subs: dict[Basic, usernum] = {}) -> tuple[Callable, list[Basic]]:
        return (lambda *args, **kwargs: None, [])
    
    def solve(self, subs: dict[Basic, usernum]) -> dict[Basic, npfloat]:
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

    def init_residual_type(self, residual_type: str):
        if residual_type not in ["differential", "left_rational", "right_rational"]:
            raise ValueError(f"Invalid residual_type = '{residual_type}'")
        self.residual_type = residual_type
    
    def get_subbed(self, subs: dict[Basic, usernum] = {}) -> "ScalarEquation":
        return ScalarEquation(self.lhs.subs(subs), self.rhs.subs(subs))

    def get_residual(self, subs: dict[Basic, usernum] = {}) -> Expr:
        subbed_eqn = self.get_subbed(subs)

        if self.residual_type == "differential":
            return sympify(subbed_eqn.lhs) - sympify(subbed_eqn.rhs)
        
        elif self.residual_type == "left_rational":
            return sympify(subbed_eqn.lhs) / sympify(subbed_eqn.rhs) - 1
        
        elif self.residual_type == "right_rational":
            return sympify(subbed_eqn.rhs) / sympify(subbed_eqn.lhs) - 1
        
        else:
            raise ValueError(f"Invalid residual_type = '{self.residual_type}'")
    
    def get_lambda_residual(self, subs: dict[Basic, usernum] = {}) -> tuple[Callable[[ndarray], usernum], list[Basic]]:
        residual = self.get_residual(subs)
        residual_free_symbols_list = list(residual.free_symbols)
        lambda_residual_func: Callable[[tuple], usernum] = lambdify(residual_free_symbols_list, residual)

        def unpack_wrapper(func: Callable[[tuple], usernum]):
            return lambda arr: func(*arr)

        return unpack_wrapper(lambda_residual_func), residual_free_symbols_list
        
    def solve(self, subs: dict[Basic, usernum], guess: usernum = 1) -> dict[Basic, npfloat]:
        residual_lambda, residual_free_symbols_list = self.get_lambda_residual(subs)

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

    def init_residual_type(self, residual_type: str | list[str]):
        if isinstance(residual_type, str) and residual_type not in ["differential", "left_inversion", "right_inversion"]:
            raise ValueError(f"Invalid residual_type = '{residual_type}'")
        
        if isinstance(residual_type, str) and residual_type in ["left_inversion", "right_inversion"] and self.shape[0] != self.shape[1]:
            raise ValueError(f"Invalid residual_type = '{self.residual_type}' for non-square MatrixEquation of shape = {self.shape}")
        
        if isinstance(residual_type, list) and any([subtype not in [] for subtype in residual_type]):
            raise ValueError(f"Invalid residual_type = '{residual_type}'")
        
        self.residual_type = residual_type

    def get_subbed(self, subs: dict[Basic, usernum] = {}) -> "MatrixEquation":
        return MatrixEquation(self.lhs.subs(subs), self.rhs.subs(subs))

    def get_residual(self, subs: dict[Basic, usernum] = {}) -> Matrix:
        subbed_eqn = self.get_subbed(subs)

        if self.residual_type == "differential":
            return sympify(subbed_eqn.lhs) - sympify(subbed_eqn.rhs)
        
        if self.residual_type == "left_inversion":
            return subbed_eqn.lhs**-1 @ subbed_eqn.rhs - Identity(self.shape[0])
        
        if self.residual_type == "right_inversion":
            return subbed_eqn.lhs @ subbed_eqn.rhs**-1 - Identity(self.shape[0])
                 
        else:
            raise ValueError(f"Invalid residual_type = '{self.residual_type}'")

    def get_lambda_residual(self, subs: dict[Basic, usernum] = {}) -> tuple[Callable[[ndarray], ndarray], list[Basic]]:
        residual = self.get_residual(subs)
        residual_free_symbols_list: list[Basic] = list(residual.free_symbols)
        shaped_lambda_residual_func: Callable[[tuple], ndarray] = lambdify(residual_free_symbols_list, residual)

        def unpack_ravel_wrapper(func: Callable[[tuple], ndarray]) -> Callable[[ndarray], ndarray]:
            return lambda args: nparr(func(*args)).ravel()

        return unpack_ravel_wrapper(shaped_lambda_residual_func), residual_free_symbols_list

    def solve(self, subs: dict[Basic, usernum], guess_dict: dict[Basic, usernum] = {}) -> dict[Basic, npfloat]:
        lambda_residual_func, residual_free_symbols_list = self.get_lambda_residual(subs)

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


class Table:
    def __init__(self, df: DataFrame, indexing_columns: list[str] = [], preformatted: bool = False) -> None:
        if preformatted:
            self.df = df
        else:
            self.df = df.set_index(indexing_columns)
        
        self.len_multiindex = self.df.index.nlevels
        
        self.multiindex_adjacency = DataFrame(
            [[self.is_adjacent_multiindex(I, J) for J in self.df.index] for I in self.df.index],
            index=self.df.index,
            columns=self.df.index
        )

    def is_adjacent_multiindex(self, I: tuple, J: tuple) -> int:
        index_inequality = [1 if i != j else 0 for i, j in zip(I, J)]
        num_unequal = sum(index_inequality)

        if num_unequal == 1:
            return index_inequality.index(1)
        else:
            return -1
    
    def get_multiindex_interpolation(self, adj_index: int, A: tuple, B: tuple, col: str, val: usernum, is_proper_column: bool) -> Series:
        if is_proper_column:
            val_A, val_B = self.df.at[A, col], self.df.at[B, col]
        else:
            val_A, val_B = A[self.df.index.names.index(col)], B[self.df.index.names.index(col)]

        row_A, row_B = self.df.loc[A], self.df.loc[B]
        x = (val - val_A) / (val_B - val_A)

        I = A[:adj_index] + (A[adj_index] * (1 - x) + B[adj_index] * (x),) + A[adj_index + 1:]
        row_I: Series = row_A * (1 - x) + row_B * (x)
        row_I.name = I
        return row_I    

    def get_subbed(self, subs: dict[str, usernum] = {}) -> "Table":
        subbed_df = self.df

        for col, val in zip(subs.keys(), subs.values()):
            is_proper_column = col in subbed_df.columns
            is_index_column = col in subbed_df.index.names

            if not is_proper_column and not is_index_column:
                raise ValueError(f"Given column {col} is not a proper column or an index column in Table\n{self.__str__()}")
            
            elif is_proper_column:
                equal_df = subbed_df[subbed_df[col] == val]

                interpolation_multiindex_adjacency = self.multiindex_adjacency.loc[
                    subbed_df[subbed_df[col] < val].index.values,
                    subbed_df[subbed_df[col] > val].index.values
                ].stack(list(range(self.len_multiindex)), future_stack=True)

            else:
                equal_df = subbed_df[subbed_df.index.get_level_values(col) == val]

                interpolation_multiindex_adjacency = self.multiindex_adjacency.loc[
                    subbed_df[subbed_df.index.get_level_values(col) < val].index.values,
                    subbed_df[subbed_df.index.get_level_values(col) > val].index.values
                ].stack(list(range(self.len_multiindex)), future_stack=True)

            interpolation_df = DataFrame([
                self.get_multiindex_interpolation(adj_index, AB[:self.len_multiindex], AB[self.len_multiindex:], col, val, is_proper_column)
                for AB, adj_index in zip(
                    interpolation_multiindex_adjacency.index.values,
                    interpolation_multiindex_adjacency.values
                )
                if adj_index != -1
            ])

            if not interpolation_df.empty:
                interpolation_df.index.names = subbed_df.index.names

            subbed_df = concat([equal_df, interpolation_df])

        return Table(subbed_df, preformatted=True)

    def __getitem__(self, *args, **kwargs):
        return self.df.__getitem__(*args, **kwargs)

    def __str__(self):
        return self.df.__str__()


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
    
    def get_subbed_sys(self, subs: dict[Basic, usernum] = {}) -> "System":
        subbed_eqn_list = [eqn.get_subbed(subs) for eqn in self.eqn_list]

        return System(subbed_eqn_list)

    def get_lambda_residual(self, subs: dict[Basic, usernum]) -> tuple[Callable[[ndarray], ndarray], list[Basic]]:
        lambda_residual_list: list[tuple[Callable[[ndarray], usernum | ndarray], list[Basic]]] = [eqn.get_lambda_residual(subs) for eqn in self.eqn_list]
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

    def solve(self, subs: dict[Basic, usernum], guess_dict: dict[Basic, usernum] = {})-> dict[Basic, npfloat]:
        lambda_residual_func, residual_free_symbols_list = self.get_lambda_residual(subs)
    
        exception_output = f"\
                System:\n    {self.__str__().replace('\n', '    ')}\n\
                subs:\n    {subs.__str__().replace('\n', '    ')}\n\
                residual_free_symbols_list:\n    {residual_free_symbols_list.__str__().replace('\n', '    ')}"

        if len(residual_free_symbols_list) > self.size:
            raise ValueError(f"Insufficient subs to solve matrix equation\n{exception_output}")
        
        guess_vect = [guess_dict.get(free_symbol) if free_symbol in guess_dict.keys() else 1 for free_symbol in residual_free_symbols_list]
               
        solution = nparr(fsolve(lambda_residual_func, guess_vect), dtype=npfloat)

        return dict(zip(residual_free_symbols_list, solution))        

