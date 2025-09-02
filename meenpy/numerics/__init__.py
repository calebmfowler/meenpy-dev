from scipy.optimize import fsolve
from sympy import sympify, lambdify, Basic, Expr, Matrix, Identity
from numpy import array as nparr, ndarray, float64 as npfloat, concatenate, prod, sum, min, argmin
from typing import Callable, get_type_hints
from pandas import Series, DataFrame, concat

usernum = int | float | npfloat

class Equation:
    def __init__(self):
        self.size = -1

    def get_subbed(self, subs: dict) -> "Equation":
        return self
    
    def get_lambda_residual(self, subs: dict = {}) -> tuple[Callable, list]:
        return (lambda *args, **kwargs: None, [])
    
    def solve(self, subs: dict = {}) -> dict:
        return {}


class AlgebraicEquation(Equation):
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

    def get_subbed(self, subs: dict[Basic, usernum]) -> "AlgebraicEquation":
        return self

    def get_residual(self, subs: dict[Basic, usernum] = {}) -> Expr | Matrix:
        return Expr()
    
    def get_lambda_residual(self, subs: dict[Basic, usernum] = {}) -> tuple[Callable, list[Basic]]:
        return (lambda *args, **kwargs: None, [])
    
    def solve(self, subs: dict[Basic, usernum] = {}) -> dict[Basic, npfloat]:
        return {}

    def __str__(self) -> str:
        return self.lhs.__str__() + ' = ' + self.rhs.__str__()


class ScalarEquation(AlgebraicEquation):
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
    
    def get_subbed(self, subs: dict[Basic, usernum]) -> "ScalarEquation":
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
        
    def solve(self, subs: dict[Basic, usernum] = {}, guess: usernum = 1) -> dict[Basic, npfloat]:
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


class MatrixEquation(AlgebraicEquation):
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

    def get_subbed(self, subs: dict[Basic, usernum]) -> "MatrixEquation":
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

    def solve(self, subs: dict[Basic, usernum] = {}, guess_dict: dict[Basic, usernum] = {}) -> dict[Basic, npfloat]:
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


class TabularEquation(Equation):
    def __init__(self, df: DataFrame, indexing_columns: list[str] = [], preformatted: bool = False) -> None:
        if preformatted:
            self.df = df
        else:
            self.df = df.set_index(indexing_columns)
        
        self.len_multiindex = self.df.index.nlevels

        self.at = self.df.at
        self.columns = self.df.columns
        self.index = self.df.index
        self.iloc = self.df.iloc
        self.loc = self.df.loc
        
        self.multiindex_adjacency = DataFrame(
            [[self._how_is_multiindex_adjacent(I, J)[0] for J in self.df.index] for I in self.df.index],
            index=self.df.index,
            columns=self.df.index
        )
    
    def _how_is_multiindex_adjacent(self, I: tuple, J: tuple) -> tuple[int, bool]:
        index_inequality = [1 if i != j else 0 for i, j in zip(I, J)]
        num_unequal = sum(index_inequality)
        
        if num_unequal == 1:
            index = index_inequality.index(1)
            return (index, I[index] > J[index])

        else:
            return (-1, False)
    
    def _get_interpolated_row_for_substitution(self, adj_index: int, A: tuple, B: tuple, col: str, val: usernum, is_proper_column: bool) -> Series:
        if is_proper_column:
            val_A, val_B = self.at[A, col], self.at[B, col]
        else:
            val_A, val_B = A[adj_index], B[adj_index]

        row_A, row_B = self.loc[A], self.loc[B]
        x = (val - val_A) / (val_B - val_A)

        N = A[:adj_index] + (A[adj_index] * (1 - x) + B[adj_index] * (x),) + A[adj_index + 1:]
        row_N: Series = row_A * (1 - x) + row_B * (x)
        row_N.name = N

        return row_N

    def get_subbed(self, subs: dict[str, usernum]) -> "TabularEquation":
        subbed_df = self.df

        for col, val in zip(subs.keys(), subs.values()):
            is_proper_column = col in subbed_df.columns
            is_index_column = col in subbed_df.index.names

            if not is_proper_column and not is_index_column:
                raise ValueError(f"Given column {col} is not a proper column or an index column in Table\n{self.__str__()}")
            
            if is_proper_column:
                equal_df = subbed_df[subbed_df[col] == val]

                lesser_index_candidates = subbed_df[subbed_df[col] < val]
                greater_index_candidates = subbed_df[subbed_df[col] > val]

            else:
                equal_df = subbed_df[subbed_df.index.get_level_values(col) == val]
                
                lesser_index_candidates = subbed_df[subbed_df.index.get_level_values(col) < val]
                greater_index_candidates = subbed_df[subbed_df.index.get_level_values(col) > val]

            interpolation_candidates_multiindex_adjacency = self.multiindex_adjacency.loc[
                lesser_index_candidates.index.values,
                greater_index_candidates.index.values
            ].stack(list(range(self.len_multiindex)), future_stack=True)

            interpolated_df = DataFrame([
                self._get_interpolated_row_for_substitution(adj_index, AB[:self.len_multiindex], AB[self.len_multiindex:], col, val, is_proper_column)
                for AB, adj_index in zip(
                    interpolation_candidates_multiindex_adjacency.index.values,
                    interpolation_candidates_multiindex_adjacency.values
                )
                if adj_index != -1
            ])

            if not interpolated_df.empty:
                interpolated_df.index.names = subbed_df.index.names

            subbed_df = concat([equal_df, interpolated_df])

        return TabularEquation(subbed_df, preformatted=True)

    def _get_multiindex_seperation(self, N: tuple, I: tuple) -> npfloat:
        return sum([n - i for n, i in zip(N, I)])

    def _get_interpolated_row_for_lambda_residual(self, adj_index: int, A: tuple, N: tuple, B: tuple) -> Series:
        val_A, val_N, val_B = A[adj_index], N[adj_index], B[adj_index]
        x = (val_N - val_A) / (val_B - val_A)
        row_A, row_B = self.loc[A], self.loc[B]
        row_N: Series = row_A * (1 - x) + row_B * (x)
        row_N.name = N

        return row_N

    def get_lambda_residual(self, subs: dict[str, usernum] = {}) -> tuple[Callable[[ndarray], ndarray], list[str]]:
        subbed_table = self.get_subbed(subs)

        def lambda_residual(vals: ndarray) -> ndarray:
            N = tuple(vals[:subbed_table.len_multiindex])
            col_vals_N = vals[subbed_table.len_multiindex:]
            row_N = Series(dict(zip(subbed_table.columns, col_vals_N)), name=N)

            if N in subbed_table.index:
                return nparr(row_N - subbed_table.loc[N])
            
            novel_multiindex_adjacency = [subbed_table._how_is_multiindex_adjacent(I, N) for I in subbed_table.index]

            lesser_index_candidates = Series({
                I: adj_index
                for I, (adj_index, is_greater) in zip(subbed_table.index, novel_multiindex_adjacency)
                if adj_index != -1 and not is_greater
            })

            greater_index_candidates = Series({
                I: adj_index
                for I, (adj_index, is_greater) in zip(subbed_table.index, novel_multiindex_adjacency)
                if adj_index != -1 and is_greater
            })

            interpolation_candidates_multiindex_adjacency = subbed_table.multiindex_adjacency.loc[
                lesser_index_candidates.index.values,
                greater_index_candidates.index.values
            ].stack(list(range(subbed_table.len_multiindex)), future_stack=True)

            interpolated_df = DataFrame([
                subbed_table._get_interpolated_row_for_lambda_residual(adj_index, AB[:subbed_table.len_multiindex], N, AB[subbed_table.len_multiindex:])
                for AB, adj_index in zip(
                    interpolation_candidates_multiindex_adjacency.index.values,
                    interpolation_candidates_multiindex_adjacency.values
                )
                if adj_index != -1
            ])
            interpolated_df.index.names = subbed_table.index.names

            if not interpolated_df.empty:
                return nparr(row_N - interpolated_df.iloc[0])

            multiindex_seperation = [subbed_table._get_multiindex_seperation(N, I) for I in subbed_table.index]
            i_nearest_multiindex = argmin(multiindex_seperation)
            nearest_multiindex: tuple = subbed_table.index.values[i_nearest_multiindex]
            nearest_multiindex_seperation: npfloat = multiindex_seperation[i_nearest_multiindex]

            return lambda_residual(nparr(list(nearest_multiindex) + col_vals_N.tolist())) * 1e3 * nearest_multiindex_seperation**2

        return lambda_residual, [str(name) for name in subbed_table.index.names] + subbed_table.columns.to_list()

    def __getitem__(self, *args, **kwargs):
        return self.df.__getitem__(*args, **kwargs)

    def __str__(self):
        return self.df.__str__()


class System:
    def __init__(self, eqn_list: list[Equation]) -> None:
        self.eqn_list = eqn_list
        self.size = sum([eqn.size for eqn in self.eqn_list])
    
    def __str__(self) -> str:
        return "| " + "\n| ".join([eqn.__str__().replace('\n', '\n| ') for eqn in self.eqn_list])
        
    def get_subbed(self, subs: dict[Basic, usernum] = {}) -> "System":
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

