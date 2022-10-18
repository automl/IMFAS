import logging
import copy
from itertools import combinations_with_replacement
from typing import Callable, Tuple, List, Dict

from imfas.models.baselines.lcdb_parametric_lc import ParametricLC, BestParametricLC
import numpy as np
import pysmt.fnode
from pysmt.shortcuts import Symbol, And, GE, LE, Times, Pow, Minus, Plus, Exists, Real, Div, reset_env, Xor, Or
from pysmt.typing import REAL
from pysmt.shortcuts import qelim, is_sat
import inspect

logger = logging.getLogger(__name__)
x, y, z = [Symbol(s, REAL) for s in "xyz"]


def pow2(x: int, a: pysmt.fnode.FNode, b: float):
    return Times(-a, Real(x ** (- b)))


def pow3(x: int, a: pysmt.fnode.FNode, b: pysmt.fnode.FNode, c: float):
    return Minus(a, Times(b, Real(x ** (-c))))


def log2(x: int, a: pysmt.fnode.FNode, b: pysmt.fnode.FNode):
    return Times(a, Plus(Real(np.log(x)), b))


def exp3(x: int, a: pysmt.fnode.FNode, b: float, c: pysmt.fnode.FNode):
    return Plus(Times(a, Real(np.exp(-b * x))), c)


def exp2(x: int, a: pysmt.fnode.FNode, b: float):
    return Times(a, Real(np.exp(-b * x)))


def lin2(x: int, a: pysmt.fnode.FNode, b: pysmt.fnode.FNode):
    return Plus(Times(a, Real(x)), b)


# vap3 is not applicable for smt solvers


def mmf4(x: int, a: pysmt.fnode.FNode, b: pysmt.fnode.FNode, c: pysmt.fnode.FNode, d: float):
    return Div(
        Plus(Times(a, b), Times(c, Real(x ** d))),
        Plus(b, Real(x ** d))
    )


def wbl4(x: int, a: float, b: pysmt.fnode.FNode, c: pysmt.fnode.FNode, d: float):
    return Minus(
        c,
        Times(b, Real(np.exp(-a * (x ** d))))
    )


def exp4(x: int, a: float, b: float, c: pysmt.fnode.FNode, d: float):
    return Minus(c, Real(np.exp(-a * (x ** d) + b)))


def expp3(x: int, a: float, b: float, c: pysmt.fnode.FNode):
    return Minus(c, Real(np.exp((x - b) ** a)))


def pow4(x: int, a: pysmt.fnode.FNode, b: pysmt.fnode.FNode, c: float, d: pysmt.fnode.FNode):
    return Minus(
        a,
        Times(b, Pow(Plus(Real(x), d), Real(c)))
    )


def expd3(x: int, a: pysmt.fnode.FNode, b: float, c: pysmt.fnode.FNode):
    return Minus(
        c,
        Times(
            Minus(c, a),
            Real(np.exp(-b * x))
        )
    )


def logpower3(x: int, a: pysmt.fnode.FNode, b: float, c: float):
    return Div(a, Real(1 + (x / np.exp(b)) ** c))


def last1(x: int, a: pysmt.fnode.FNode):
    return Minus(Plus(a, Real(x)), Real(x))


x_start = 1
x_end = 20

apow2 = Symbol('apow2', REAL)
bpow2 = 0.5

apow3 = Symbol('apow3', REAL)
bpow3 = Symbol('bpow3', REAL)
cpow3 = 0.1
f = Exists([apow2, apow3, bpow3], And(
    GE(pow2(x_start, apow2, bpow2), pow3(x_start, apow3, bpow3, cpow3)),
    LE(pow2(x_end, apow2, bpow2), pow3(x_end, apow3, bpow3, cpow3)),
))

qf_f = qelim(f, solver_name="z3")
print("Quantifier-Free equivalent: %s" % qf_f)
# Quantifier-Free equivalent: (7/2 <= (z + y))

res = is_sat(qf_f, solver_name="msat")
print("SAT check using MathSAT: %s" % res)
import pdb
pdb.set_trace()

# SAT check using MathSAT: True


class SynthericParametericCurves(ParametricLC):
    def __init__(self, function: str, budgets: List[int], restarts: int = 10, parameters_lc: np.ndarray = np.ones(4)):
        super(SynthericParametericCurves, self).__init__(function, budgets, restarts)
        self.parameters_lc = parameters_lc[:self.n_parameters]
        if np.isnan(self.predict(x[0])):
            raise ValueError('Invalid Parameteric values')

    def fit(self, x: np.ndarray, Y: np.ndarray, ) -> None:
        """
        :param parameters_lc: np.ndarray, a ndarray that
        """
        pass


def get_fixed_and_free_paras(func: Callable) -> Tuple[List, List]:
    """get the fixed and free parameters that can be passed to the SMT solver"""
    free_params = []
    fixed_params = []
    for key, value in inspect.signature(func).parameters:
        if key == 'x':
            # 'x' is used to pass budget values
            continue
        if value.annotation.__name__ == 'FNode':
            fixed_params.append(key)
        else:
            free_params.append(key)
    return free_params, fixed_params


class SynthericParametericCurvesSets(BestParametricLC):
    functionals = {
        'pow2': pow2,
        'pow3': pow3,
        'log2': log2,
        'exp3': exp3,
        'exp2': exp2,
        'lin2': lin2,
        # 'vap3': lambda x, a, b, c: np.exp(a + b / x + c * np.log(x)),
        'mmf4': mmf4,
        'wbl4': wbl4,
        'exp4': exp4,
        'expp3': expp3,
        # fun = lambda x: a * np.exp(-b*x) + c

        'pow4': pow4,  # has to closely match pow3,
        'expd3': expd3,
        'logpower3': logpower3,
        'last1': last1
    }

    def __init__(self, budgets: List[int], restarts: int = 10, seed=0):
        super(SynthericParametericCurvesSets, self).__init__(budgets=budgets, restarts=restarts)
        self.rng = np.random.RandomState(seed)

    def fit(self, num_learning_curves: int,
            fixed_para_init_values: Dict,
            restarts: int = 10,
            min_n_intersections: int = 5,
            intersection_budget_bounds: List[List[int]] = [[1, 5], [5, 10], [25,50]]) -> None:
        """
        Here I propose an incremental approach to generate a set of curves that has min_n_intersections intersection
        points: every time we pick 2 curves and check if they have an intersection points. One of these 2 curves must
        have free parameters. we try to generate a set of learning curves


        :param num_learning_curves: int, The number of learning curves that we wish to generate.
        :param fixed_para_init_values: Dict, a dictionary that indicate the proper exponent values for all the functions
            as they cannot be solved with the SMT solvers
        :param restarts: int, the number of restarts to use for the optimization.
        :param min_n_intersections: int the minimal number of intersection points
        :param intersection_budget_bounds: the bounds where the intersection needs to happen
        """
        n_generated_lcs = 0
        curves_generated: List[SynthericParametericCurves] = []

        n_curves_checked = 2  # This parameter indicates the amount of the learning curves that we want to compare at
        # each round

        while len(curves_generated) < num_learning_curves:
            for idx_restart in range(restarts):
                reset_env()
                all_fixed_lc = True
                while not all_fixed_lc:
                    lc_indices = self.rng.choice(num_learning_curves, n_curves_checked, replace=False).sort()
                    lc_is_generated = lc_indices < len(curves_generated)
                    n_curves_fixed = sum(lc_is_generated)
                    if n_curves_fixed < n_curves_checked:
                        all_fixed_lc = False

                x_start, x_end = intersection_budget_bounds[self.rng.randint(len(intersection_budget_bounds))]
                fixed_curves = [curves_generated[lc_idx] for lc_idx in lc_indices[:n_curves_fixed]]
                fixed_curve_bound_values = [[curve.predict(x_start), curve.predict(x_end)] for curve in fixed_curves]

                free_lcs_types = self.rng.choice(list(self.functionals.keys()), n_curves_checked - n_curves_fixed)
                lc_types, count_types = np.unique(free_lcs_types, return_counts=True)

                all_tested_lcs = []
                all_free_pars = []

                free_curve_bound_values = {}
                free_curve_pars = {}

                for lc_type, count_type in zip(lc_types, count_types):
                    free_params, fixed_params = get_fixed_and_free_paras(self.functionals[lc_type])
                    all_tested_lcs.extend([f'{lc_type}_{idx}' for idx in range(count_type)])

                    # fixme find a way to initialize fixed params
                    for idx in range(count_type):
                        lc_name = f'{lc_type}_{idx}'
                        func_params = copy.deepcopy(fixed_para_init_values[lc_type])
                        for free_par in free_params:
                            par = Symbol(f'{lc_type}_{idx}_{free_par}', REAL)
                            func_params[free_par] = par
                            all_free_pars.append(par)

                        free_curve_pars[lc_name] = func_params
                        lc_value_start = self.functionals[lc_type](x=x_start, **func_params)
                        lc_value_end = self.functionals[lc_type](x=x_end, **func_params)
                        free_curve_bound_values[lc_name] = ([lc_value_start, lc_value_end])

                smt_handles = []
                # compare free-par curves with fixed_par_curves
                for free_curve_bound in free_curve_bound_values.values():
                    for fix_bound in fixed_curve_bound_values:
                        smt_handles.append(
                            Xor(
                                GE(free_curve_bound[0], fix_bound[1]),
                                GE(free_curve_bound[1], fix_bound[1]),
                            )
                        )
                for fre_curve_bound_1, fre_curve_bound_2 in combinations_with_replacement(free_curve_bound_values):
                    smt_handles.append(
                        Xor(
                            GE(fre_curve_bound_1[0], fre_curve_bound_2[1]),
                            GE(fre_curve_bound_1[1], fre_curve_bound_2[1]),
                        )
                    )
                f = Exists(all_free_pars, Or(*smt_handles))
                qf_f = qelim(f, solver_name="z3")

    def sample_lc(self):
        pass
