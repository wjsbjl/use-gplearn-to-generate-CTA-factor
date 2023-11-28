"""The functions used to create programs.

The :mod:`gplearn.functions` module contains all of the functions used by
gplearn programs. It also contains helper methods for a user to define their
own custom functions.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
from joblib import wrap_non_picklable_objects

__all__ = ['make_function']


class _Function(object):

    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """

    def __init__(self, function, name, arity):
        self.function = function
        self.name = name
        self.arity = arity

    def __call__(self, *args):
        return self.function(*args)


def make_function(*, function, name, arity, wrap=True):
    """Make a function node, a representation of a mathematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom functions is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    """
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(function, np.ufunc):
        if function.__code__.co_argcount != arity:
            raise ValueError('arity %d does not match required number of '
                             'function arguments of %d.'
                             % (arity, function.__code__.co_argcount))
    if not isinstance(name, str):
        raise ValueError('name must be a string, got %s' % type(name))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))

    # Check output shape
    args = [np.ones(10) for _ in range(arity)]
    try:
        function(*args)
    except (ValueError, TypeError):
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    # Check closure for zero & negative input arguments
    args = [np.zeros(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)
    args = [-1 * np.ones(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)

    if wrap:
        return _Function(function=wrap_non_picklable_objects(function),
                         name=name,
                         arity=arity)
    return _Function(function=function,
                     name=name,
                     arity=arity)


def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """Closure of log for zero and negative arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def _protected_inverse(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)


def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))


add2 = _Function(function=np.add, name='add', arity=2)
sub2 = _Function(function=np.subtract, name='sub', arity=2)
mul2 = _Function(function=np.multiply, name='mul', arity=2)
div2 = _Function(function=_protected_division, name='div', arity=2)
sqrt1 = _Function(function=_protected_sqrt, name='sqrt', arity=1)
log1 = _Function(function=_protected_log, name='log', arity=1)
neg1 = _Function(function=np.negative, name='neg', arity=1)
inv1 = _Function(function=_protected_inverse, name='inv', arity=1)
abs1 = _Function(function=np.abs, name='abs', arity=1)
max2 = _Function(function=np.maximum, name='max', arity=2)
min2 = _Function(function=np.minimum, name='min', arity=2)
sin1 = _Function(function=np.sin, name='sin', arity=1)
cos1 = _Function(function=np.cos, name='cos', arity=1)
tan1 = _Function(function=np.tan, name='tan', arity=1)
sig1 = _Function(function=_sigmoid, name='sig', arity=1)


# 新加的函数，还没进一步整理
def _andpn(x1, x2):
    return np.where((x1 > 0) & (x2 > 0), 1, -1)


def _orpn(x1, x2):
    return np.where((x1 > 0) | (x2 > 0), 1, -1)


def _ltpn(x1, x2):
    return np.where(x1 < x2, 1, -1)


def _gtpn(x1, x2):
    return np.where(x1 > x2, 1, -1)


def _andp(x1, x2):
    return np.where((x1 > 0) & (x2 > 0), 1, 0)


def _orp(x1, x2):
    return np.where((x1 > 0) | (x2 > 0), 1, 0)


def _ltp(x1, x2):
    return np.where(x1 < x2, 1, 0)


def _gtp(x1, x2):
    return np.where(x1 > x2, 1, 0)


def _andn(x1, x2):
    return np.where((x1 > 0) & (x2 > 0), -1, 0)


def _orn(x1, x2):
    return np.where((x1 > 0) | (x2 > 0), -1, 0)


def _ltn(x1, x2):
    return np.where(x1 < x2, -1, 0)


def _gtn(x1, x2):
    return np.where(x1 > x2, -1, 0)

def _if(x1, x2, x3):
    return np.where(x1 > 0, x2, x3)

def _delayy(x1):
    return np.nan_to_num(np.concatenate([[np.nan], x1[:-1]]), nan=0)


def _delta(x1):
    _ = np.nan_to_num(x1, nan=0)
    return _ - np.nan_to_num(_delayy(_), nan=0)


def _signedpower(x1):
    _ = np.nan_to_num(x1, nan=0)
    return np.sign(_) * (abs(_) ** 2)


def _decay_linear(x1):
    _ = pd.DataFrame({'x1': x1}).fillna(0)
    __ = _.fillna(method='ffill').rolling(10).mean() - _
    return np.array(__['x1'].fillna(0))

def _stdd(x1):
    return np.array([np.std(x1)] * len(x1))

def _rankk(x1):
    return x1.argsort()

gp_if = make_function(function=_if, name='if', arity=3)
gp_gtpn = make_function(function=_gtpn, name='gt', arity=2)
gp_andpn = make_function(function=_andpn, name='and', arity=2)
gp_orpn = make_function(function=_orpn, name='or', arity=2)
gp_ltpn = make_function(function=_ltpn, name='lt', arity=2)
gp_gtp = make_function(function=_gtp, name='gt', arity=2)
gp_andp = make_function(function=_andp, name='and', arity=2)
gp_orp = make_function(function=_orp, name='or', arity=2)
gp_ltp = make_function(function=_ltp, name='lt', arity=2)
gp_gtn = make_function(function=_gtn, name='gt', arity=2)
gp_andn = make_function(function=_andn, name='and', arity=2)
gp_orn = make_function(function=_orn, name='or', arity=2)
gp_ltn = make_function(function=_ltn, name='lt', arity=2)
gp_delayy = make_function(function=_delayy, name='delayy', arity=1)
gp_delta = make_function(function=_delta, name='_delta', arity=1)
gp_signedpower = make_function(function=_signedpower, name='_signedpower', arity=1)
gp_decayl = make_function(function=_decay_linear, name='_decayl', arity=1)
gp_stdd = make_function(function=_stdd, name='stdd', arity=1)
gp_rankk = make_function(function=_rankk, name='rankk', arity=1)

_function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'div': div2,
                 'sqrt': sqrt1,
                 'log': log1,
                 'abs': abs1,
                 'neg': neg1,
                 'inv': inv1,
                 'max': max2,
                 'min': min2,
                 'sin': sin1,
                 'cos': cos1,
                 'tan': tan1,

                'if': gp_if,
                'gtpn': gp_gtpn, # 新增
                'andpn': gp_andpn,
                'orpn': gp_orpn,
                'ltpn': gp_ltpn,
                'gtp': gp_gtp,
                'andp': gp_andp,
                'orp': gp_orp,
                'ltp': gp_ltp,
                'gtn': gp_gtn,
                'andn': gp_andn,
                'orn': gp_orn,
                'ltn': gp_ltn,
                'delayy': gp_delayy,
                'delta': gp_delta,
                'signedpower': gp_signedpower,
                'decayl': gp_decayl,
                'stdd': gp_stdd,
                'rankk': gp_rankk}


# def correlation(x, y, d):
#     """
#     :param x:
#     :param y:
#     :param d:
#     :return:
#     """
#     x = pd.Series(x)
#     y = pd.Series(y)
#     return x.rolling(d).corr(y).fillna(0)


# correlation5 = functools.partial(correlation, d=5)
# correlation5 = _Function(function=correlation5, name='correlation5', arity=2)
# correlation10 = functools.partial(correlation, d=10)
# correlation10 = _Function(function=correlation10, name='correlation10', arity=2)