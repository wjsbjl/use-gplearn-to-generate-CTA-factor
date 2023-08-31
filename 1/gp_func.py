import numpy as np
import pandas as pd
from gplearn.functions import make_function

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
