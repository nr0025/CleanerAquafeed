"""Direct optimisation benchmarks for the screened shortlist.

* direct LP: minimum of Eq. (20) subject to the formulation constraints and rules R1, R2 and R4;
* the same LP with an iso-energetic constraint (assimilable energy equal to the benchmark), which
  fixes predicted growth at that of the benchmark (see growth.py);
* a weighted-sum multi-objective formulation adapted from Garcia-Launay et al. (2018);
* re-solving the direct LP with altered inclusion limits;
* re-formulating the least-cost low-fishmeal reference diet with altered limits on total plant meals.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import dataclasses

from .data import FISHMEAL, IMPACTS
from .evaluation import Rules, evaluate
from .formulation import TOTAL_KG, LinearModel, NutrientSpec, normalised_coefficients, reference_diets


def _add_rules(m: LinearModel, x_bench: np.ndarray, rules: Rules, gwp_strict: float = 1e-9) -> LinearModel:
    m = m.copy()
    p, _, e = normalised_coefficients(m.db, x_bench)
    m.le(p, 1.0 + rules.cost_tol)                      # R1
    for j in range(e.shape[0]):
        m.le(e[j], 1.0 + rules.burden_tol)             # R4
    if rules.require_gwp:
        m.le(e[0], 1.0 - gwp_strict)                   # R2
    return m


def direct_lp(m: LinearModel, x_bench: np.ndarray, rules: Rules, energy: str | None = None, overrides: dict | None = None):
    """Best robust formulation by a single LP.

    energy: None, 'equal' (E_A = E_A of the benchmark) or 'at_least' (E_A >= benchmark).
    overrides: {ingredient: (lb, ub)} applied to the inclusion limits.
    R3 is not imposed; whether the optimum satisfies it is reported.
    """
    mm = _add_rules(m, x_bench, rules)
    db = m.db
    for name, (lo, hi) in (overrides or {}).items():
        j = db.idx(name)
        mm.lb[j] = lo if lo is not None else mm.lb[j]
        mm.ub[j] = hi if hi is not None else mm.ub[j]
    ea_b = db.ea_coef @ x_bench
    if energy == "equal":
        mm.eq(db.ea_coef, ea_b)
    elif energy == "at_least":
        mm.ge(db.ea_coef, ea_b)
    p, _, e = normalised_coefficients(db, x_bench)
    c = rules.w_cost * p + rules.w_env * e.mean(axis=0)
    res = mm.solve(c)
    if res.status != 0:
        return None
    return np.clip(res.x, mm.lb, mm.ub)


def garcia_launay_front(m: LinearModel, x_bench: np.ndarray, alphas=None, impact_cap: float = 1.05, cp_cap: bool = True) -> pd.DataFrame:
    """min (1 - alpha) C_r + alpha E_m with every indicator <= impact_cap times the benchmark and,
    as in the original method, crude protein not above that of the benchmark."""
    db = m.db
    alphas = np.round(np.arange(0, 1.0001, 0.01), 2) if alphas is None else alphas
    mm = m.copy()
    p, _, e = normalised_coefficients(db, x_bench)
    for j in range(e.shape[0]):
        mm.le(e[j], impact_cap)
    if cp_cap:
        mm.le(db.frac("cp"), db.frac("cp") @ x_bench + 1e-9)
    rows, X = [], []
    for a in alphas:
        res = mm.solve((1 - a) * p + a * e.mean(axis=0))
        if res.status != 0:
            continue
        x = np.clip(res.x, mm.lb, mm.ub)
        X.append(x)
        rows.append({"alpha": float(a), "C_r": float(p @ x), "E_m": float((e @ x).mean())})
    df = pd.DataFrame(rows)
    df[db.names] = np.array(X)
    return df


def alpha_lim(front: pd.DataFrame) -> float:
    """Garcia-Launay et al. (2018): the alpha beyond which the marginal decrease of the environmental
    index is smaller than the marginal increase of the cost index."""
    f = front.sort_values("alpha").reset_index(drop=True)
    last = 0
    for i in range(1, len(f)):
        dC = f.C_r[i] - f.C_r[last]
        dE = f.E_m[last] - f.E_m[i]
        if dC <= 1e-12 and dE <= 1e-12:
            continue
        if dE < dC:
            return float(f.alpha[last])
        last = i
    return float(f.alpha[last])


def distance_to_front(front: pd.DataFrame, C_r: np.ndarray, E_m: np.ndarray) -> np.ndarray:
    """Vertical distance (in E_m) of points above the piecewise-linear cost-environment front."""
    pts = np.unique(np.round(front[["C_r", "E_m"]].to_numpy(), 9), axis=0)
    pts = pts[np.argsort(pts[:, 0])]
    lower = np.interp(C_r, pts[:, 0], pts[:, 1], left=np.nan, right=pts[-1, 1])
    return E_m - lower


def inclusion_limit_sensitivity(m: LinearModel, x_bench: np.ndarray, rules: Rules, cases: list[tuple[str, dict]]) -> pd.DataFrame:
    rows = []
    db = m.db
    for label, ov in cases:
        x = direct_lp(m, x_bench, rules, overrides=ov)
        if x is None:
            rows.append({"case": label, "problem": "direct LP, Eq. (20)", "feasible": False})
            continue
        ev = evaluate(db, x[None, :], x_bench, rules).iloc[0]
        rows.append(_limit_row(label, "direct LP, Eq. (20)", db, x, ev))
    return pd.DataFrame(rows)


def _limit_row(label: str, problem: str, db, x: np.ndarray, ev: pd.Series) -> dict:
    return {"case": label, "problem": problem, "feasible": True, "cleaner_score": ev.cleaner_score,
            "cost_change_pct": ev.cost_change_pct, "mean_env_change_pct": ev.mean_env_change_pct,
            **{f"{k}_change_pct": ev[f"{k}_change_pct"] for k in IMPACTS}, "n_reduced": int(ev.n_reduced),
            "fishmeal_kg_t": x[db.idx(FISHMEAL)], "lipid_pct": x @ db.frac("lipid") / TOTAL_KG * 100,
            "plant_meals_pct": x @ db.plant_mask / TOTAL_KG * 100,
            **{n: round(float(v), 2) for n, v in zip(db.names, x) if v > 1e-6}}


def plant_limit_reference(db, spec: NutrientSpec, x_bench: np.ndarray, rules: Rules, level: float, label: str,
                          caps=(45.0, 50.0, None)) -> pd.DataFrame:
    """Least-cost diet at a fixed fishmeal level (Eqs. 1-6) re-formulated with different limits on total plant
    meals; changes are expressed relative to the benchmark of the base case (x_bench)."""
    rows = []
    for cap in caps:
        s = dataclasses.replace(spec, plant_max=cap)
        x = reference_diets(db, s, levels=(level,), labels=(label,)).loc[label].to_numpy(float)
        ev = evaluate(db, x[None, :], x_bench, rules).iloc[0]
        txt = "no limit" if cap is None or cap >= 100 else f"<= {cap:g} %"
        rows.append(_limit_row(f"{label} least cost, plant meals {txt}", "least-cost reference diet, Eq. (1)", db, x, ev))
    return pd.DataFrame(rows)


def lp_timing(m: LinearModel, x_bench: np.ndarray, n: int = 200, seed: int = 0) -> float:
    """Mean wall time (ms) of one weighted LP solve."""
    rng = np.random.default_rng(seed)
    p, _, e = normalised_coefficients(m.db, x_bench)
    basis = np.vstack([p, e])
    t = time.perf_counter()
    for _ in range(n):
        m.solve(rng.dirichlet(np.ones(basis.shape[0])) @ basis)
    return 1e3 * (time.perf_counter() - t) / n
