"""Price and LCA-coefficient Monte Carlo (Eqs. 21-22) and sensitivity of the eligibility thresholds."""
from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data import IngredientDB
from .evaluation import Rules, evaluate


def sample_prices(db: IngredientDB, n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Prices (CNY/kg), uniform between the low and high estimates.

    rho = 0 gives independent draws; rho > 0 imposes one common factor through a Gaussian copula,
    so that every pair of ingredient prices has latent correlation rho.
    """
    k = len(db)
    if rho <= 0:
        u = rng.random((n, k))
    else:
        z = np.sqrt(rho) * rng.standard_normal((n, 1)) + np.sqrt(1 - rho) * rng.standard_normal((n, k))
        u = norm.cdf(z)
    return db.price_low + u * (db.price_high - db.price_low)


def price_risk(db: IngredientDB, X: np.ndarray, x_bench: np.ndarray, n: int, rho: float, seed: int, cost_tol: float = 0.10) -> pd.DataFrame:
    """Cost-risk probability P_risk (Eq. 22) and percentiles of cost under price uncertainty."""
    rng = np.random.default_rng(seed)
    P = sample_prices(db, n, rho, rng)              # (n, k)
    cost = X @ P.T                                   # (m, n) CNY per tonne
    cost_b = P @ x_bench                             # (n,)
    ratio = cost / cost_b
    return pd.DataFrame({
        "p_risk": (ratio > 1.0 + cost_tol).mean(axis=1),
        "cost_p05_cny_t": np.percentile(cost, 5, axis=1),
        "cost_p50_cny_t": np.percentile(cost, 50, axis=1),
        "cost_p95_cny_t": np.percentile(cost, 95, axis=1),
        "cost_ratio_p50": np.percentile(ratio, 50, axis=1),
        "cost_ratio_p95": np.percentile(ratio, 95, axis=1),
    })


def lca_robust_probability(db: IngredientDB, X: np.ndarray, x_bench: np.ndarray, rules: Rules, n: int, spread: float,
                           shared: bool, seed: int) -> np.ndarray:
    """P_robust: share of samples in which a formulation satisfies R1-R4 when every impact factor is
    multiplied by U(1 - spread, 1 + spread). With shared=True one factor per ingredient is applied
    to all six indicators. Cost is held at its deterministic value."""
    rng = np.random.default_rng(seed)
    k, m = db.impacts.shape
    E0 = db.impacts
    C_r = (X @ db.price) / (x_bench @ db.price)
    r1 = C_r <= 1.0 + rules.cost_tol + 1e-12
    hits = np.zeros(X.shape[0])
    for _ in range(n):
        f = rng.uniform(1 - spread, 1 + spread, size=(k, 1) if shared else (k, m))
        E = E0 * f
        R = (X @ E) / (x_bench @ E)
        ok = r1 & np.all(R <= 1.0 + rules.burden_tol + 1e-12, axis=1) & ((R < 1).sum(axis=1) >= rules.min_reduced)
        if rules.require_gwp:
            ok &= R[:, 0] < 1
        hits += ok
    return hits / n


def threshold_sensitivity(db: IngredientDB, X: np.ndarray, ids: list[str], x_bench: np.ndarray, base: Rules, top_ids: list[str]):
    """Robust-set size under one-at-a-time changes and over the full 5 x 4 x 4 grid of thresholds."""
    def run(r: Rules):
        ev = evaluate(db, X, x_bench, r)
        rob = ev[ev["robust"]].copy()
        rob["formula_id"] = np.array(ids)[ev["robust"].to_numpy()]
        best = rob.sort_values("cleaner_score")["formula_id"].iloc[0] if len(rob) else None
        kept = int(np.isin(top_ids, rob["formula_id"]).sum())
        return int(ev["robust"].sum()), best, kept

    one = []
    for v in [0.0, 0.05, 0.10, 0.15, 0.20]:
        one.append(("cost tolerance", v, *run(replace(base, cost_tol=v))))
    for v in [3, 4, 5, 6]:
        one.append(("min. reduced indicators", v, *run(replace(base, min_reduced=v))))
    for v in [0.0, 0.05, 0.10, 0.20]:
        one.append(("burden-shifting tolerance", v, *run(replace(base, burden_tol=v))))
    one.append(("GWP rule removed", 0, *run(replace(base, require_gwp=False))))
    cols = ["parameter", "value", "n_robust", "best_formula", "base_top20_retained"]
    grid = []
    for c, k, b in product([0.0, 0.05, 0.10, 0.15, 0.20], [3, 4, 5, 6], [0.0, 0.05, 0.10, 0.20]):
        grid.append((c, k, b, *run(replace(base, cost_tol=c, min_reduced=k, burden_tol=b))))
    return pd.DataFrame(one, columns=cols), pd.DataFrame(grid, columns=["cost_tol", "min_reduced", "burden_tol", "n_robust", "best_formula", "base_top20_retained"])
