"""Benchmark-normalised evaluation, burden-shifting penalty, cleaner score and eligibility rules (Eqs. 13-20)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import FISHMEAL, IMPACTS, IngredientDB
from .formulation import TOTAL_KG


@dataclass
class Rules:
    cost_tol: float = 0.10        # R1: C_r <= 1 + cost_tol
    require_gwp: bool = True      # R2: R_GWP < 1
    min_reduced: int = 4          # R3: at least this many R_k < 1
    burden_tol: float = 0.10      # R4: every R_k <= 1 + burden_tol (B_f = 0)
    w_cost: float = 0.5
    w_env: float = 0.5
    w_burden: float = 2.0

    @classmethod
    def from_dict(cls, d: dict) -> "Rules":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


CLASS_LABELS = {
    "robust": "Robust cleaner candidate",
    "low_gwp_burden": "Lower GWP with burden shifting",
    "partial": "Partial improvement without GWP reduction",
    "limited": "Within cost tolerance, limited environmental gain",
    "cost": "Exceeds cost tolerance",
}


def composition(db: IngredientDB, X: np.ndarray, premix_kg: float = 20.0, premix_dm: float = 0.0) -> pd.DataFrame:
    """Nutrient and energy contents of each diet (as-fed; the premix is not counted)."""
    X = np.atleast_2d(X)
    f = lambda k: X @ db.frac(k) / TOTAL_KG * 100.0
    return pd.DataFrame({
        "cp_pct": f("cp"), "lipid_pct": f("lipid"), "starch_pct": f("starch"), "fibre_pct": f("fibre"),
        "ash_pct": f("ash"), "lys_pct": f("lys"), "p_pct": f("p"),
        "dm_pct": (X @ db.frac("dm") + premix_dm * premix_kg) / TOTAL_KG * 100.0,
        "ge_mj_kg": X @ db.ge_coef / TOTAL_KG,
        "ea_mj_kg": X @ db.ea_coef / TOTAL_KG,
    })


def evaluate(db: IngredientDB, X: np.ndarray, x_bench: np.ndarray, rules: Rules, price: np.ndarray | None = None,
             impacts: np.ndarray | None = None) -> pd.DataFrame:
    """Cost, impacts, ratios, penalty, score and classification for every row of X (kg/t)."""
    X = np.atleast_2d(np.asarray(X, float))
    price = db.price if price is None else price
    E = db.impacts if impacts is None else impacts
    cost = X @ price                                  # CNY per tonne (premix excluded, identical in all diets)
    cost_b = float(x_bench @ price)
    I = X @ E / TOTAL_KG                              # Eq. (14), per kg feed
    I_b = x_bench @ E / TOTAL_KG
    R = I / I_b                                       # Eq. (16)
    C_r = cost / cost_b                               # Eq. (15)
    E_m = R.mean(axis=1)                              # Eq. (17)
    B = np.maximum(0.0, R - (1.0 + rules.burden_tol)).sum(axis=1)  # Eq. (18)
    S = rules.w_cost * C_r + rules.w_env * E_m + rules.w_burden * B  # Eq. (19)
    n_red = (R < 1.0).sum(axis=1)
    r1 = C_r <= 1.0 + rules.cost_tol + 1e-12
    r2 = (R[:, 0] < 1.0) if rules.require_gwp else np.ones(len(X), bool)
    r3 = n_red >= rules.min_reduced
    r4 = np.all(R <= 1.0 + rules.burden_tol + 1e-12, axis=1)
    robust = r1 & r2 & r3 & r4
    cls = np.where(~r1, "cost", np.where(robust, "robust", np.where((R[:, 0] < 1) & ~r4, "low_gwp_burden",
                   np.where((R[:, 0] >= 1) & (E_m < 1), "partial", "limited"))))
    out = pd.DataFrame({"cost_cny_t": cost, "cost_ratio": C_r, "cost_change_pct": 100 * (C_r - 1)})
    for j, k in enumerate(IMPACTS):
        out[k] = I[:, j]
        out[f"{k}_ratio"] = R[:, j]
        out[f"{k}_change_pct"] = 100 * (R[:, j] - 1)
    out["mean_env_ratio"] = E_m
    out["mean_env_change_pct"] = 100 * (E_m - 1)
    out["burden_penalty"] = B
    out["cleaner_score"] = S
    out["n_reduced"] = n_red
    out["R1"], out["R2"], out["R3"], out["R4"] = r1, r2, r3, r4
    out["robust"] = robust
    out["class"] = cls
    return out


def evaluate_full(db: IngredientDB, cand: pd.DataFrame, x_bench: np.ndarray, rules: Rules, premix_kg: float = 50.0,
                  premix_dm: float = 1.0) -> pd.DataFrame:
    X = cand[db.names].to_numpy(float)
    ev = evaluate(db, X, x_bench, rules)
    comp = composition(db, X, premix_kg, premix_dm)
    out = pd.concat([cand[["formula_id", "source", "origin"]].reset_index(drop=True), ev, comp], axis=1)
    out["fishmeal_kg_t"] = cand[FISHMEAL].to_numpy()
    return out


def rank_robust(ev: pd.DataFrame) -> pd.DataFrame:
    """Robust candidates ordered by S (Eq. 20); ties broken by formula number."""
    r = ev[ev["robust"]].copy()
    r["_n"] = r["formula_id"].str[1:].astype(int)
    return r.sort_values(["cleaner_score", "_n"]).drop(columns="_n")
