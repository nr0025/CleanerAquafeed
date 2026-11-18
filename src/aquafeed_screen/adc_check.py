"""Comparison of the database digestibility coefficients with values measured in largemouth bass.

The database ADCs are salmonid values of the INRAE-CIRAD-AFZ feed tables: nitrogen digestibility for ADC-N and,
because the tables give no lipid digestibility for fish, energy digestibility for ADC-L. They are applied to
largemouth bass although most were not measured in that species. This
module compares them with published largemouth bass values (data/adc_largemouth_bass.csv) and repeats the
growth simulation with the largemouth bass values substituted, so that the effect of the assumption on the
results can be read directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import IngredientDB
from .growth import GrowthSettings, simulate_diets


def load_literature(path) -> pd.DataFrame:
    lit = pd.read_csv(path)
    num = ["fish_weight_g", "adc_cp_pct", "adc_lipid_pct", "adc_lys_pct", "adc_total_aa_pct"]
    lit[num] = lit[num].apply(pd.to_numeric, errors="coerce")
    return lit


def _position(v: float, ref: pd.Series) -> str:
    if v < ref.min() - 1e-9:
        return "below"
    if v > ref.max() + 1e-9:
        return "above"
    return "within"


def compare(db: IngredientDB, lit: pd.DataFrame) -> pd.DataFrame:
    """One row per database ingredient with largemouth bass data.

    Protein: crude-protein ADCs where reported; where a study reports only amino-acid digestibility
    (Masagounder et al. 2009), its total amino-acid ADC is used and the basis is stated.
    """
    t = db.table.set_index("name")
    rows = []
    for name, g in lit.groupby("db_ingredient", sort=False):
        if name not in t.index:
            raise KeyError(f"{name} is not in the ingredient database")
        cp = g.adc_cp_pct.dropna()
        basis = "crude protein"
        if cp.empty:
            cp, basis = g.adc_total_aa_pct.dropna(), "total amino acids"
        lip = g.adc_lipid_pct.dropna()
        lys = g.adc_lys_pct.dropna()
        dbn, dbl = 100 * t.loc[name, "adc_protein"], 100 * t.loc[name, "adc_lipid"]
        rows.append({
            "ingredient": name,
            "n_values_N": len(cp), "basis_N": basis,
            "db_adc_N_pct": dbn, "lmb_N_min": cp.min(), "lmb_N_max": cp.max(), "lmb_N_mean": cp.mean(),
            "db_minus_lmb_mean_N": dbn - cp.mean(), "db_N_vs_lmb_range": _position(dbn, cp),
            "n_values_L": len(lip), "db_adc_L_pct": dbl,
            "lmb_L_min": lip.min() if len(lip) else np.nan, "lmb_L_max": lip.max() if len(lip) else np.nan,
            "lmb_L_mean": lip.mean() if len(lip) else np.nan,
            "db_minus_lmb_mean_L": dbl - lip.mean() if len(lip) else np.nan,
            "lmb_lys_min": lys.min() if len(lys) else np.nan, "lmb_lys_max": lys.max() if len(lys) else np.nan,
            "sources": "; ".join(dict.fromkeys(g.source)),
        })
    return pd.DataFrame(rows)


def _with_adc(db: IngredientDB, n: dict | None = None, lipid: dict | None = None) -> IngredientDB:
    t = db.table.copy().set_index("name")
    for k, v in (n or {}).items():
        t.loc[k, "adc_protein"] = v / 100
    for k, v in (lipid or {}).items():
        t.loc[k, "adc_lipid"] = v / 100
    return IngredientDB(t.reset_index())


def growth_scenarios(db: IngredientDB, cmp: pd.DataFrame, X: np.ndarray, ids: list[str], benchmark: str,
                     gs: GrowthSettings) -> pd.DataFrame:
    """Days to the target weight of each diet with the database ADCs and with the largemouth bass means substituted."""
    n = dict(zip(cmp.ingredient, cmp.lmb_N_mean))
    lip = {k: v for k, v in zip(cmp.ingredient, cmp.lmb_L_mean) if np.isfinite(v)}
    cases = {"database": db, "lmb_N": _with_adc(db, n=n), "lmb_L": _with_adc(db, lipid=lip),
             "lmb_N_and_L": _with_adc(db, n=n, lipid=lip)}
    out = pd.DataFrame({"formula_id": ids})
    unit = X @ db.price / 1000
    for key, d in cases.items():
        ea = X @ d.ea_coef / 1000
        s, _ = simulate_diets(ids, ea, unit, gs)
        out[f"ea_{key}"] = s.ea_mj_kg.to_numpy()
        out[f"days_{key}"] = s.days.to_numpy()
        out[f"FCR_{key}"] = s.FCR.to_numpy()
        b = float(s.loc[s.formula_id == benchmark, "days"].iloc[0])
        out[f"days_pct_vs_benchmark_{key}"] = 100 * (s.days.to_numpy() / b - 1)
    return out
