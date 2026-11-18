"""Linear formulation model (Eqs. 1-12): reference diets, LP vertices and convex combinations.

Inclusions x_i are in kg per tonne of complete feed. A fixed premix (default 20 kg/t) completes
the tonne; it is identical in all diets and is not counted in nutrients, energy, cost or impacts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .data import FISHMEAL, IngredientDB

TOTAL_KG = 1000.0


@dataclass
class NutrientSpec:
    """Limits on total nutrients, % of as-fed feed (Table 1)."""

    cp_min: float = 42.0
    cp_max: float | None = 54.0
    lipid_min: float = 7.0
    lipid_max: float = 12.0
    fibre_max: float = 5.0
    ash_max: float = 15.0
    starch_max: float = 15.0
    lys_min: float = 3.0
    lys_cp_min: float = 5.5          # lysine as % of crude protein
    p_min: float = 1.0
    p_max: float = 2.2
    dm_min: float = 89.0
    ge_abs_min: float = 12.5         # MJ/kg; candidates also need >= ge_rel_min * GE of the benchmark
    ge_rel_min: float = 0.97
    plant_max: float | None = 45.0   # plant meals, brans and flours together, % of feed; None or 100 switches the limit off
    premix_kg: float = 20.0
    premix_dm: float = 0.0           # dry-matter fraction of the premix

    @classmethod
    def from_dict(cls, d: dict) -> "NutrientSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class LinearModel:
    """Constraint set of Eqs. (2)-(5), (7) and (9) in scipy.optimize.linprog form."""

    db: IngredientDB
    spec: NutrientSpec
    lb: np.ndarray
    ub: np.ndarray
    A_ub: list = field(default_factory=list)
    b_ub: list = field(default_factory=list)
    A_eq: list = field(default_factory=list)
    b_eq: list = field(default_factory=list)

    def ge(self, row, value):
        self.A_ub.append(-np.asarray(row, float)); self.b_ub.append(-float(value))

    def le(self, row, value):
        self.A_ub.append(np.asarray(row, float)); self.b_ub.append(float(value))

    def eq(self, row, value):
        self.A_eq.append(np.asarray(row, float)); self.b_eq.append(float(value))

    def copy(self) -> "LinearModel":
        return LinearModel(self.db, self.spec, self.lb.copy(), self.ub.copy(), list(self.A_ub), list(self.b_ub), list(self.A_eq), list(self.b_eq))

    def solve(self, c: np.ndarray):
        res = linprog(np.asarray(c, float), A_ub=np.array(self.A_ub), b_ub=np.array(self.b_ub),
                      A_eq=np.array(self.A_eq), b_eq=np.array(self.b_eq),
                      bounds=list(zip(self.lb, self.ub)), method="highs")
        return res

    def is_feasible(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        x = np.asarray(x, float)
        ok = np.all(x >= self.lb - tol) and np.all(x <= self.ub + tol)
        if self.A_ub:
            ok &= bool(np.all(np.array(self.A_ub) @ x <= np.array(self.b_ub) + tol * 10))
        if self.A_eq:
            ok &= bool(np.allclose(np.array(self.A_eq) @ x, np.array(self.b_eq), atol=tol * 10))
        return bool(ok)


def base_model(db: IngredientDB, spec: NutrientSpec) -> LinearModel:
    """Mass balance, nutrient limits, lysine/CP ratio, total plant meals and inclusion limits (Eqs. 2-5)."""
    m = LinearModel(db, spec, db.col("lb").copy(), db.col("ub").copy())
    n = len(db)
    kg = TOTAL_KG / 100.0  # converts a % limit into kg of nutrient per tonne
    cp, lipid, fibre, ash = db.frac("cp"), db.frac("lipid"), db.frac("fibre"), db.frac("ash")
    starch, lys, p, dm = db.frac("starch"), db.frac("lys"), db.frac("p"), db.frac("dm")
    m.eq(np.ones(n), TOTAL_KG - spec.premix_kg)
    m.ge(cp, spec.cp_min * kg)
    if spec.cp_max is not None:
        m.le(cp, spec.cp_max * kg)
    m.ge(lipid, spec.lipid_min * kg)
    m.le(lipid, spec.lipid_max * kg)
    m.le(fibre, spec.fibre_max * kg)
    m.le(ash, spec.ash_max * kg)
    m.le(starch, spec.starch_max * kg)
    m.ge(lys, spec.lys_min * kg)
    m.ge(lys - spec.lys_cp_min / 100.0 * cp, 0.0)          # Eq. (4) in linear form
    m.ge(p, spec.p_min * kg)
    m.le(p, spec.p_max * kg)
    m.ge(dm, spec.dm_min * kg - spec.premix_dm * spec.premix_kg)
    if spec.plant_max is not None and spec.plant_max < 100.0:
        m.le(db.plant_mask, spec.plant_max * kg)              # total plant meals
    return m


def reference_diets(db: IngredientDB, spec: NutrientSpec, levels=(350.0, 450.0, 550.0), labels=("FM35", "FM45", "FM55")) -> pd.DataFrame:
    """Least-cost diets with fishmeal fixed at each level (Eqs. 1-6)."""
    i_fm = db.idx(FISHMEAL)
    rows = {}
    for lab, level in zip(labels, levels):
        m = base_model(db, spec)
        m.lb[i_fm] = m.ub[i_fm] = level
        res = m.solve(db.price)
        if res.status != 0:
            raise RuntimeError(f"reference diet {lab} is infeasible: {res.message}")
        rows[lab] = res.x
    return pd.DataFrame(rows, index=db.names).T


def candidate_model(db: IngredientDB, spec: NutrientSpec, x_bench: np.ndarray, fm_min: float = 300.0, fm_max: float = 450.0) -> LinearModel:
    """Constraint set for candidates: Eqs. (2)-(5), fishmeal window (7) and energy floor (9)."""
    m = base_model(db, spec)
    i_fm = db.idx(FISHMEAL)
    m.lb[i_fm] = max(m.lb[i_fm], fm_min)
    m.ub[i_fm] = min(m.ub[i_fm], fm_max)
    ge_bench = db.ge_coef @ x_bench / TOTAL_KG
    ge_min = max(spec.ge_abs_min, spec.ge_rel_min * ge_bench)
    m.ge(db.ge_coef, ge_min * TOTAL_KG)
    return m


def normalised_coefficients(db: IngredientDB, x_bench: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-kg coefficients whose sums over a diet give C_r and R_k relative to the benchmark.

    Returns (cost, cost at high prices, impacts with shape (6, n)).
    """
    p = db.price / (db.price @ x_bench)
    p_high = db.price_high / (db.price_high @ x_bench)
    e = (db.impacts / (x_bench @ db.impacts)).T
    return p, p_high, e


@dataclass
class VertexSettings:
    n_random: int = 2000
    eps_rel: float = 1e-4            # tie-breaking perturbation, relative to the mean coefficient
    dedup_decimals: int = 4          # vertices equal to 1e-4 kg/t are treated as one
    seed: int = 20261006


def lp_vertices(model: LinearModel, x_bench: np.ndarray, s: VertexSettings) -> pd.DataFrame:
    """Solve the LP for the named and randomly weighted objectives (Eqs. 10-11); keep distinct optima.

    Named objectives: minimum cost at base and at high prices, minimum of each of the six indicators,
    and the equal cost-environment compromise (Eq. 20). For a random objective the cost weight w_0 is
    drawn from U(0, 1) and the remaining weight 1 - w_0 is split among the six indicators by a flat
    Dirichlet draw, so that the cost-environment balance is sampled uniformly.
    """
    db = model.db
    rng = np.random.default_rng(s.seed)
    p, p_high, e = normalised_coefficients(db, x_bench)
    objectives = [("min_cost", p), ("min_cost_high_price", p_high)]
    objectives += [(f"min_{k}", e[j]) for j, k in enumerate(["gwp", "feu", "lu", "wu", "meu", "ta"])]
    objectives.append(("compromise", 0.5 * p + 0.5 * e.mean(axis=0)))
    for _ in range(s.n_random):
        w0 = rng.random()
        wk = (1.0 - w0) * rng.dirichlet(np.ones(e.shape[0]))
        objectives.append(("random", w0 * p + wk @ e))

    seen, rows, origin = {}, [], []
    for name, c in objectives:
        eps = s.eps_rel * np.abs(c).mean() * rng.random(c.size)
        res = model.solve(c + eps)
        if res.status != 0:
            continue
        x = np.clip(res.x, model.lb, model.ub)
        key = tuple(np.round(x, s.dedup_decimals))
        if key in seen:
            continue
        seen[key] = len(rows)
        rows.append(x)
        origin.append(name)
    out = pd.DataFrame(rows, columns=db.names)
    out.insert(0, "origin", origin)
    return out


@dataclass
class MixSettings:
    n_combinations: int = 1000
    k_min: int = 2
    k_max: int = 5
    seed: int = 20261007


def convex_combinations(vertices: np.ndarray, s: MixSettings) -> tuple[np.ndarray, list[str]]:
    """Random convex combinations of the vertices (Eq. 12).

    Each draw selects k vertices without replacement, k uniform on {k_min, ..., k_max},
    and weights them with lambda ~ Dirichlet(1, ..., 1).
    """
    rng = np.random.default_rng(s.seed)
    v = vertices.shape[0]
    out, desc = [], []
    for _ in range(s.n_combinations):
        k = int(rng.integers(s.k_min, min(s.k_max, v) + 1))
        idx = rng.choice(v, size=k, replace=False)
        lam = rng.dirichlet(np.ones(k))
        out.append(lam @ vertices[idx])
        desc.append(";".join(f"{i}:{l:.4f}" for i, l in zip(idx, lam)))
    return np.array(out), desc


def generate_candidates(model: LinearModel, x_bench: np.ndarray, vs: VertexSettings, ms: MixSettings) -> pd.DataFrame:
    """Candidate set = distinct LP vertices followed by convex combinations, all checked for feasibility."""
    vt = lp_vertices(model, x_bench, vs)
    V = vt[model.db.names].to_numpy()
    mix, desc = convex_combinations(V, ms)
    X = np.vstack([V, mix])
    infeasible = [i for i, x in enumerate(X) if not model.is_feasible(x)]
    if infeasible:
        raise RuntimeError(f"{len(infeasible)} candidates violate the constraints")
    df = pd.DataFrame(X, columns=model.db.names)
    df.insert(0, "source", ["lp_vertex"] * len(V) + ["convex_combination"] * len(mix))
    df.insert(1, "origin", list(vt["origin"]) + desc)
    df.insert(0, "formula_id", [f"F{i}" for i in range(len(df))])
    return df
