"""Consistency checks; run with `python -m pytest -q`."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aquafeed_screen import benchmarks as bm  # noqa: E402
from aquafeed_screen.data import load_ingredients  # noqa: E402
from aquafeed_screen.evaluation import Rules, evaluate  # noqa: E402
from aquafeed_screen.formulation import (MixSettings, NutrientSpec, VertexSettings, candidate_model,  # noqa: E402
                                         generate_candidates, reference_diets)
from aquafeed_screen.growth import GrowthSettings, simulate  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    db = load_ingredients(ROOT / "data" / "example" / "ingredient_database_example.csv")
    spec = NutrientSpec()
    ref = reference_diets(db, spec)
    xb = ref.loc["FM45"].to_numpy(float)
    m = candidate_model(db, spec, xb)
    cand = generate_candidates(m, xb, VertexSettings(n_random=60), MixSettings(n_combinations=60))
    return db, spec, xb, m, cand


def test_benchmark_ratios_are_one(setup):
    db, _, xb, _, _ = setup
    ev = evaluate(db, xb[None, :], xb, Rules()).iloc[0]
    assert ev.cost_ratio == pytest.approx(1.0)
    assert ev.mean_env_ratio == pytest.approx(1.0)
    assert ev.burden_penalty == 0


def test_candidates_feasible_and_mass_balanced(setup):
    db, spec, _, m, cand = setup
    X = cand[db.names].to_numpy(float)
    assert np.allclose(X.sum(axis=1), 1000 - spec.premix_kg)
    assert all(m.is_feasible(x) for x in X)


def test_plant_meal_limit(setup):
    db, spec, _, _, cand = setup
    ref = reference_diets(db, spec)
    for X in (ref[db.names].to_numpy(float), cand[db.names].to_numpy(float)):
        assert np.all(X @ db.plant_mask <= spec.plant_max * 10 + 1e-6)
    # without the limit the least-cost FM35 diet exceeds it, so the constraint is active there
    free = reference_diets(db, NutrientSpec(plant_max=None))
    assert free.loc["FM35", db.names].to_numpy(float) @ db.plant_mask > spec.plant_max * 10


def test_direct_lp_not_worse_than_enumeration(setup):
    db, _, xb, m, cand = setup
    rules = Rules()
    ev = evaluate(db, cand[db.names].to_numpy(float), xb, rules)
    x = bm.direct_lp(m, xb, rules)
    s_lp = evaluate(db, x[None, :], xb, rules).iloc[0].cleaner_score
    assert s_lp <= ev.loc[ev.robust, "cleaner_score"].min() + 1e-9


def test_iso_energetic_constraint(setup):
    db, _, xb, m, _ = setup
    x = bm.direct_lp(m, xb, Rules(), energy="equal")
    assert x @ db.ea_coef == pytest.approx(xb @ db.ea_coef, rel=1e-8)


def test_growth_monotonic_in_assimilable_energy():
    s = GrowthSettings()
    d = [simulate(e, s)[0]["days"] for e in (12.0, 12.5, 13.0, 13.5, 14.0)]
    assert all(b < a for a, b in zip(d, d[1:]))


def test_adc_literature_matches_database(setup):
    from aquafeed_screen.adc_check import compare, load_literature
    db = setup[0]
    lit = load_literature(ROOT / "data" / "example" / "adc_largemouth_bass.csv")
    cmp = compare(db, lit)
    assert set(cmp.ingredient) <= set(db.names)
    assert ((cmp.lmb_N_min <= cmp.lmb_N_mean) & (cmp.lmb_N_mean <= cmp.lmb_N_max)).all()
    v = lit[["adc_cp_pct", "adc_lipid_pct", "adc_lys_pct", "adc_total_aa_pct"]].to_numpy(float)
    assert np.nanmin(v) > 0 and np.nanmax(v) <= 100


def test_attribution_is_exact(setup):
    from aquafeed_screen.attribution import check_exact, contributions
    db, _, xb, _, cand = setup
    X = cand[db.names].to_numpy(float)
    assert check_exact(db, X, xb, Rules()) < 1e-9
    # the benchmark has zero contribution from every ingredient
    assert np.abs(contributions(db, xb[None, :], xb)).max() == 0
