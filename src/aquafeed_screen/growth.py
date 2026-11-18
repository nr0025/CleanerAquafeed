"""Bioenergetic growth check (Eqs. 23-25), parameterised for juvenile largemouth bass.

Diet composition enters the model only through the ADC-adjusted assimilable energy E_A (MJ/kg feed).
Feed intake, respiration and fish energy density depend on body weight and temperature alone, so for
a given temperature regime predicted growth increases monotonically with E_A.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass
class GrowthSettings:
    days: float = 200.0              # maximum horizon
    w0_g: float = 10.0
    w_target_g: float | None = 50.0   # stop when this weight is reached
    temperature_c: float = 28.0
    intake_cap_bw: float = 0.03       # as-fed feed per g wet body weight per day
    cmax_a: float = 0.065             # C_max = a * W^b * tau(T)
    cmax_b: float = 0.75
    k_sda: float = 0.20
    u: float = 0.15
    resp_a: float = 0.3393            # respiration: 13.56 * a * W^b * exp(m T) * exp(g v) * W * 0.024 (kJ/d)
    resp_b: float = -0.3486
    resp_m: float = 0.028
    resp_g: float = 0.0196
    swim_speed_cm_s: float = 5.0
    ef_a: float = 5.01                # fish energy density E_f = a * W^b (kJ/g)
    ef_b: float = 0.046
    t_max_feed: float = 37.0
    t_opt_feed: float = 27.5
    q10_feed: float = 2.65
    n_fish: int = 2000                # scales feed cost only
    n_eval: int = 181

    @classmethod
    def from_dict(cls, d: dict) -> "GrowthSettings":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def feed_tau(s: GrowthSettings) -> float:
    """Temperature multiplier of maximum intake (Kitchell-type function)."""
    lnq = math.log(s.q10_feed)
    w = lnq * (s.t_max_feed - s.t_opt_feed)
    y = lnq * (s.t_max_feed - s.t_opt_feed + 2.0)
    x = w ** 2 * (1 + math.sqrt(1 + 40.0 / y)) ** 2 / 400.0
    v = (s.t_max_feed - s.temperature_c) / (s.t_max_feed - s.t_opt_feed)
    return 0.0 if v <= 0 else v ** x * math.exp(x * (1.0 - v))


def rates(W: float, ea: float, s: GrowthSettings, tau: float) -> tuple[float, float, float, float]:
    W = max(W, 1e-8)
    C = min(s.cmax_a * W ** s.cmax_b * tau, s.intake_cap_bw * W)                     # Eq. (24), g/d
    A = C * ea                                                                        # kJ/d
    RA = 13.56 * s.resp_a * W ** s.resp_b * math.exp(s.resp_m * s.temperature_c) * math.exp(s.resp_g * s.swim_speed_cm_s) * W * 0.024
    Ef = s.ef_a * W ** s.ef_b
    dW = ((1.0 - s.k_sda - s.u) * A - RA) / Ef                                         # Eq. (25)
    return dW, C, A, RA


def cap_binding_weight(s: GrowthSettings) -> float:
    """Body weight below which the intake cap is the binding term of Eq. (24)."""
    tau = feed_tau(s)
    return (s.intake_cap_bw / (s.cmax_a * tau)) ** (1.0 / (s.cmax_b - 1.0))


def simulate(ea_mj_kg: float, s: GrowthSettings) -> tuple[dict, pd.DataFrame]:
    if not np.isfinite(ea_mj_kg) or ea_mj_kg <= 0:
        raise ValueError("assimilable energy must be positive")
    tau = feed_tau(s)

    def rhs(t, y):
        dW, C, A, RA = rates(y[0], ea_mj_kg, s, tau)
        return [dW, C, A, RA]

    events = None
    if s.w_target_g:
        def reach(t, y):
            return y[0] - s.w_target_g
        reach.terminal, reach.direction = True, 1
        events = reach
    sol = solve_ivp(rhs, (0.0, s.days), [s.w0_g, 0.0, 0.0, 0.0], method="RK45", rtol=1e-8, atol=1e-10,
                    events=events, dense_output=True)
    if not sol.success:
        raise RuntimeError(sol.message)
    if s.w_target_g and sol.status != 1:
        raise RuntimeError(f"target weight {s.w_target_g} g not reached within {s.days} days")
    t_end = float(sol.t[-1])
    t_eval = np.linspace(0.0, t_end, s.n_eval)
    Y = sol.sol(t_eval)
    W_end, feed = float(Y[0, -1]), float(Y[1, -1])
    gain = W_end - s.w0_g
    summary = {
        "ea_mj_kg": ea_mj_kg, "days": t_end, "W_end_g": W_end, "gain_g": gain, "feed_g_per_fish": feed,
        "FCR": feed / gain if gain > 0 else np.nan,
        "SGR_pct_d": 100 * (math.log(W_end) - math.log(s.w0_g)) / t_end,
        "assimilated_kj": float(Y[2, -1]), "respiration_kj": float(Y[3, -1]),
    }
    ts = pd.DataFrame({"day": t_eval, "W_g": Y[0], "cum_feed_g": Y[1]})
    return summary, ts


def simulate_diets(ids: list[str], ea: np.ndarray, unit_cost_cny_kg: np.ndarray, s: GrowthSettings):
    """Simulate every diet; returns (summary table, long time series)."""
    rows, series = [], []
    for fid, e, c in zip(ids, ea, unit_cost_cny_kg):
        summ, ts = simulate(float(e), s)
        summ.update(formula_id=fid, feed_unit_cost_cny_kg=float(c),
                    feed_cost_population_cny=float(c) * summ["feed_g_per_fish"] / 1000.0 * s.n_fish,
                    feed_cost_per_kg_gain_cny=float(c) * summ["FCR"])
        rows.append(summ)
        ts.insert(0, "formula_id", fid)
        series.append(ts)
    cols = ["formula_id", "ea_mj_kg", "days", "W_end_g", "gain_g", "feed_g_per_fish", "FCR", "SGR_pct_d", "feed_unit_cost_cny_kg",
            "feed_cost_population_cny", "feed_cost_per_kg_gain_cny", "assimilated_kj", "respiration_kj"]
    return pd.DataFrame(rows)[cols], pd.concat(series, ignore_index=True)
