"""Ingredient database: loading, unit conversion and validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Column names used internally, with the spellings accepted in input files.
COLUMN_ALIASES = {
    "name": ["name", "ingredient"],
    "price": ["price", "price_cny_t"],
    "price_low": ["price low", "price_low", "price_low_cny_t"],
    "price_high": ["price high", "price_high", "price_high_cny_t"],
    "lb": ["lb", "lb_kg_t"],
    "ub": ["ub", "ub_kg_t"],
    "dm": ["dm", "dm_pct"],
    "cp": ["cp", "cp_pct"],
    "fibre": ["fib", "fibre", "fibre_pct"],
    "lipid": ["ee", "lipid", "lipid_pct"],
    "ash": ["ash", "ash_pct"],
    "starch": ["st", "starch", "starch_pct"],
    "lys": ["lys", "lys_pct"],
    "p": ["p", "p_pct"],
    "avp": ["avp", "avp_pct"],
    "adc_lipid": ["adc-l", "adc_l", "adc_lipid"],
    "adc_protein": ["adc-n", "adc_n", "adc_protein"],
    "de": ["de", "de_mj_kg"],
    "gwp": ["global warming", "gwp"],
    "feu": ["freshwater eutrophication", "feu"],
    "meu": ["marine eutrophication", "meu"],
    "ta": ["terrestrial acidification", "ta"],
    "lu": ["land use", "lu"],
    "wu": ["water consumption", "water use", "wu"],
    "origin": ["origin", "category"],
}

# Six ReCiPe 2016 midpoint (H) indicators, per kg of as-fed ingredient.
IMPACTS = ["gwp", "feu", "lu", "wu", "meu", "ta"]
IMPACT_LABELS = {"gwp": "GWP", "feu": "FEU", "lu": "LU", "wu": "WU", "meu": "MEU", "ta": "TA"}
IMPACT_UNITS = {"gwp": "kg CO2-eq", "feu": "kg P-eq", "lu": "m2a crop-eq", "wu": "m3", "meu": "kg N-eq", "ta": "kg SO2-eq"}
NUTRIENTS = ["dm", "cp", "fibre", "lipid", "ash", "starch", "lys", "p"]
REQUIRED = ["name", "price", "price_low", "price_high", "lb", "ub", *NUTRIENTS, "adc_lipid", "adc_protein", *IMPACTS]

FISHMEAL = "Fish meal-65% CP"
ORIGINS = {"animal", "plant", "oil", "other"}   # "plant" = plant meals, brans and flours; oils are counted separately


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


@dataclass
class IngredientDB:
    """Ingredient data as arrays. Nutrients are fractions of as-fed mass; prices are CNY per kg."""

    table: pd.DataFrame
    names: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.names = list(self.table["name"])

    def __len__(self) -> int:
        return len(self.names)

    def idx(self, name: str) -> int:
        return self.names.index(name)

    def col(self, key: str) -> np.ndarray:
        return self.table[key].to_numpy(dtype=float)

    # Nutrient contents as fractions of as-fed mass
    def frac(self, key: str) -> np.ndarray:
        return self.col(key) / 100.0

    @property
    def price(self) -> np.ndarray:  # CNY per kg
        return self.col("price") / 1000.0

    @property
    def price_low(self) -> np.ndarray:
        return self.col("price_low") / 1000.0

    @property
    def price_high(self) -> np.ndarray:
        return self.col("price_high") / 1000.0

    @property
    def plant_mask(self) -> np.ndarray:
        """1 for plant meals, brans and flours, 0 otherwise (oils are not counted)."""
        if "origin" not in self.table.columns:
            raise ValueError("the ingredient file has no 'origin' column, which the plant-meal limit needs")
        return (self.table["origin"] == "plant").to_numpy(dtype=float)

    @property
    def impacts(self) -> np.ndarray:  # shape (n_ingredients, 6), per kg ingredient
        return self.table[IMPACTS].to_numpy(dtype=float)

    @property
    def ge_coef(self) -> np.ndarray:
        """Gross energy of each ingredient, MJ per kg (Eq. 8)."""
        return 23.6 * self.frac("cp") + 39.5 * self.frac("lipid") + 17.2 * self.frac("starch")

    @property
    def ea_coef(self) -> np.ndarray:
        """ADC-adjusted assimilable energy of each ingredient, MJ per kg (Eq. 23)."""
        return 23.6 * self.frac("cp") * self.col("adc_protein") + 39.5 * self.frac("lipid") * self.col("adc_lipid")

    def with_prices(self, price_cny_t: np.ndarray) -> "IngredientDB":
        t = self.table.copy()
        t["price"] = price_cny_t
        return IngredientDB(t)


def load_ingredients(path: str | Path, overrides: dict | None = None) -> IngredientDB:
    """Read the ingredient database (xlsx or csv) and check it.

    ``overrides`` maps ingredient name -> {column: value} and is applied after reading,
    so that corrections are documented in the configuration rather than hidden in the data file.
    """
    path = Path(path)
    raw = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path, float_precision="round_trip")
    lookup = {_norm(c): c for c in raw.columns}
    rename = {}
    for key, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if _norm(a) in lookup:
                rename[lookup[_norm(a)]] = key
                break
    df = raw.rename(columns=rename)
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        raise ValueError(f"ingredient file lacks columns: {missing_cols}")
    df = df[[c for c in COLUMN_ALIASES if c in df.columns]].copy()
    df["name"] = df["name"].astype(str).str.strip()
    if "origin" in df.columns:
        df["origin"] = df["origin"].astype(str).str.strip().str.lower()
    for c in df.columns:
        if c not in ("name", "origin"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for name, values in (overrides or {}).items():
        if name not in set(df["name"]):
            raise KeyError(f"override for unknown ingredient '{name}'")
        for c, v in values.items():
            df.loc[df["name"] == name, c] = v

    validate(df)
    return IngredientDB(df.reset_index(drop=True))


def validate(df: pd.DataFrame) -> None:
    """Raise an error on missing or impossible values."""
    problems = []
    for c in REQUIRED[1:]:
        bad = df.loc[df[c].isna(), "name"].tolist()
        if bad:
            problems.append(f"missing {c}: {bad}")
    for c in ["adc_lipid", "adc_protein"]:
        bad = df.loc[(df[c] < 0) | (df[c] > 1), "name"].tolist()
        if bad:
            problems.append(f"{c} outside [0, 1]: {bad}")
    bad = df.loc[df["lb"] > df["ub"], "name"].tolist()
    if bad:
        problems.append(f"lb > ub: {bad}")
    bad = df.loc[df["dm"] > 100, "name"].tolist()
    if bad:
        problems.append(f"dry matter above 100 %: {bad}")
    bad = df.loc[(df["price"] < df["price_low"]) | (df["price"] > df["price_high"]), "name"].tolist()
    if bad:
        problems.append(f"base price outside its low-high range: {bad}")
    if "origin" in df.columns:
        bad = df.loc[~df["origin"].isin(ORIGINS), "name"].tolist()
        if bad:
            problems.append(f"origin not one of {sorted(ORIGINS)}: {bad}")
    if FISHMEAL not in set(df["name"]):
        problems.append(f"fishmeal entry '{FISHMEAL}' not found")
    if problems:
        raise ValueError("ingredient database failed validation:\n  " + "\n  ".join(problems))


def plausibility_report(db: IngredientDB) -> pd.DataFrame:
    """Ratios for manual checking; they do not stop a run."""
    t = db.table
    out = pd.DataFrame({
        "name": t["name"],
        "lys_per_cp_pct": np.where(t["cp"] > 0, 100 * t["lys"] / t["cp"].replace(0, np.nan), np.nan),
        "sum_proximate_pct_of_dm": 100 * (t["cp"] + t["lipid"] + t["ash"] + t["fibre"] + t["starch"]) / t["dm"],
    })
    return out
