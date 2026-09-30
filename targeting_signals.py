"""Synthetic transaction log and the targeting signals a retail media team would mine from it."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from simulator import SEED

CATEGORIES = ["Coffee", "Creamer", "Cereal", "Milk", "Chips", "Salsa", "Soda", "Beer", "Pasta", "Pasta sauce", "Frozen pizza", "Ice cream"]
MISSIONS = {
    "Breakfast": {"Coffee": .55, "Creamer": .40, "Cereal": .45, "Milk": .60},
    "Snack & party": {"Chips": .60, "Salsa": .42, "Soda": .55, "Beer": .30},
    "Dinner tonight": {"Pasta": .50, "Pasta sauce": .45, "Frozen pizza": .35, "Soda": .20, "Milk": .20},
    "Treat": {"Ice cream": .55, "Chips": .25, "Soda": .25},
    "Stock-up": {category: .22 for category in CATEGORIES},
}
MISSION_MIX = {  # probability of each shopping mission by daypart
    "Morning": [.50, .08, .05, .05, .32],
    "Midday": [.15, .30, .15, .10, .30],
    "Evening": [.05, .30, .35, .15, .15],
}
HOURS = np.arange(7, 23)
BRANDS = ["Brand A", "Competitor B", "Competitor C", "Private label"]
ADVERTISED_BRAND, ADVERTISED_CATEGORY = "Brand A", "Salsa"


def daypart_of(hour: np.ndarray) -> np.ndarray:
    return np.where(hour < 11, "Morning", np.where(hour < 16, "Midday", "Evening"))


def transaction_log(seed: int = SEED, baskets: int = 40_000, stores: int = 24) -> pd.DataFrame:
    """One row per basket: store, hour, mission, category flags and the salsa brand bought."""
    rng = np.random.default_rng(seed + 500)
    traffic = 1 + .8 * np.exp(-(HOURS - 8) ** 2 / 2) + 1.0 * np.exp(-(HOURS - 12.5) ** 2 / 2) + 1.6 * np.exp(-(HOURS - 18) ** 2 / 3)
    hour = rng.choice(HOURS, baskets, p=traffic / traffic.sum())
    store = rng.integers(0, stores, baskets)
    part = daypart_of(hour)
    mission = np.empty(baskets, dtype=object)
    for name, mix in MISSION_MIX.items():
        mask = part == name
        mission[mask] = rng.choice(list(MISSIONS), mask.sum(), p=mix)
    probability = np.full((baskets, len(CATEGORIES)), .04)
    for name, items in MISSIONS.items():
        mask = mission == name
        for category, value in items.items():
            probability[mask, CATEGORIES.index(category)] = value
    # Stores differ in how well the salsa category and Brand A sell.
    category_affinity = rng.lognormal(0, .28, stores)
    probability[:, CATEGORIES.index(ADVERTISED_CATEGORY)] = np.clip(probability[:, CATEGORIES.index(ADVERTISED_CATEGORY)] * category_affinity[store], 0, .95)
    items = rng.random(probability.shape) < probability
    brand_share = np.clip(rng.normal(.34, .09, stores), .12, .6)
    salsa = items[:, CATEGORIES.index(ADVERTISED_CATEGORY)]
    other_brands = rng.choice(BRANDS[1:], baskets, p=[.4, .25, .35])
    brand = np.where(rng.random(baskets) < brand_share[store], ADVERTISED_BRAND, other_brands)
    log = pd.DataFrame(items, columns=CATEGORIES)
    log.insert(0, "store_id", [f"S{s + 1:02d}" for s in store])
    log.insert(1, "hour", hour)
    log.insert(2, "daypart", part)
    log.insert(3, "mission", mission)
    log["salsa_brand"] = np.where(salsa, brand, "")
    return log


def affinity(log: pd.DataFrame) -> dict[str, Any]:
    items = log[CATEGORIES].to_numpy(dtype=float)
    n = len(items)
    support = items.mean(axis=0)
    together = items.T @ items / n
    lift = together / np.outer(support, support)
    np.fill_diagonal(lift, np.nan)
    rules = []
    for i, anchor in enumerate(CATEGORIES):
        for j, complement in enumerate(CATEGORIES):
            if i != j:
                rules.append({"Anchor (screen placed here)": anchor, "Complement (advertised)": complement,
                              "Support": together[i, j], "Attach rate": together[i, j] / support[i], "Lift": lift[i, j]})
    rules = pd.DataFrame(rules).query("Support >= 0.02").sort_values("Lift", ascending=False)
    rules = rules.drop_duplicates(subset=["Lift"]).head(6)
    return {"lift": pd.DataFrame(lift, index=CATEGORIES, columns=CATEGORIES), "rules": rules}


def hour_index(log: pd.DataFrame) -> pd.DataFrame:
    """Category sales index by hour: 100 = the category sells in line with store traffic."""
    baskets_by_hour = log.groupby("hour").size()
    traffic_share = baskets_by_hour / baskets_by_hour.sum()
    category_by_hour = log.groupby("hour")[CATEGORIES].sum()
    category_share = category_by_hour / category_by_hour.sum()
    return category_share.div(traffic_share, axis=0).mul(100).T


def store_development(log: pd.DataFrame) -> pd.DataFrame:
    """Category and brand development indices by store for the advertised category."""
    by_store = log.groupby("store_id").agg(
        baskets=("hour", "size"),
        category=(ADVERTISED_CATEGORY, "sum"),
        brand=("salsa_brand", lambda s: (s == ADVERTISED_BRAND).sum()),
    )
    chain_category = by_store.category.sum() / by_store.baskets.sum()
    chain_brand = by_store.brand.sum() / by_store.baskets.sum()
    by_store["CDI"] = by_store.category / by_store.baskets / chain_category * 100
    by_store["BDI"] = by_store.brand / by_store.baskets / chain_brand * 100
    return by_store.reset_index()


def switching(seed: int = SEED, shoppers: int = 3_000) -> pd.DataFrame:
    """Loyalty shoppers' brand in one quarter vs. the next: row-normalized switching shares."""
    rng = np.random.default_rng(seed + 600)
    start = rng.choice(BRANDS, shoppers, p=[.30, .32, .16, .22])
    transitions = {
        "Brand A": [.78, .10, .04, .08],
        "Competitor B": [.14, .70, .06, .10],
        "Competitor C": [.08, .09, .73, .10],
        "Private label": [.05, .07, .05, .83],
    }
    end = np.array([rng.choice(BRANDS, p=transitions[brand]) for brand in start])
    matrix = pd.crosstab(pd.Series(start, name="Last quarter"), pd.Series(end, name="This quarter"), normalize="index") * 100
    return matrix.loc[BRANDS, BRANDS]


def load_signals(seed: int = SEED) -> dict[str, Any]:
    log = transaction_log(seed)
    return {
        "log": log, "affinity": affinity(log), "hour_index": hour_index(log),
        "stores": store_development(log), "switching": switching(seed),
    }
