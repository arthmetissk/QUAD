"""Campaign design matrix: dimension options, a measurement recommendation engine,
and a simulator that generates data for any combination and runs the recommended method."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t

from simulator import SEED, _coefficient_result, _interval

PLACEMENTS = {
    "endcap": "Endcap / high-traffic display",
    "checkout": "Checkout / point-of-sale screen",
    "shelf": "Shelf-edge / in-aisle",
    "entrance": "Entrance / lobby screen",
}
TARGETING = {
    "blanket": "None / blanket",
    "category": "Category-adjacent",
    "crosssell": "Cross-sell / complementary product",
    "conquest": "Competitive conquesting",
    "daypart": "Daypart / time-of-day",
}
OFFERS = {
    "awareness": "Pure awareness",
    "coupon": "Digital coupon tie-in",
    "bundle": "Bundle / multi-buy",
    "trial": "New product trial messaging",
}
ROLLOUTS = {
    "onoff": "Simple on/off",
    "staggered": "Staggered rollout",
    "dose": "Dose variation",
}
DIMENSIONS = {"placement": PLACEMENTS, "targeting": TARGETING, "offer": OFFERS, "rollout": ROLLOUTS}
DIMENSION_TITLES = {"placement": "Placement", "targeting": "Targeting logic", "offer": "Offer mechanic", "rollout": "Rollout structure"}

_PHRASES = {
    "placement": {"endcap": "Endcap screens", "checkout": "Checkout screens", "shelf": "Shelf-edge screens", "entrance": "Entrance screens"},
    "targeting": {"blanket": "blanket content in every store", "category": "category-adjacent content", "crosssell": "cross-sell targeting", "conquest": "competitive conquesting", "daypart": "evening daypart targeting"},
    "offer": {"awareness": "pure awareness messaging", "coupon": "a digital coupon tie-in", "bundle": "a bundle / multi-buy offer", "trial": "new-product trial messaging"},
    "rollout": {"onoff": "a single on/off launch", "staggered": "a staggered three-wave rollout", "dose": "play frequency varied by store"},
}

# Relative effect-size multipliers used by the simulator.
PLACEMENT_REACH = {"shelf": 1.0, "endcap": 0.85, "checkout": 0.6, "entrance": 0.35}
OFFER_STRENGTH = {"awareness": 1.0, "coupon": 1.5, "bundle": 1.3, "trial": 1.0}
TARGETING_FIT = {"blanket": 1.0, "category": 1.1, "crosssell": 1.0, "conquest": 1.1, "daypart": 1.0}
PERSONAL_OFFERS = {"coupon"}  # offers that leave a redemption record
DAYPARTS = {"Morning": 0.3, "Midday": 0.3, "Evening": 0.4}
TARGET_DAYPART = "Evening"
DOSE_LEVELS = {"Low": 0.45, "Medium": 0.9, "High": 1.4}
# key -> (label, baseline share of advertised-product sales, share of advertised lift transferred)
SECONDARY = {
    "neighbor": ("Neighboring own-brand product", 0.6, -0.30),
    "complement": ("Complementary product", 0.4, 0.25),
    "competitor": ("Competitor product", 0.7, -0.40),
}
STORE_COUNT, STORE_WEEKS, CAMPAIGN_START = 40, 16, 9
MARKET_LIFT = 0.06


@dataclass(frozen=True)
class Design:
    placement: str
    targeting: str
    offer: str
    rollout: str

    def describe(self) -> str:
        p = _PHRASES
        return f"{p['placement'][self.placement]} with {p['targeting'][self.targeting]}, {p['offer'][self.offer]} and {p['rollout'][self.rollout]}."

    def short(self) -> str:
        return " · ".join(DIMENSIONS[dim][getattr(self, dim)].split(" /")[0] for dim in DIMENSIONS)

    @property
    def code(self) -> int:
        index = [list(DIMENSIONS[dim]).index(getattr(self, dim)) for dim in DIMENSIONS]
        return ((index[0] * 5 + index[1]) * 5 + index[2]) * 4 + index[3]  # fixed radix keeps seeds stable if options change


@dataclass
class Recommendation:
    status: str  # valid | caution | invalid
    headline: str
    method: str
    detail: str
    level: str  # store | shopper
    grain: str
    exposure_proxy: str
    outcomes: list[str] = field(default_factory=list)
    cannot_support: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    analyses: list[tuple[str, bool, str]] = field(default_factory=list)
    plain: str = ""


PRESETS = {
    "Regional pilot · endcap coupon holdout": Design("endcap", "category", "coupon", "onoff"),
    "Daypart activation · checkout dose test": Design("checkout", "daypart", "coupon", "dose"),
    "Conquesting at shelf · staggered waves": Design("shelf", "conquest", "awareness", "staggered"),
    "Basket builder · shelf-edge cross-sell bundle": Design("shelf", "crosssell", "bundle", "onoff"),
    "Guardrail · Blanket blast": Design("entrance", "blanket", "awareness", "onoff"),
    "Guardrail · Unanchored product launch": Design("endcap", "blanket", "trial", "onoff"),
}

WEAK_DESIGNS = [
    {
        "name": "Blanket blast",
        "guardrail": "Keep a comparison group when the whole network launches",
        "design": PRESETS["Guardrail · Blanket blast"],
        "story": "Every store switches on entrance screens in the same week, just as a seasonal upswing hits the whole market. With no holdout, before-vs-after is the only comparison, so the season gets credited to the campaign.",
        "fix": Design("entrance", "blanket", "awareness", "staggered"),
        "fix_note": "Keep the blanket creative but launch in three waves. Stores that haven't launched yet absorb the market-wide shift.",
    },
    {
        "name": "Unanchored product launch",
        "guardrail": "Give new-product launches a comparison in time",
        "design": PRESETS["Guardrail · Unanchored product launch"],
        "story": "A new product launches with trial messaging in every store at once. There's no sales history to difference against and no store without the campaign, so the only number available is total product sales, and all of it gets called lift.",
        "fix": Design("endcap", "blanket", "trial", "staggered"),
        "fix_note": "Stagger the messaging across three waves and compare launched vs. not-yet-launched stores cross-sectionally, adjusting for category sales.",
    },
]

# Brand goals: the sales conversation starts here, and each goal pre-fills a matched design.
# (goal, what we'd propose, what the brand will be able to say afterwards, design)
GOALS = [
    ("Launch a new product", "Trial messaging on endcaps, rolled out in three waves.", "How much trial the campaign created, even with no sales history.", Design("endcap", "category", "trial", "staggered")),
    ("Win share from a competitor", "Conquesting creative at the shelf edge, rolled out in waves.", "The brand's lift, and the share it took from the competitor.", Design("shelf", "conquest", "awareness", "staggered")),
    ("Grow the basket", "A cross-sell bundle at the shelf edge, with holdout stores.", "Lift for the product, the halo on its complement, net of cannibalization.", Design("shelf", "crosssell", "bundle", "onoff")),
    ("Drive offer redemptions", "A digital coupon on endcaps, with holdout stores.", "Redemptions, and how much of them was truly incremental.", Design("endcap", "category", "coupon", "onoff")),
    ("Reach shoppers at the right moment", "An evening checkout coupon at three play frequencies.", "When the lift happens, and how much frequency is enough.", Design("checkout", "daypart", "coupon", "dose")),
    ("Find the right play frequency", "The same endcap message at low, medium and high frequency.", "How much frequency is enough before extra plays stop adding sales.", Design("endcap", "category", "awareness", "dose")),
]

_ALL = {dim: set(options) for dim, options in DIMENSIONS.items()}
MATURITY = [
    {
        "name": "Early-stage partner",
        "comparable": "Homeland",
        "summary": "A grocer new to retail media, like Homeland (15 stores, 2024). New launches such as Smart & Final's 25 stores start here, with the chain's other stores as a ready comparison group. Illustrative data: store-week POS.",
        "allowed": {"placement": {"endcap", "checkout", "entrance"}, "targeting": {"blanket", "category"}, "offer": {"awareness", "coupon"}, "rollout": {"onoff", "staggered"}},
        "measurement": "Store-level DiD or staggered DiD with holdout stores. Coupon redemptions reported as context, not as lift.",
        "example": Design("endcap", "category", "coupon", "onoff"),
    },
    {
        "name": "Mature partner",
        "comparable": "Save Mart",
        "summary": "An established network, like Save Mart's activation \"tailored to location and time of day\". Its growth in waves (15 stores, plans for 179 more) gives each wave a built-in comparison. Illustrative data: timestamps, traffic, play logs.",
        "allowed": {"placement": {"endcap", "checkout", "entrance"}, "targeting": {"blanket", "category", "conquest", "daypart"}, "offer": _ALL["offer"], "rollout": _ALL["rollout"]},
        "measurement": "Fixed-effects panel regression, dose-response curves, daypart-grain models and cross-sectional launch designs.",
        "example": Design("checkout", "daypart", "coupon", "dose"),
    },
    {
        "name": "Instrumented network",
        "comparable": "a mature network",
        "summary": "A network with aisle-level traffic counts and basket-level sales, so a shelf-edge screen can be tied to one product and the products around it. Most networks grow into this stage. Illustrative data: aisle traffic, basket POS.",
        "allowed": _ALL,
        "measurement": "Shelf-level reads with aisle traffic, lift net of cannibalization and cross-sell halo, and play-frequency optimization.",
        "example": Design("shelf", "crosssell", "bundle", "onoff"),
    },
]


def maturity_required(design: Design) -> int:
    for index, tier in enumerate(MATURITY):
        if all(getattr(design, dim) in tier["allowed"][dim] for dim in DIMENSIONS):
            return index
    return len(MATURITY) - 1


def secondary_keys(design: Design) -> list[str]:
    keys = []
    if design.placement == "shelf":
        keys.append("neighbor")
    if design.targeting == "crosssell":
        keys.append("complement")
    if design.targeting == "conquest":
        keys.append("competitor")
    return keys


def recommend(design: Design) -> Recommendation:
    blanket = design.targeting == "blanket"
    trial = design.offer == "trial"
    personal = design.offer in PERSONAL_OFFERS
    shelf = design.placement == "shelf"
    daypart = design.targeting == "daypart"
    rollout = design.rollout
    status, headline, warnings, cannot = "valid", "Design and measurement are matched", [], []

    level = "store"
    grain = "Store × daypart × week sales" if daypart else "Store × week sales"
    if blanket and rollout == "onoff":
        status = "invalid"
        if trial:
            headline, method = "No baseline and no comparison group", "Not identifiable"
            detail = "A new product has no sales history, and every store launches at once with the same content. Nothing is left to compare against, so the only available number is the product's total sales, not lift."
        else:
            headline, method = "No control group", "Pre/post comparison (confounded)"
            detail = "Every store switches on in the same week. The only comparison left is before vs. after, which credits any seasonal or market-wide shift at launch to the campaign."
    elif trial:
        method = {
            "onoff": "Cross-sectional comparison with a pre-period covariate (ANCOVA)",
            "staggered": "Cross-sectional comparison of launched vs. not-yet-launched stores",
            "dose": "Cross-sectional dose-response (ANCOVA)",
        }[rollout]
        detail = "The new product has no sales history, so difference-in-differences is impossible. Compare campaign-period sales across stores in the same weeks, adjusting for each store's pre-period category sales. Week FE, clustered by store."
        warnings.append("Cross-sectional comparisons lean on stores being comparable after the category-sales adjustment, so assign campaign stores at random.")
    else:
        method = {
            "onoff": "Difference-in-differences (two-way fixed effects)",
            "staggered": "Staggered difference-in-differences (two-way fixed effects)",
            "dose": "Fixed-effects panel regression on dose level",
        }[rollout]
        detail = {
            "onoff": "Campaign stores vs. holdout stores, before vs. after launch. Store FE remove baseline gaps; week FE remove market-wide movement. Clustered by store.",
            "staggered": "Each wave's launch is compared with stores that haven't launched yet or never will. Store and week FE, clustered by store.",
            "dose": "Low, medium and high play frequency enter as separate treatment indicators, tracing out the response curve. Store and week FE, clustered by store.",
        }[rollout]
    if blanket and rollout == "staggered":
        status, headline = "caution", "Identified, but only by timing"
        warnings.append("Only not-yet-launched stores serve as controls. After the final wave launches nothing is left to compare against, and uneven effects across waves can bias two-way FE (use a Callaway–Sant'Anna or Sun–Abraham estimator in production).")
        cannot.append("Effects in the weeks after the final wave launches.")
    if blanket and rollout == "dose":
        status, headline = "caution", "Only relative dose effects are identified"
        warnings.append("Every store gets some dose, so the model identifies high-vs-low differences but not lift relative to no campaign.")
        cannot.append("Absolute lift vs. no campaign: there are no zero-dose stores.")
    if rollout == "staggered" and not blanket:
        warnings.append("Two-way FE assumes the effect is similar across waves. If later waves respond differently, switch to a heterogeneity-robust staggered estimator.")
    if daypart and method != "Not identifiable":
        method += ", at daypart grain"
        detail += " Treatment is interacted with the targeted daypart; weekly totals would bury the evening effect in untargeted hours."

    if trial and status != "invalid":
        cannot.append("Before/after or DiD claims: the product has no sales history to difference against.")

    exposure_proxy = {
        "shelf": "Aisle traffic counts → likely impressions near one product",
        "endcap": "Store traffic → store-level exposure only",
        "checkout": "Transaction (lane) counts → near-universal exposure among buyers",
        "entrance": "Store footfall → broad reach, low precision; expect a small, diluted effect",
    }[design.placement]

    outcomes = ["New-product trial (first purchase)" if trial else "Advertised-product sales"]
    for key in secondary_keys(design):
        outcomes.append({
            "neighbor": "Neighboring own-brand product sales (cannibalization)",
            "complement": "Complementary product sales (cross-sell halo)",
            "competitor": "Competitor product sales (share taken)",
        }[key])
    if personal:
        outcomes.append("Offer redemption events")
    if design.offer == "bundle":
        outcomes.append("Basket-level units per transaction")
    if daypart:
        outcomes.append("Sales by daypart (hourly or daypart timestamps)")

    if not shelf:
        cannot.append("Cannibalization claims: without a shelf-edge screen there is no defined neighboring product.")
    cannot.append("Shopper-level effects: an in-store screen reaches everyone nearby, so results are read at store level.")
    if design.placement in {"endcap", "entrance", "checkout"}:
        cannot.append("Per-impression effects: exposure is known only at store or traffic level.")
    if design.offer in {"awareness", "bundle"}:
        cannot.append("Direct attribution: the effect is inferred from aggregate sales, with no redemption or scan to trace.")
    if not daypart:
        cannot.append("Time-of-day optimization: the design doesn't vary content by daypart.")
    if blanket and rollout == "onoff" and not trial:
        cannot.insert(0, "Incremental lift at all: blanket targeting with no holdout and a single launch can't separate the campaign from a general sales trend.")

    analyses = [
        ("Cannibalization decomposition", shelf, "Neighbor-product outcome tracked" if shelf else "Requires shelf-edge placement: only a screen next to one product defines a neighbor"),
        ("Cross-sell halo", design.targeting == "crosssell", "Complementary product tracked" if design.targeting == "crosssell" else "Requires cross-sell targeting"),
        ("Competitor share shift", design.targeting == "conquest", "Competitor product tracked" if design.targeting == "conquest" else "Requires competitive conquesting"),
        ("Daypart breakdown", daypart, "Daypart-grain outcome" if daypart else "Requires daypart targeting and timestamped sales"),
        ("Redemption vs. incrementality", personal, "Redemptions logged" if personal else "Requires a digital coupon tie-in"),
        ("Dose-response curve", rollout == "dose", "Three play-frequency levels" if rollout == "dose" else "Requires dose variation"),
        ("Staggered wave view", rollout == "staggered", "Three launch waves" if rollout == "staggered" else "Requires a staggered rollout"),
    ]
    if blanket and rollout == "onoff":
        plain = "Nothing to compare against: every store starts at once." if trial else "Compare sales before and after launch in the same stores. The season and the campaign get mixed together."
    elif trial:
        plain = {
            "onoff": "Compare new-product sales in campaign stores vs. holdout stores in the same weeks, adjusted for store size.",
            "staggered": "Compare stores that have launched with stores that haven't yet, week by week, adjusted for store size.",
            "dose": "Compare new-product sales across low, medium and high play frequency, adjusted for store size.",
        }[rollout]
    else:
        plain = {
            "onoff": "Compare how sales changed in campaign stores vs. holdout stores, before vs. after launch.",
            "staggered": "Compare each wave's stores with stores that haven't launched yet.",
            "dose": "Compare sales lift across low, medium and high play frequency.",
        }[rollout]
    if daypart and not (blanket and rollout == "onoff"):
        plain += " Read by daypart, so evening lift isn't diluted by the rest of the day."
    return Recommendation(status, headline, method, detail, level, grain, exposure_proxy, outcomes, cannot, warnings, analyses, plain)


def _dose_response(intensity: np.ndarray | float) -> np.ndarray:
    return (1 - np.exp(-1.1 * np.asarray(intensity))) / (1 - np.exp(-1.1 * DOSE_LEVELS["Medium"]))


def _fit(formula: str, data: pd.DataFrame, term: str, cluster: str) -> dict[str, float]:
    model = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data[cluster]})
    return _coefficient_result(model, term)


def _mean_test(values: np.ndarray) -> dict[str, float]:
    count = len(values)
    estimate = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(count))
    low, high = _interval(estimate, standard_error, count - 1)
    return {"estimate": estimate, "se": standard_error, "ci_low": float(low), "ci_high": float(high), "p_value": float(2 * t.sf(abs(estimate / standard_error), count - 1))}


def _store_level(design: Design, rng: np.random.Generator) -> dict[str, Any]:
    blanket = design.targeting == "blanket"
    trial = design.offer == "trial"
    daypart = design.targeting == "daypart"
    dose = design.rollout == "dose"
    no_control = blanket and design.rollout == "onoff"
    start = CAMPAIGN_START

    order = rng.permutation(STORE_COUNT)
    treated_order = order if blanket else order[: STORE_COUNT // 2]
    launch = np.full(STORE_COUNT, 99)
    group = np.full(STORE_COUNT, "Holdout stores", dtype=object)
    level = np.full(STORE_COUNT, "", dtype=object)
    intensity = np.zeros(STORE_COUNT)
    waves = [start, start + 2, start + 4] if trial else [start - 2, start, start + 2]
    for i, store in enumerate(treated_order):
        if design.rollout == "staggered":
            wave = i * 3 // len(treated_order)
            launch[store], group[store] = waves[wave], f"Wave {wave + 1}"
        else:
            launch[store] = start
            if dose:
                name = list(DOSE_LEVELS)[i % 3]
                level[store], intensity[store], group[store] = name, DOSE_LEVELS[name], f"{name} dose"
            else:
                group[store] = "All stores" if blanket else "Campaign stores"
    baseline = rng.normal(1000, 110, STORE_COUNT)

    names, shares = list(DAYPARTS), np.array(list(DAYPARTS.values()))
    parts = len(names)
    s = np.repeat(np.arange(STORE_COUNT), STORE_WEEKS * parts)
    w = np.tile(np.repeat(np.arange(1, STORE_WEEKS + 1), parts), STORE_COUNT)
    p = np.tile(np.arange(parts), STORE_COUNT * STORE_WEEKS)
    share = shares[p]
    # A market-wide seasonal upswing begins with the campaign window in every store.
    expected = baseline[s] * share * (1 + 0.025 * np.sin(w / 2.4)) * np.where(w >= start, 1 + MARKET_LIFT, 1.0)
    active = (w >= launch[s]).astype(int)
    target = p == names.index(TARGET_DAYPART)
    focus = np.where(target, 1.5, 0.0) if daypart else np.ones(s.size)
    scale = _dose_response(intensity[s]) if dose else 1.0
    fit = PLACEMENT_REACH[design.placement] * TARGETING_FIT[design.targeting]
    if trial:
        available = w >= start
        organic = np.where(available, 0.10 * expected, 0.0)
        lift = organic * 0.35 * fit * scale * focus * active
        sales = np.where(available, np.clip(organic + lift + rng.normal(0, 9, s.size) * np.sqrt(share), 0, None), 0.0)
    else:
        lift = expected * 0.08 * fit * OFFER_STRENGTH[design.offer] * scale * focus * active
        sales = expected + lift + rng.normal(0, 38, s.size) * np.sqrt(share)
    frame = pd.DataFrame({
        "store_id": [f"R{i + 1:02d}" for i in s], "week": w, "daypart": np.array(names)[p],
        "group": group[s], "level": level[s], "launch": launch[s], "active": active,
        "category": expected + rng.normal(0, 30, s.size) * np.sqrt(share), "sales": sales, "lift": lift,
    })
    secondaries = secondary_keys(design)
    for key in secondaries:
        _, base_share, transfer = SECONDARY[key]
        frame[key] = expected * base_share + transfer * lift + rng.normal(0, 25, s.size) * np.sqrt(share)
    measures = ["sales", "category", "lift", *secondaries]
    if design.offer in PERSONAL_OFFERS:
        # Redemptions include most incremental units plus baseline buyers who would have purchased anyway.
        frame["redemptions"] = np.clip(0.85 * lift + 0.05 * expected * active * (focus > 0) + rng.normal(0, 3, s.size), 0, None)
        measures.append("redemptions")

    weekly = frame.groupby(["store_id", "week"], as_index=False).agg(
        group=("group", "first"), level=("level", "first"), launch=("launch", "first"), active=("active", "max"),
        **{m: (m, "sum") for m in measures},
    )
    weekly["unit"] = weekly.store_id
    frame["unit"] = frame.store_id + "·" + frame.daypart
    data = frame if daypart else weekly
    data["treat"] = data.active * ((data.daypart == TARGET_DAYPART).astype(int) if daypart else 1)
    for name in DOSE_LEVELS:
        data[f"dose_{name.lower()}"] = data.treat * (data.level == name).astype(int)

    unit_label = ("new-product units" if trial else "units") + (" / store-week in the evening daypart" if daypart else " / store-week")
    result: dict[str, Any] = {"level": "store", "fmt": "units", "unit": unit_label, "rows": [], "decomposition": None,
                              "daypart": None, "dose": None, "redemption": None, "exposure": None}
    series = weekly.groupby(["week", "group"], as_index=False).sales.mean()
    result.update(series=series, launches=sorted({int(x) for x in launch if x < 99}),
                  series_y="Avg new-product units / store-week" if trial else "Avg advertised units / store-week")

    if no_control:
        per_store = weekly.assign(post=weekly.week >= start).groupby(["store_id", "post"]).sales.mean().unstack()
        values = (per_store[True] if trial else per_store[True] - per_store[False]).to_numpy()
        truth = float(weekly.loc[weekly.week >= start, "lift"].mean())
        label = "Average new-product sales, all credited to the campaign" if trial else "Before/after change in sales"
        primary = {"term": "prepost", "label": label, **_mean_test(values), "truth": truth}
        result.update(primary=primary, rows=[primary], unit="units / store-week")
        return result

    campaign_only = trial
    time_fe = "C(week):C(daypart)" if daypart else "C(week)"

    def prepare(table: pd.DataFrame, unit_col: str) -> tuple[pd.DataFrame, str]:
        if not campaign_only:
            return table, f"C({unit_col}) + {time_fe if table is data else 'C(week)'}"
        sample = table.loc[table.week >= start].copy()
        sample["category_pre"] = sample[unit_col].map(table.loc[table.week < start].groupby(unit_col).category.mean())
        return sample, f"category_pre + {time_fe if table is data else 'C(week)'}"

    sample, controls = prepare(data, "unit")
    terms = ["treat"]
    if dose:
        terms = [f"dose_{name.lower()}" for name in DOSE_LEVELS]
        if blanket:
            terms = terms[1:]
    model = smf.ols(f"sales ~ {' + '.join(terms)} + {controls}", data=sample).fit(cov_type="cluster", cov_kwds={"groups": sample.store_id})
    low_truth = float(sample.loc[sample.dose_low == 1, "lift"].mean()) if dose and blanket else 0.0
    rows = []
    for term in terms:
        truth = float(sample.loc[sample[term] == 1, "lift"].mean()) - low_truth
        if term == "treat":
            label = "Incremental sales (evening daypart)" if daypart else "Incremental sales"
        else:
            label = f"{term.split('_')[1].title()} dose" + (" vs. low dose" if blanket else "")
        rows.append({"term": term, "label": label, **_coefficient_result(model, term), "truth": truth})
    result["rows"] = rows
    result["primary"] = next(row for row in rows if row["term"] in {"treat", "dose_medium"})

    if dose:
        grid = np.linspace(0, 1.5, 100)
        medium_truth = float(sample.loc[sample.dose_medium == 1, "lift"].mean())
        curve = medium_truth * (_dose_response(grid) - (_dose_response(DOSE_LEVELS["Low"]) if blanket else 0))
        points = [{"intensity": DOSE_LEVELS[row["label"].split()[0]], **row} for row in rows]
        result["dose"] = {"points": points, "grid": grid, "truth_curve": curve, "relative": blanket}

    pooled_ok = not (blanket and dose)
    if pooled_ok and secondaries:
        decomposition = []
        for key in ["sales", *secondaries]:
            estimate = _fit(f"{key} ~ treat + {controls}", sample, "treat", "store_id")
            transfer = 1.0 if key == "sales" else SECONDARY[key][2]
            label = "Advertised product" if key == "sales" else SECONDARY[key][0]
            decomposition.append({"key": key, "label": label, **estimate, "truth": transfer * float(sample.loc[sample.treat == 1, "lift"].mean())})
        result["decomposition"] = decomposition

    if daypart and pooled_ok:
        for name in names:
            data[f"active_{name.lower()}"] = data.active * (data.daypart == name).astype(int)
        sample, controls = prepare(data, "unit")
        dp_terms = [f"active_{name.lower()}" for name in names]
        dp_model = smf.ols(f"sales ~ {' + '.join(dp_terms)} + {controls}", data=sample).fit(cov_type="cluster", cov_kwds={"groups": sample.store_id})
        breakdown = [{"daypart": name, **_coefficient_result(dp_model, term), "truth": float(sample.loc[sample[term] == 1, "lift"].mean())} for name, term in zip(names, dp_terms)]
        weekly_sample, weekly_controls = prepare(weekly, "unit")
        weekly_fit = _fit(f"sales ~ active + {weekly_controls}", weekly_sample, "active", "store_id")
        weekly_fit["truth"] = float(weekly_sample.loc[weekly_sample.active == 1, "lift"].mean())
        result["daypart"] = {"breakdown": breakdown, "weekly": weekly_fit, "targeted": result["primary"]}

    if "redemptions" in data and pooled_ok:
        treated = sample.loc[sample.treat == 1]
        result["redemption"] = {"attributed": float(treated.redemptions.mean()), "incremental": result["primary"]["estimate"] if not dose else _fit(f"sales ~ treat + {controls}", sample, "treat", "store_id")["estimate"], "truth": float(treated.lift.mean())}
    return result


def simulate(design: Design, seed: int = SEED) -> dict[str, Any]:
    """Generate data matching the design and run the recommended measurement."""
    rng = np.random.default_rng(seed + 1000 + design.code)
    return _store_level(design, rng)
