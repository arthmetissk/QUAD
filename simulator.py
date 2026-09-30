"""Deterministic synthetic campaign data and statistical measurements."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize_scalar
from scipy.stats import t, ttest_ind

SEED = 20260929


def _interval(estimate: float, standard_error: float, degrees_freedom: float) -> tuple[float, float]:
    critical = t.ppf(0.975, degrees_freedom)
    return estimate - critical * standard_error, estimate + critical * standard_error


def _coefficient_result(model: Any, term: str) -> dict[str, float]:
    estimate = float(model.params[term])
    standard_error = float(model.bse[term])
    lower, upper = model.conf_int().loc[term].astype(float).tolist()
    return {
        "estimate": estimate,
        "se": standard_error,
        "ci_low": float(lower),
        "ci_high": float(upper),
        "p_value": float(model.pvalues[term]),
    }


def generate_easy(seed: int = SEED) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 1)
    store_count, pre_weeks, campaign_weeks = 40, 8, 8
    treatment_ids = set(rng.choice(store_count, size=20, replace=False).tolist())
    store_baselines = rng.normal(1000, 110, store_count)
    rows = []
    for store in range(store_count):
        treatment = int(store in treatment_ids)
        store_effect = rng.normal(0, 22)
        for week in range(pre_weeks + campaign_weeks):
            is_campaign = week >= pre_weeks
            week_effect = 12 * np.sin(week / 2.8) + 1.2 * week
            true_lift = store_baselines[store] * 0.08 if treatment and is_campaign else 0
            sales = store_baselines[store] + store_effect + week_effect + true_lift + rng.normal(0, 38)
            rows.append({
                "store_id": f"S{store + 1:02d}", "week": week + 1,
                "period": "Campaign" if is_campaign else "Pre-period",
                "group": "Treatment" if treatment else "Control",
                "sales": sales,
            })
    data = pd.DataFrame(rows)
    store_period = data.groupby(["store_id", "group", "period"], as_index=False).sales.mean()
    pivot = store_period.pivot(index=["store_id", "group"], columns="period", values="sales").reset_index()
    pivot["change"] = pivot["Campaign"] - pivot["Pre-period"]
    treated = pivot.loc[pivot.group == "Treatment", "change"].to_numpy()
    control = pivot.loc[pivot.group == "Control", "change"].to_numpy()
    test = ttest_ind(treated, control, equal_var=False)
    variance_t, variance_c = treated.var(ddof=1) / len(treated), control.var(ddof=1) / len(control)
    standard_error = float(np.sqrt(variance_t + variance_c))
    df = float((variance_t + variance_c) ** 2 / (variance_t**2 / (len(treated) - 1) + variance_c**2 / (len(control) - 1)))
    estimate = float(treated.mean() - control.mean())
    low, high = _interval(estimate, standard_error, df)
    means = data.groupby(["week", "group"], as_index=False).sales.mean()
    pre_gap = float(pivot.loc[pivot.group == "Treatment", "Pre-period"].mean() - pivot.loc[pivot.group == "Control", "Pre-period"].mean())
    post_gap = float(pivot.loc[pivot.group == "Treatment", "Campaign"].mean() - pivot.loc[pivot.group == "Control", "Campaign"].mean())
    true_effect = float(np.mean([store_baselines[store] * 0.08 for store in treatment_ids]))
    return {
        "data": data, "means": means, "pre_gap": pre_gap, "post_gap": post_gap,
        "estimate": estimate, "ci_low": low, "ci_high": high,
        "p_value": float(test.pvalue), "true_effect": true_effect,
        "true_lift_pct": 8.0,
    }


def generate_medium(seed: int = SEED) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 2)
    wave_launches = {1: 5, 2: 7, 3: 9}
    amplitude, true_b = 150.0, 1.1
    intensity_levels = np.array([0.45, 0.9, 1.4])
    rows = []
    for store in range(60):
        treatment = store < 30
        wave = store // 10 + 1 if treatment else 0
        launch = wave_launches.get(wave, 99)
        intensity = float(rng.choice(intensity_levels)) if treatment else 0.0
        baseline = rng.normal(1000, 95)
        store_noise = rng.normal(0, 18)
        for week in range(1, 17):
            time_effect = 18 * np.sin(week / 3.1) + week * 1.4
            dose = intensity if week >= launch else 0.0
            true_lift = amplitude * (1 - np.exp(-true_b * dose)) if dose else 0.0
            sales = baseline + store_noise + time_effect + true_lift + rng.normal(0, 35)
            rows.append({
                "store_id": f"M{store + 1:02d}", "week": week,
                "wave": f"Wave {wave}" if treatment else "Control",
                "launch_week": launch if treatment else np.nan,
                "intensity": intensity, "dose": dose, "sales": sales,
            })
    data = pd.DataFrame(rows)

    def fit_profile(b_value: float) -> tuple[float, float]:
        basis = 1 - np.exp(-b_value * data.dose.to_numpy())
        y = data.sales.to_numpy()
        y_within = y - data.groupby("store_id").sales.transform("mean").to_numpy() - data.groupby("week").sales.transform("mean").to_numpy() + y.mean()
        x = basis - pd.Series(basis).groupby(data.store_id).transform("mean").to_numpy() - pd.Series(basis).groupby(data.week).transform("mean").to_numpy() + basis.mean()
        coefficient = float(np.dot(x, y_within) / np.dot(x, x))
        return float(np.sum((y_within - coefficient * x) ** 2)), coefficient

    optimum = minimize_scalar(lambda candidate: fit_profile(candidate)[0], bounds=(0.25, 2.5), method="bounded")
    fitted_b = float(optimum.x)
    data["dose_response"] = 1 - np.exp(-fitted_b * data.dose)
    model = smf.ols("sales ~ dose_response + C(store_id) + C(week)", data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data.store_id}
    )
    response = _coefficient_result(model, "dose_response")
    data["twfe_sales"] = (
        data.sales - data.groupby("store_id").sales.transform("mean")
        - data.groupby("week").sales.transform("mean") + data.sales.mean()
    )
    pre_launch = data.loc[data.week < min(wave_launches.values())].copy()
    pre_launch["group"] = np.where(pre_launch.wave == "Control", "Control", "Treatment")
    pretrend = pre_launch.groupby(["week", "group"], as_index=False).sales.mean()
    pre_launch["treated_trend"] = (pre_launch.group == "Treatment").astype(int) * pre_launch.week
    pretrend_model = smf.ols("sales ~ treated_trend + C(store_id) + C(week)", data=pre_launch).fit(
        cov_type="cluster", cov_kwds={"groups": pre_launch.store_id}
    )
    pretrend_test = _coefficient_result(pretrend_model, "treated_trend")
    dose_points = data.loc[data.dose > 0].copy()
    dose_points["dose_bin"] = dose_points.dose.round(2)
    binned = dose_points.groupby("dose_bin", as_index=False).twfe_sales.mean()
    true_lift = float(amplitude * (1 - np.exp(-true_b * 0.9)))
    return {
        "data": data, "response": response,
        "fitted_b": fitted_b, "true_b": true_b, "amplitude": amplitude,
        "true_effect": true_lift, "pretrend": pretrend, "pretrend_test": pretrend_test, "binned": binned,
        "wave_launches": wave_launches,
    }


def generate_sophisticated(seed: int = SEED) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 3)
    store_count, shoppers_per_store, weeks = 24, 100, 8
    gross_effect, cannibalization_share = 2.0, 0.30
    shopper_rows = []
    traffic_by_store_week = {(store, week): int(rng.integers(45, 221)) for store in range(store_count) for week in range(1, weeks + 1)}
    for store in range(store_count):
        baseline = rng.normal(10, 1.8)
        assignments = rng.binomial(1, 0.5, shoppers_per_store)
        for shopper in range(shoppers_per_store):
            shopper_baseline = baseline + rng.normal(0, 1.2)
            assigned = int(assignments[shopper])
            for week in range(1, weeks + 1):
                traffic = traffic_by_store_week[(store, week)]
                exposure_probability = float(np.clip(0.30 + 0.003 * traffic, 0.35, 0.92))
                actual_exposure = int(assigned and rng.random() < exposure_probability)
                week_effect = 0.25 * np.sin(week / 1.8) + week * 0.035
                noise = rng.normal(0, 2.1)
                advertised_sales = shopper_baseline + week_effect + gross_effect * actual_exposure + noise
                neighbor_sales = 6 + 0.25 * shopper_baseline + week_effect * 0.3 - gross_effect * cannibalization_share * actual_exposure + rng.normal(0, 1.8)
                shopper_rows.append({
                    "loyalty_id": f"L{store:02d}-{shopper:03d}", "store_id": f"Q{store + 1:02d}",
                    "week": week, "advertised_product_sales": advertised_sales,
                    "neighbor_product_sales": neighbor_sales, "assigned": assigned,
                    "actual_exposure": actual_exposure, "aisle_traffic_count": traffic,
                    "exposure_probability": exposure_probability,
                    "traffic_weighted_dose": assigned * exposure_probability,
                })
    data = pd.DataFrame(shopper_rows)
    cluster = data.loyalty_id
    advertised_itt_model = smf.ols(
        "advertised_product_sales ~ assigned + C(store_id) + C(week)", data=data
    ).fit(cov_type="cluster", cov_kwds={"groups": cluster})
    advertised_dose_model = smf.ols(
        "advertised_product_sales ~ traffic_weighted_dose + C(store_id) + C(week)", data=data
    ).fit(cov_type="cluster", cov_kwds={"groups": cluster})
    neighbor_model = smf.ols(
        "neighbor_product_sales ~ traffic_weighted_dose + C(store_id) + C(week)", data=data
    ).fit(cov_type="cluster", cov_kwds={"groups": cluster})
    itt = _coefficient_result(advertised_itt_model, "assigned")
    adjusted = _coefficient_result(advertised_dose_model, "traffic_weighted_dose")
    neighbor = _coefficient_result(neighbor_model, "traffic_weighted_dose")
    mean_probability = float(data.loc[data.assigned == 1, "exposure_probability"].mean())
    exposure_rate = float(data.loc[data.assigned == 1, "actual_exposure"].mean())
    net_estimate = adjusted["estimate"] + neighbor["estimate"]
    return {
        "data": data, "itt": itt, "adjusted": adjusted, "neighbor": neighbor,
        "mean_probability": mean_probability, "exposure_rate": exposure_rate,
        "gross_effect": gross_effect, "cannibalization_share": cannibalization_share,
        "true_cannibalization": gross_effect * cannibalization_share,
        "true_net": gross_effect * (1 - cannibalization_share),
        "net_estimate": net_estimate,
    }


def load_demo_data(seed: int = SEED) -> dict[str, Any]:
    """Single entry point for the reproducible synthetic data generator."""
    return {
        "easy": generate_easy(seed),
        "medium": generate_medium(seed),
        "sophisticated": generate_sophisticated(seed),
    }
