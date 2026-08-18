#!/usr/bin/env python3
"""Local-only NUTS sampling service for Flux MMM."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / ".flux-artifacts" / "mcmc"
MAX_BODY_BYTES = 32_000_000
SAMPLING_CONTRACT_VERSION = (
    "flux-pymc-nuts-v2.2.0-full-response-same-window-experiment-roi"
)


@dataclass
class Job:
    id: str
    fingerprint: str
    status: str = "queued"
    progress: dict[str, Any] = field(
        default_factory=lambda: {
            "stage": "queued",
            "completed": 0,
            "total": 1,
            "chain": 0,
            "detail": "Sampling job queued.",
        }
    )
    result: dict[str, Any] | None = None
    error: str | None = None


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def safe_number(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else fallback
    except (TypeError, ValueError):
        return fallback


def hdi_bounds(values: np.ndarray, probability: float = 0.95) -> tuple[float, float]:
    flattened = np.sort(np.asarray(values, dtype=float).reshape(-1))
    if flattened.size == 0:
        return 0.0, 0.0
    interval_count = max(1, min(int(math.floor(probability * flattened.size)), flattened.size - 1))
    widths = flattened[interval_count:] - flattened[: flattened.size - interval_count]
    start = int(np.argmin(widths)) if widths.size else 0
    end = min(start + interval_count, flattened.size - 1)
    return float(flattened[start]), float(flattened[end])


def posterior_summary(values: np.ndarray) -> tuple[float, float, float, float, float]:
    low, high = hdi_bounds(values)
    return (
        float(np.mean(values)),
        low,
        float(np.median(values)),
        high,
        float(np.std(values, ddof=1)),
    )


def hdi_by_column(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    bounds = [hdi_bounds(array[:, index]) for index in range(array.shape[1])]
    return (
        np.asarray([bound[0] for bound in bounds], dtype=float),
        np.asarray([bound[1] for bound in bounds], dtype=float),
    )


def update_job(job: Job, **progress: Any) -> None:
    with JOBS_LOCK:
        job.progress = {**job.progress, **progress}


def lognormal_parameters(mean: float, sd: float) -> tuple[float, float]:
    safe_mean = max(mean, 1e-9)
    safe_sd = max(sd, safe_mean * 0.05, 1e-9)
    variance = math.log1p((safe_sd / safe_mean) ** 2)
    return math.log(safe_mean) - variance / 2, math.sqrt(variance)


def lag_matrix(values: np.ndarray, lag_count: int) -> np.ndarray:
    matrix = np.zeros((values.size, lag_count), dtype=float)
    for lag in range(lag_count):
        if lag == 0:
            matrix[:, lag] = values
        else:
            matrix[lag:, lag] = values[:-lag]
    return matrix


def gaussian_basis(row_count: int, knot_count: int, bandwidth: Any) -> Any:
    times = np.linspace(0, 1, row_count, dtype=float)[:, None]
    knots = np.linspace(0, 1, max(knot_count, 1), dtype=float)[None, :]
    raw = pt.exp(-((times - knots) ** 2) / (2 * bandwidth**2))
    return raw / pt.maximum(pt.sum(raw, axis=1, keepdims=True), 1e-12)


def relative_lognormal(name: str, center: float, relative_sd: float, initval: float) -> Any:
    mean = max(center, 1e-6)
    sd = max(mean * relative_sd, 1e-6)
    log_mu, log_sigma = lognormal_parameters(mean, sd)
    return pm.LogNormal(name, mu=log_mu, sigma=log_sigma, initval=max(initval, 1e-6))


def validate_compiled_model(compiled: dict[str, Any]) -> None:
    """Reject stale or incomplete browser payloads before starting NUTS."""
    if compiled.get("version") != SAMPLING_CONTRACT_VERSION:
        raise ValueError(
            "This Production payload was compiled by an older Flux version. "
            "Refresh the workspace, restore or re-promote the winning candidate, "
            "and start a new sampling run."
        )
    matrix = np.asarray(compiled.get("matrix", []), dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or not np.isfinite(matrix).all():
        raise ValueError("The compiled sampling design must be a finite two-dimensional matrix.")
    if not compiled.get("response"):
        raise ValueError("The compiled sampling payload is missing its response contract.")
    media_contracts = compiled.get("media", [])
    if not media_contracts:
        raise ValueError("The compiled sampling payload contains no media channels.")
    for media in media_contracts:
        channel = str(media.get("channel", "media channel"))
        raw_spend = media.get("rawSpend")
        if not isinstance(raw_spend, list) or len(raw_spend) != matrix.shape[0]:
            raise ValueError(
                f"The Production payload for {channel} is stale because raw spend "
                "history is missing. Refresh the workspace and start a new sampling run."
            )
        spend_array = np.asarray(raw_spend, dtype=float)
        if not np.isfinite(spend_array).all() or np.any(spend_array < 0):
            raise ValueError(
                f"The Production payload for {channel} contains invalid raw spend values."
            )
        if compiled["response"].get("channelSpecific"):
            channel_response = media.get("response")
            required_response_fields = {
                "adstockType",
                "adstock",
                "weibullShape",
                "weibullScale",
                "saturation",
                "halfSaturationQuantile",
                "kernelNormalization",
            }
            missing_fields = sorted(
                required_response_fields - set(channel_response or {})
            )
            if missing_fields:
                raise ValueError(
                    f"The Production payload for {channel} is missing its channel "
                    f"response contract: {', '.join(missing_fields)}. Refresh the "
                    "workspace and start a new sampling run."
                )


def build_model(compiled: dict[str, Any]):
    validate_compiled_model(compiled)
    matrix = np.asarray(compiled["matrix"], dtype=float)
    target = np.asarray(compiled["target"], dtype=float)
    outcome = np.asarray(compiled["outcome"], dtype=float)
    priors = compiled["priors"]
    parameter_names = compiled["parameterNames"]
    target_scale = max(float(np.std(target)), 1e-3)
    target_center = float(np.mean(target))
    standardized_target = (target - target_center) / target_scale

    response = compiled.get("response", {})
    media_contracts = compiled.get("media", [])
    media_parameter_indexes = {
        int(index)
        for media in media_contracts
        for index in media.get("indexes", [])
    }
    planning_parameter_index = response.get("planningParameterIndex")
    excluded_indexes = set(media_parameter_indexes)
    if planning_parameter_index is not None:
        excluded_indexes.add(int(planning_parameter_index))
    baseline_indexes = [
        index for index in range(len(priors)) if index not in excluded_indexes
    ]

    with pm.Model() as model:
        coefficients = []
        initial_values: dict[str, float] = {}
        for index, prior in enumerate(priors):
            variable_name = f"beta_{index:03d}"
            raw_name = f"raw_{index:03d}"
            kind = prior["kind"]
            prior_mean = safe_number(prior.get("mean"))
            prior_sd = max(safe_number(prior.get("standardDeviation"), 1.0), 1e-8)
            initial = safe_number(prior.get("initialValue"), prior_mean)
            if kind == "normal":
                raw = pm.Normal(
                    raw_name,
                    mu=0,
                    sigma=1,
                    initval=(initial - prior_mean) / prior_sd,
                )
                coefficient = pm.Deterministic(
                    variable_name,
                    prior_mean + prior_sd * raw,
                )
            elif kind == "log-normal":
                log_mu, log_sigma = lognormal_parameters(prior_mean, prior_sd)
                raw = pm.Normal(
                    raw_name,
                    mu=0,
                    sigma=1,
                    initval=(math.log(max(initial, 1e-8)) - log_mu) / log_sigma,
                )
                coefficient = pm.Deterministic(
                    variable_name,
                    pt.exp(log_mu + log_sigma * raw),
                )
            elif kind == "truncated-normal":
                lower = -prior_mean / prior_sd
                raw = pm.TruncatedNormal(
                    raw_name,
                    mu=0,
                    sigma=1,
                    lower=lower,
                    initval=max((initial - prior_mean) / prior_sd, lower + 1e-6),
                )
                coefficient = pm.Deterministic(
                    variable_name,
                    prior_mean + prior_sd * raw,
                )
            else:
                raw = pm.HalfNormal(
                    raw_name,
                    sigma=1,
                    initval=max(initial / prior_sd, 1e-8),
                )
                coefficient = pm.Deterministic(
                    variable_name,
                    prior_sd * raw,
                )
            coefficients.append(coefficient)
            if kind == "normal":
                initial_values[raw_name] = (initial - prior_mean) / prior_sd
            elif kind == "log-normal":
                initial_values[raw_name] = (
                    math.log(max(initial, 1e-8)) - log_mu
                ) / log_sigma
            elif kind == "truncated-normal":
                initial_values[raw_name] = max(
                    (initial - prior_mean) / prior_sd,
                    -prior_mean / prior_sd + 1e-6,
                )
            else:
                initial_values[raw_name] = max(initial / prior_sd, 1e-8)

        beta = pt.stack(coefficients)
        linear_predictor = pt.dot(
            matrix[:, baseline_indexes],
            beta[np.asarray(baseline_indexes, dtype=int)],
        )

        channel_specific_response = bool(response.get("channelSpecific"))
        if not channel_specific_response:
            hill_shape = relative_lognormal(
                "hill_shape",
                safe_number(response.get("hillShape"), 1.2),
                0.25,
                safe_number(response.get("hillShape"), 1.2),
            )
            initial_values["hill_shape"] = safe_number(response.get("hillShape"), 1.2)
        else:
            hill_shape = None
        if not channel_specific_response and response.get("adstockType") == "weibull":
            weibull_shape = relative_lognormal(
                "weibull_shape",
                safe_number(response.get("weibullShape"), 2.5),
                0.25,
                safe_number(response.get("weibullShape"), 2.5),
            )
            weibull_scale = relative_lognormal(
                "weibull_scale",
                safe_number(response.get("weibullScale"), 4.0),
                0.25,
                safe_number(response.get("weibullScale"), 4.0),
            )
            initial_values["weibull_shape"] = safe_number(response.get("weibullShape"), 2.5)
            initial_values["weibull_scale"] = safe_number(response.get("weibullScale"), 4.0)
            adstock_decay = None
        elif not channel_specific_response:
            decay_center = min(max(safe_number(response.get("adstockDecay"), 0.35), 0.01), 0.99)
            concentration = 30.0
            adstock_decay = pm.Beta(
                "adstock_decay",
                alpha=decay_center * concentration,
                beta=(1 - decay_center) * concentration,
                initval=decay_center,
            )
            initial_values["adstock_decay"] = decay_center
            weibull_shape = None
            weibull_scale = None
        else:
            adstock_decay = None
            weibull_shape = None
            weibull_scale = None

        if response.get("timeVarying"):
            kernel_bandwidth = relative_lognormal(
                "kernel_bandwidth",
                safe_number(response.get("kernelBandwidth"), 0.18),
                0.25,
                safe_number(response.get("kernelBandwidth"), 0.18),
            )
            initial_values["kernel_bandwidth"] = safe_number(response.get("kernelBandwidth"), 0.18)
        else:
            kernel_bandwidth = pt.as_tensor_variable(1.0)

        if response.get("planningIntensity") and planning_parameter_index is not None:
            planning_knots = min(6, max(3, matrix.shape[0] // 26))
            planning_bandwidth = 0.22
            time_values = np.linspace(0, 1, matrix.shape[0], dtype=float)[:, None]
            knot_values = np.linspace(0, 1, planning_knots, dtype=float)[None, :]
            planning_basis_np = np.exp(
                -((time_values - knot_values) ** 2) / (2 * planning_bandwidth**2)
            )
            planning_basis_np /= np.maximum(planning_basis_np.sum(axis=1, keepdims=True), 1e-12)
            planning_initial = np.asarray(response.get("planningInitial", []), dtype=float)
            if planning_initial.size != matrix.shape[0]:
                planning_initial = np.zeros(matrix.shape[0], dtype=float)
            planning_deviation_weight = pm.Normal(
                "planning_deviation_weight",
                mu=0,
                sigma=0.15,
                shape=planning_knots,
                initval=np.zeros(planning_knots),
            )
            planning_path = pm.Deterministic(
                "planning_factor",
                planning_initial
                + pt.dot(planning_basis_np, planning_deviation_weight),
            )
            linear_predictor = (
                linear_predictor
                + coefficients[int(planning_parameter_index)] * planning_path
            )

        media_contributions = []
        media_runtimes = []
        for media_index, media in enumerate(media_contracts):
            raw_spend = np.asarray(media["rawSpend"], dtype=float)
            media_response = media.get("response", response)
            if channel_specific_response:
                suffix = f"_{media_index:03d}"
                channel_hill_shape = relative_lognormal(
                    f"hill_shape{suffix}",
                    safe_number(media_response.get("saturation"), 1.2),
                    0.25,
                    safe_number(media_response.get("saturation"), 1.2),
                )
                initial_values[f"hill_shape{suffix}"] = safe_number(
                    media_response.get("saturation"), 1.2
                )
                if media_response.get("adstockType") == "weibull":
                    channel_weibull_shape = relative_lognormal(
                        f"weibull_shape{suffix}",
                        safe_number(media_response.get("weibullShape"), 2.5),
                        0.25,
                        safe_number(media_response.get("weibullShape"), 2.5),
                    )
                    channel_weibull_scale = relative_lognormal(
                        f"weibull_scale{suffix}",
                        safe_number(media_response.get("weibullScale"), 4.0),
                        0.25,
                        safe_number(media_response.get("weibullScale"), 4.0),
                    )
                    initial_values[f"weibull_shape{suffix}"] = safe_number(
                        media_response.get("weibullShape"), 2.5
                    )
                    initial_values[f"weibull_scale{suffix}"] = safe_number(
                        media_response.get("weibullScale"), 4.0
                    )
                    channel_adstock_decay = None
                else:
                    decay_center = min(
                        max(safe_number(media_response.get("adstock"), 0.35), 0.01),
                        0.99,
                    )
                    concentration = 30.0
                    channel_adstock_decay = pm.Beta(
                        f"adstock_decay{suffix}",
                        alpha=decay_center * concentration,
                        beta=(1 - decay_center) * concentration,
                        initval=decay_center,
                    )
                    initial_values[f"adstock_decay{suffix}"] = decay_center
                    channel_weibull_shape = None
                    channel_weibull_scale = None
            else:
                channel_hill_shape = hill_shape
                channel_weibull_shape = weibull_shape
                channel_weibull_scale = weibull_scale
                channel_adstock_decay = adstock_decay

            if media_response.get("adstockType") == "weibull":
                lag_count = min(53, raw_spend.size)
                lags = np.arange(lag_count, dtype=float) + 0.5
                ratio = lags / channel_weibull_scale
                weights = (
                    (channel_weibull_shape / channel_weibull_scale)
                    * ratio ** (channel_weibull_shape - 1)
                    * pt.exp(-(ratio**channel_weibull_shape))
                )
                weights = weights / pt.maximum(pt.max(weights), 1e-12)
            else:
                lag_count = raw_spend.size
                weights = channel_adstock_decay ** np.arange(lag_count, dtype=float)
            carried = pt.dot(lag_matrix(raw_spend, lag_count), weights)
            if media_response.get("kernelNormalization") == "sum":
                carried = carried * raw_spend.sum() / pt.maximum(pt.sum(carried), 1e-12)
            half_saturation = max(safe_number(media.get("halfSaturation"), 1.0), 1e-9)
            powered_carried = pt.exp(
                channel_hill_shape * pt.log(pt.maximum(carried, 1e-9))
            )
            powered_half = pt.exp(channel_hill_shape * math.log(half_saturation))
            transformed = powered_carried / pt.maximum(
                powered_carried + powered_half,
                1e-12,
            )
            indexes = [int(index) for index in media["indexes"]]
            if response.get("timeVarying"):
                kernel = gaussian_basis(
                    raw_spend.size,
                    int(response.get("kernelKnots", len(indexes))),
                    kernel_bandwidth,
                )
                coefficient_path = pt.dot(
                    kernel,
                    beta[np.asarray(indexes, dtype=int)],
                )
            else:
                coefficient_path = coefficients[indexes[0]]
            contribution = pm.Deterministic(
                f"media_contribution_{media_index:03d}",
                transformed * coefficient_path,
            )
            media_contributions.append(contribution)
            media_runtimes.append(
                {
                    "raw_spend": raw_spend,
                    "weights": weights,
                    "lag_count": lag_count,
                    "normalize": media_response.get("kernelNormalization") == "sum",
                    "hill_shape": channel_hill_shape,
                    "half_saturation": half_saturation,
                    "transformed": transformed,
                    "coefficient_path": coefficient_path,
                }
            )
            linear_predictor = linear_predictor + contribution

        linear_predictor = pm.Deterministic("linear_predictor", linear_predictor)
        standardized_predictor = (
            linear_predictor - target_center
        ) / target_scale
        sigma_raw = pm.HalfNormal("sigma_raw", sigma=1, initval=0.5)
        sigma = pm.Deterministic("sigma", sigma_raw * target_scale)
        initial_values["sigma_raw"] = 0.5
        likelihood = compiled["likelihood"]
        if likelihood == "student-t":
            pm.StudentT(
                "observed_outcome",
                nu=max(safe_number(compiled.get("studentTDegreesFreedom"), 4), 2.1),
                mu=standardized_predictor,
                sigma=sigma_raw,
                observed=standardized_target,
            )
        elif likelihood == "log-normal":
            pm.Normal(
                "observed_outcome",
                mu=standardized_predictor,
                sigma=sigma_raw,
                observed=standardized_target,
            )
        else:
            pm.Normal(
                "observed_outcome",
                mu=standardized_predictor,
                sigma=sigma_raw,
                observed=standardized_target,
            )

        if likelihood == "log-normal":
            mean_outcome = pm.Deterministic(
                "mean_outcome",
                pt.exp(pt.clip(linear_predictor + 0.5 * sigma**2, -30, 30)),
            )
        else:
            mean_outcome = pm.Deterministic("mean_outcome", linear_predictor)

        def roi_function(media_index: int, rows: list[int]) -> Any:
            media = media_contracts[media_index]
            selected_rows = np.asarray(rows, dtype=int)
            spend = max(
                float(np.sum(np.asarray(media["rawSpend"], dtype=float)[selected_rows])),
                1.0,
            )
            contribution = media_contributions[media_index]
            if likelihood == "log-normal":
                without_channel = pt.exp(
                    pt.clip(
                        linear_predictor - contribution + 0.5 * sigma**2,
                        -30,
                        30,
                    )
                )
                return pt.sum(mean_outcome[selected_rows] - without_channel[selected_rows]) / spend
            return pt.sum(contribution[selected_rows]) / spend

        def experiment_roi(calibration: dict[str, Any], media_index: int) -> Any:
            runtime = media_runtimes[media_index]
            raw_spend = np.asarray(runtime["raw_spend"], dtype=float)
            spend_rows = np.asarray(
                calibration.get("spendRows", calibration.get("rows", [])),
                dtype=int,
            )
            outcome_rows = np.asarray(
                calibration.get("outcomeRows", calibration.get("rows", [])),
                dtype=int,
            )
            spend_rows = spend_rows[
                (spend_rows >= 0) & (spend_rows < raw_spend.size)
            ]
            outcome_rows = outcome_rows[
                (outcome_rows >= 0) & (outcome_rows < raw_spend.size)
            ]
            counterfactual_spend = raw_spend.copy()
            counterfactual_spend[spend_rows] = 0.0
            counterfactual_carried = pt.dot(
                lag_matrix(counterfactual_spend, int(runtime["lag_count"])),
                runtime["weights"],
            )
            if runtime["normalize"]:
                counterfactual_carried = (
                    counterfactual_carried
                    * counterfactual_spend.sum()
                    / pt.maximum(pt.sum(counterfactual_carried), 1e-12)
                )
            powered_counterfactual = pt.exp(
                runtime["hill_shape"]
                * pt.log(pt.maximum(counterfactual_carried, 1e-9))
            )
            powered_half = pt.exp(
                runtime["hill_shape"]
                * math.log(max(float(runtime["half_saturation"]), 1e-9))
            )
            counterfactual_transformed = powered_counterfactual / pt.maximum(
                powered_counterfactual + powered_half,
                1e-12,
            )
            delta_contribution = (
                runtime["transformed"] - counterfactual_transformed
            ) * runtime["coefficient_path"]
            spend = max(safe_number(calibration.get("spend"), 1.0), 1.0)
            if likelihood == "log-normal":
                without_test_spend = pt.exp(
                    pt.clip(
                        linear_predictor[outcome_rows]
                        - delta_contribution[outcome_rows]
                        + 0.5 * sigma**2,
                        -30,
                        30,
                    )
                )
                return pt.sum(
                    mean_outcome[outcome_rows] - without_test_spend
                ) / spend
            return pt.sum(delta_contribution[outcome_rows]) / spend

        all_rows = list(range(matrix.shape[0]))
        for media_index, media in enumerate(media_contracts):
            evidence = media.get("priorEvidence")
            if evidence and evidence.get("source") != "experiment":
                evidence_rows = evidence.get("rows") or all_rows
                pm.Potential(
                    f"roi_prior_{media_index:03d}",
                    pm.logp(
                        pm.Normal.dist(
                            mu=safe_number(evidence.get("mean")),
                            sigma=max(safe_number(evidence.get("standardDeviation"), 1.0), 1e-6),
                        ),
                        roi_function(media_index, evidence_rows),
                    ),
                )

        for calibration_index, calibration in enumerate(compiled.get("calibrations", [])):
            media_index = next(
                index
                for index, media in enumerate(media_contracts)
                if media["channel"].lower() == calibration["channel"].lower()
            )
            roi_value = experiment_roi(calibration, media_index)
            calibration_sigma = max(
                safe_number(calibration.get("standardError"), 1.0),
                1e-6,
            )
            calibration_target = safe_number(calibration.get("observedRoi"))
            if calibration.get("route") == "prior":
                pm.Potential(
                    f"calibration_prior_{calibration_index:03d}",
                    pm.logp(
                        pm.Normal.dist(
                            mu=calibration_target,
                            sigma=calibration_sigma,
                        ),
                        roi_value,
                    ),
                )
            else:
                pm.Normal(
                    f"calibration_{calibration_index:03d}",
                    mu=roi_value,
                    sigma=calibration_sigma,
                    observed=calibration_target,
                )

    return model, initial_values, parameter_names, matrix, outcome


def extract_beta(idata: Any, parameter_count: int) -> np.ndarray:
    return np.stack(
        [
            np.asarray(idata.posterior[f"beta_{index:03d}"].values, dtype=float)
            for index in range(parameter_count)
        ],
        axis=-1,
    )


def posterior_skewness(values: np.ndarray) -> float:
    standard_deviation = float(np.std(values, ddof=1))
    if standard_deviation <= 1e-12:
        return 0.0
    centered = values - float(np.mean(values))
    return float(np.mean(centered ** 3) / standard_deviation ** 3)


def smoothed_relative_density(
    values: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    smoothed = np.convolve(
        counts.astype(float),
        np.asarray([1, 2, 3, 2, 1], dtype=float) / 9,
        mode="same",
    )
    return smoothed / max(float(np.max(smoothed)), 1.0)


def density_explanation(
    laplace_roi: np.ndarray,
    posterior_roi: np.ndarray,
) -> dict[str, Any]:
    combined = np.concatenate([laplace_roi, posterior_roi])
    positive = np.maximum(combined, 0)
    lower = max(float(np.quantile(positive, 0.0025)), 0.0)
    upper = max(float(np.quantile(positive, 0.9975)), lower + 1e-6)
    central = max(float(np.median(positive)), 1e-6)
    use_log = upper > max(10.0, central * 20)
    if use_log:
        transformed = np.log1p(positive)
        transformed_lower = max(float(np.quantile(transformed, 0.0025)), 0.0)
        transformed_upper = max(
            float(np.quantile(transformed, 0.9975)),
            transformed_lower + 1e-6,
        )
        edges = np.linspace(transformed_lower, transformed_upper, 57)
        laplace_values = np.log1p(np.maximum(laplace_roi, 0))
        posterior_values = np.log1p(np.maximum(posterior_roi, 0))
    else:
        edges = np.linspace(lower, upper, 57)
        laplace_values = laplace_roi
        posterior_values = posterior_roi
    centers = (edges[:-1] + edges[1:]) / 2
    return {
        "scale": "log1p" if use_log else "linear",
        "grid": [float(value) for value in centers],
        "laplace": [
            float(value) for value in smoothed_relative_density(laplace_values, edges)
        ],
        "posterior": [
            float(value) for value in smoothed_relative_density(posterior_values, edges)
        ],
    }


def approximation_assessment(
    location_shift_sd: float,
    uncertainty_ratio: float,
) -> str:
    center_differs = location_shift_sd >= 0.75
    shape_differs = uncertainty_ratio < 0.67 or uncertainty_ratio > 1.5
    if center_differs and shape_differs:
        return "center-shape"
    if center_differs:
        return "center"
    if shape_differs:
        return "shape"
    return "compatible"


def rank_traces(traces: np.ndarray) -> list[list[float]]:
    chain_count, draw_count = traces.shape
    flattened = traces.reshape(-1)
    order = np.argsort(flattened, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(flattened), dtype=float)
    ranks = (ranks + 0.5) / max(len(flattened), 1)
    ranked = ranks.reshape(chain_count, draw_count)
    sample_indexes = np.linspace(0, draw_count - 1, min(draw_count, 160), dtype=int)
    return [[float(value) for value in chain[sample_indexes]] for chain in ranked]


def trace_payload(traces: np.ndarray) -> list[list[float]]:
    draw_count = traces.shape[1]
    sample_indexes = np.linspace(0, draw_count - 1, min(draw_count, 180), dtype=int)
    return [[float(value) for value in chain[sample_indexes]] for chain in traces]


def diagnostic_values(idata: Any, variable_name: str) -> tuple[float, float, float, float]:
    try:
        rhat = float(np.asarray(az.rhat(idata, var_names=[variable_name])[variable_name]).max())
    except Exception:
        rhat = float("inf")
    try:
        bulk = float(
            np.asarray(az.ess(idata, var_names=[variable_name], method="bulk")[variable_name]).min()
        )
    except Exception:
        bulk = 0.0
    try:
        tail = float(
            np.asarray(az.ess(idata, var_names=[variable_name], method="tail")[variable_name]).min()
        )
    except Exception:
        tail = 0.0
    try:
        mcse = float(
            np.asarray(az.mcse(idata, var_names=[variable_name], method="mean")[variable_name]).max()
        )
    except Exception:
        mcse = float("inf")
    return rhat, bulk, tail, mcse


def posterior_predictive_draws(
    linear_draws: np.ndarray,
    sigma_draws: np.ndarray,
    likelihood: str,
    student_t_degrees_freedom: float,
    rng: np.random.Generator,
) -> np.ndarray:
    noise_scale = sigma_draws[:, None]
    if likelihood == "log-normal":
        return np.exp(
            np.clip(
                linear_draws + noise_scale * rng.normal(size=linear_draws.shape),
                -30,
                30,
            )
        )
    if likelihood == "student-t":
        return (
            linear_draws
            + noise_scale
            * rng.standard_t(
                max(student_t_degrees_freedom, 2.1),
                size=linear_draws.shape,
            )
        )
    return linear_draws + noise_scale * rng.normal(size=linear_draws.shape)


def posterior_decision_metrics(
    channels: list[dict[str, Any]],
    outcome: np.ndarray,
    predictive_low: np.ndarray,
    predictive_high: np.ndarray,
) -> dict[str, float | bool]:
    predictive_coverage = float(
        np.mean((outcome >= predictive_low) & (outcome <= predictive_high))
    )
    maximum_implausible_probability = max(
        (safe_number(channel.get("implausibleProbability")) for channel in channels),
        default=0.0,
    )
    maximum_relative_width = max(
        (safe_number(channel.get("relativeIntervalWidth")) for channel in channels),
        default=0.0,
    )
    return {
        "predictiveCoverage": predictive_coverage,
        "maxImplausibleProbability": maximum_implausible_probability,
        "maxRelativeRoiWidth": maximum_relative_width,
        "predictivePassed": predictive_coverage >= 0.8,
        "plausibilityPassed": maximum_implausible_probability <= 0.2,
        "precisionPassed": maximum_relative_width <= 3.0,
    }


def sampling_summary(
    job: Job,
    compiled: dict[str, Any],
    contract: dict[str, Any],
    fingerprint: str,
    idata: Any,
    runtime_seconds: float,
) -> dict[str, Any]:
    del job
    parameter_names = compiled["parameterNames"]
    beta = extract_beta(idata, len(parameter_names))
    sigma_draws = np.asarray(idata.posterior["sigma"].values, dtype=float)
    parameter_diagnostics = []
    all_rhat: list[float] = []
    all_bulk: list[float] = []
    all_tail: list[float] = []
    all_mcse_ratio: list[float] = []

    for index, label in enumerate(parameter_names):
        variable_name = f"beta_{index:03d}"
        traces = beta[:, :, index]
        posterior_mean, low, median, high, posterior_sd = posterior_summary(traces)
        rhat, bulk, tail, mcse = diagnostic_values(idata, variable_name)
        all_rhat.append(rhat)
        all_bulk.append(bulk)
        all_tail.append(tail)
        all_mcse_ratio.append(mcse / max(posterior_sd, 1e-12))
        parameter_diagnostics.append(
            {
                "name": variable_name,
                "label": label,
                "mean": posterior_mean,
                "standardDeviation": posterior_sd,
                "low": low,
                "median": median,
                "high": high,
                "rhat": rhat,
                "bulkEss": bulk,
                "tailEss": tail,
                "mcse": mcse,
                "traces": trace_payload(traces),
                "ranks": rank_traces(traces),
            }
        )

    response_parameter_labels = {
        "adstock_decay": "Geometric decay",
        "weibull_shape": "Weibull shape",
        "weibull_scale": "Weibull scale",
        "hill_shape": "Hill shape",
        "kernel_bandwidth": "Kernel bandwidth",
    }
    for media_index, media in enumerate(compiled.get("media", [])):
        channel_label = str(media.get("channel", f"Channel {media_index + 1}"))
        suffix = f"_{media_index:03d}"
        response_parameter_labels.update(
            {
                f"adstock_decay{suffix}": f"{channel_label} · geometric decay",
                f"weibull_shape{suffix}": f"{channel_label} · Weibull shape",
                f"weibull_scale{suffix}": f"{channel_label} · Weibull scale",
                f"hill_shape{suffix}": f"{channel_label} · Hill shape",
            }
        )
    for variable_name, label in response_parameter_labels.items():
        if variable_name not in idata.posterior:
            continue
        traces = np.asarray(idata.posterior[variable_name].values, dtype=float)
        if traces.ndim != 2:
            continue
        posterior_mean, low, median, high, posterior_sd = posterior_summary(traces)
        rhat, bulk, tail, mcse = diagnostic_values(idata, variable_name)
        all_rhat.append(rhat)
        all_bulk.append(bulk)
        all_tail.append(tail)
        all_mcse_ratio.append(mcse / max(posterior_sd, 1e-12))
        parameter_diagnostics.append(
            {
                "name": variable_name,
                "label": label,
                "mean": posterior_mean,
                "standardDeviation": posterior_sd,
                "low": low,
                "median": median,
                "high": high,
                "rhat": rhat,
                "bulkEss": bulk,
                "tailEss": tail,
                "mcse": mcse,
                "traces": trace_payload(traces),
                "ranks": rank_traces(traces),
            }
        )

    # The latent planning path is represented by smooth-basis deviation
    # weights. Keep those implementation details out of the main parameter
    # table, but require every component to pass the production diagnostics.
    if "planning_deviation_weight" in idata.posterior:
        planning_traces = np.asarray(
            idata.posterior["planning_deviation_weight"].values,
            dtype=float,
        )
        planning_sd = max(float(np.std(planning_traces, ddof=1)), 1e-12)
        planning_rhat, planning_bulk, planning_tail, planning_mcse = diagnostic_values(
            idata,
            "planning_deviation_weight",
        )
        all_rhat.append(planning_rhat)
        all_bulk.append(planning_bulk)
        all_tail.append(planning_tail)
        all_mcse_ratio.append(planning_mcse / planning_sd)

    sigma_rhat, sigma_bulk, sigma_tail, sigma_mcse = diagnostic_values(idata, "sigma")
    sigma_sd = max(float(np.std(sigma_draws, ddof=1)), 1e-12)
    all_rhat.append(sigma_rhat)
    all_bulk.append(sigma_bulk)
    all_tail.append(sigma_tail)
    all_mcse_ratio.append(sigma_mcse / sigma_sd)
    sigma_mean, sigma_low, sigma_median, sigma_high, _ = posterior_summary(
        sigma_draws
    )
    parameter_diagnostics.append(
        {
            "name": "sigma",
            "label": "Residual scale",
            "mean": sigma_mean,
            "standardDeviation": sigma_sd,
            "low": sigma_low,
            "median": sigma_median,
            "high": sigma_high,
            "rhat": sigma_rhat,
            "bulkEss": sigma_bulk,
            "tailEss": sigma_tail,
            "mcse": sigma_mcse,
            "traces": trace_payload(sigma_draws),
            "ranks": rank_traces(sigma_draws),
        }
    )

    sample_stats = idata.sample_stats
    divergences = int(np.asarray(sample_stats["diverging"].values).sum())
    if "tree_depth" in sample_stats:
        tree_depth = np.asarray(sample_stats["tree_depth"].values)
        tree_depth_hits = int((tree_depth >= int(contract["maxTreeDepth"])).sum())
    else:
        tree_depth_hits = 0
    try:
        energy = np.asarray(sample_stats["energy"].values, dtype=float)
        energy_variance = np.var(energy, axis=1)
        bfmi_values = np.mean(np.diff(energy, axis=1) ** 2, axis=1) / np.maximum(
            energy_variance,
            1e-12,
        )
        min_bfmi = float(np.nanmin(bfmi_values))
    except Exception:
        min_bfmi = 0.0

    max_rhat = max(all_rhat, default=float("inf"))
    min_bulk = min(all_bulk, default=0.0)
    min_tail = min(all_tail, default=0.0)
    max_mcse_ratio = max(all_mcse_ratio, default=float("inf"))
    validation_passed = bool(compiled.get("validationEligible"))

    gates: list[dict[str, Any]] = [
        {
            "id": "rhat",
            "label": "Chain mixing",
            "value": f"{max_rhat:.3f}",
            "threshold": "R-hat ≤ 1.01",
            "passed": max_rhat <= 1.01,
            "detail": "Independent chains should target the same posterior geometry.",
        },
        {
            "id": "ess",
            "label": "Effective samples",
            "value": f"{min(min_bulk, min_tail):.0f} minimum",
            "threshold": "Bulk and tail ESS ≥ 400",
            "passed": min_bulk >= 400 and min_tail >= 400,
            "detail": "Reliable means and interval tails require enough effectively independent draws.",
        },
        {
            "id": "divergences",
            "label": "Divergences",
            "value": str(divergences),
            "threshold": "0 post-warmup",
            "passed": divergences == 0,
            "detail": "Divergences can indicate biased exploration of difficult posterior geometry.",
        },
        {
            "id": "treedepth",
            "label": "Tree depth",
            "value": f"{tree_depth_hits} saturated",
            "threshold": "0 saturated transitions",
            "passed": tree_depth_hits == 0,
            "detail": "Repeated depth saturation suggests inefficient exploration.",
        },
        {
            "id": "bfmi",
            "label": "Energy mixing",
            "value": f"{min_bfmi:.2f}",
            "threshold": "BFMI > 0.30",
            "passed": min_bfmi > 0.3,
            "detail": "Energy transitions should explore the typical set efficiently.",
        },
        {
            "id": "mcse",
            "label": "Monte Carlo error",
            "value": f"{max_mcse_ratio * 100:.1f}% maximum",
            "threshold": "MCSE / posterior SD < 5%",
            "passed": max_mcse_ratio < 0.05,
            "detail": "Simulation noise should be small relative to posterior uncertainty.",
        },
        {
            "id": "validation",
            "label": "Promoted evidence",
            "value": "Passed" if validation_passed else "Review",
            "threshold": "All applicable validation gates pass",
            "passed": validation_passed,
            "detail": "Sampling hardens uncertainty; it does not replace model validation.",
        },
    ]

    flat_beta = beta.reshape(-1, beta.shape[-1])
    flat_sigma = sigma_draws.reshape(-1)
    channels = []
    outcome = np.asarray(compiled["outcome"], dtype=float)
    draw_total = flat_beta.shape[0]
    posterior_indexes = np.linspace(0, draw_total - 1, min(draw_total, 500), dtype=int)
    sampled_beta = flat_beta[posterior_indexes]
    sampled_sigma = flat_sigma[posterior_indexes]
    linear_draws = np.asarray(
        idata.posterior["linear_predictor"].values,
        dtype=float,
    ).reshape(draw_total, -1)[posterior_indexes]
    mean_draws = np.asarray(
        idata.posterior["mean_outcome"].values,
        dtype=float,
    ).reshape(draw_total, -1)[posterior_indexes]
    rng = np.random.default_rng(int(contract["seed"]) + 10_111)
    likelihood = compiled["likelihood"]
    observation_draws = posterior_predictive_draws(
        linear_draws,
        sampled_sigma,
        likelihood,
        safe_number(compiled.get("studentTDegreesFreedom"), 4),
        rng,
    )

    posterior_roi_by_channel: list[np.ndarray] = []
    contribution_draws_by_channel: list[np.ndarray] = []
    for media_index, media in enumerate(compiled["media"]):
        contribution = np.asarray(
            idata.posterior[f"media_contribution_{media_index:03d}"].values,
            dtype=float,
        ).reshape(draw_total, -1)[posterior_indexes]
        contribution_draws_by_channel.append(contribution)
        spend = max(safe_number(media.get("spend"), 1.0), 1.0)
        if likelihood == "log-normal":
            without_channel = np.exp(
                np.clip(
                    linear_draws - contribution + 0.5 * sampled_sigma[:, None] ** 2,
                    -30,
                    30,
                )
            )
            roi_draws = np.sum(mean_draws - without_channel, axis=1) / spend
        else:
            roi_draws = np.sum(contribution, axis=1) / spend
        posterior_roi_by_channel.append(roi_draws)

    roi_correlation = (
        np.corrcoef(np.stack(posterior_roi_by_channel, axis=0))
        if len(posterior_roi_by_channel) > 1
        else np.eye(1)
    )
    media_parameter_indexes = {
        int(index)
        for item in compiled["media"]
        for index in item["indexes"]
    }
    baseline_parameter_indexes = [
        index
        for index in range(sampled_beta.shape[1])
        if index not in media_parameter_indexes
    ]
    for media_index, media in enumerate(compiled["media"]):
        roi_draws = posterior_roi_by_channel[media_index]
        map_roi = safe_number(media.get("screeningRoi"))
        map_low = safe_number(media.get("screeningLow"))
        map_high = safe_number(media.get("screeningHigh"))
        posterior_mean, low, median, high, _ = posterior_summary(roi_draws)
        screening_sd = max((map_high - map_low) / 3.92, abs(map_roi) * 0.05, 1e-6)
        laplace_roi = np.maximum(
            0,
            rng.normal(map_roi, screening_sd, size=8_000),
        )
        map_width = max(map_high - map_low, 1e-9)
        posterior_width = max(high - low, 1e-9)
        posterior_sd = max(float(np.std(roi_draws, ddof=1)), 1e-9)
        location_shift_sd = abs(median - map_roi) / posterior_sd
        uncertainty_ratio = posterior_width / map_width
        overlap = max(0.0, min(map_high, high) - max(map_low, low)) / max(
            min(map_width, posterior_width),
            1e-9,
        )
        assessment = approximation_assessment(
            location_shift_sd,
            uncertainty_ratio,
        )
        tradeoff_candidates: list[dict[str, Any]] = []
        if len(compiled["media"]) > 1:
            correlations = np.asarray(roi_correlation[media_index], dtype=float)
            correlations[media_index] = 0
            for tradeoff_index, correlation in enumerate(correlations):
                if tradeoff_index == media_index:
                    continue
                tradeoff_candidates.append(
                    {
                        "label": compiled["media"][tradeoff_index]["channel"],
                        "kind": "channel",
                        "correlation": safe_number(correlation),
                    }
                )
        for parameter_index in baseline_parameter_indexes:
            correlation = safe_number(
                np.corrcoef(roi_draws, sampled_beta[:, parameter_index])[0, 1]
            )
            tradeoff_candidates.append(
                {
                    "label": compiled["parameterNames"][parameter_index],
                    "kind": "baseline",
                    "correlation": correlation,
                }
            )
        strongest_tradeoff = max(
            tradeoff_candidates,
            key=lambda candidate: abs(candidate["correlation"]),
            default=None,
        )
        tradeoff = (
            strongest_tradeoff
            if strongest_tradeoff
            and abs(strongest_tradeoff["correlation"]) >= 0.15
            else None
        )
        channels.append(
            {
                "channel": media["channel"],
                "mapRoi": map_roi,
                "mapLow": map_low,
                "mapHigh": map_high,
                "posteriorMean": posterior_mean,
                "posteriorMedian": median,
                "posteriorLow": low,
                "posteriorHigh": high,
                "posteriorSamples": [safe_number(value) for value in roi_draws],
                "difference": posterior_median_difference(median, map_roi),
                "uncertaintyRatio": uncertainty_ratio,
                "materialShift": assessment != "compatible",
                "assessment": assessment,
                "plausibleUpperRoi": safe_number(media.get("plausibleUpperRoi"), 10.0),
                "implausibleProbability": float(
                    np.mean(roi_draws > safe_number(media.get("plausibleUpperRoi"), 10.0))
                ),
                "relativeIntervalWidth": float(
                    (high - low) / max(abs(median), 1.0)
                ),
                "evidenceSource": (
                    media.get("priorEvidence", {}).get("source")
                    if media.get("priorEvidence")
                    else "regularizing"
                ),
                "explanation": {
                    "density": density_explanation(laplace_roi, roi_draws),
                    "locationShiftSd": location_shift_sd,
                    "intervalOverlap": min(overlap, 1.0),
                    "posteriorSkewness": posterior_skewness(roi_draws),
                    "nearZeroProbability": float(np.mean(roi_draws <= 0.1)),
                    "topTradeoff": tradeoff,
                },
            }
        )

    predictive_low, predictive_high = hdi_by_column(observation_draws)
    predictive_median = np.median(observation_draws, axis=0)
    mean_low, mean_high = hdi_by_column(mean_draws)
    mean_median = np.median(mean_draws, axis=0)
    decision_metrics = posterior_decision_metrics(
        channels,
        outcome,
        predictive_low,
        predictive_high,
    )
    predictive_coverage = safe_number(decision_metrics["predictiveCoverage"])
    maximum_implausible_probability = safe_number(
        decision_metrics["maxImplausibleProbability"]
    )
    maximum_relative_width = safe_number(decision_metrics["maxRelativeRoiWidth"])
    gates.extend(
        [
            {
                "id": "roi-plausibility",
                "label": "ROI plausibility",
                "value": f"{maximum_implausible_probability * 100:.0f}% maximum tail",
                "threshold": "≤ 20% beyond channel plausibility bound",
                "passed": bool(decision_metrics["plausibilityPassed"]),
                "detail": "A converged posterior still needs most ROI mass inside an evidence-informed business range.",
            },
            {
                "id": "roi-precision",
                "label": "ROI identification",
                "value": f"{maximum_relative_width:.1f}× maximum width",
                "threshold": "95% HDI width ≤ 3× max(|median|, 1)",
                "passed": bool(decision_metrics["precisionPassed"]),
                "detail": "Extremely wide channel intervals indicate that the MMM cannot separate attribution precisely enough for deployment.",
            },
            {
                "id": "predictive-coverage",
                "label": "Predictive calibration",
                "value": f"{predictive_coverage * 100:.0f}% covered",
                "threshold": "At least 80% in the 95% posterior predictive interval",
                "passed": bool(decision_metrics["predictivePassed"]),
                "detail": "Observed outcomes should be compatible with replicated observations from the fitted likelihood.",
            },
        ]
    )
    diagnostics = {
        "maxRhat": max_rhat,
        "minBulkEss": min_bulk,
        "minTailEss": min_tail,
        "divergences": divergences,
        "treeDepthHits": tree_depth_hits,
        "minBfmi": min_bfmi,
        "maxMcseRatio": max_mcse_ratio,
        "predictiveCoverage": predictive_coverage,
        "maxImplausibleProbability": maximum_implausible_probability,
        "maxRelativeRoiWidth": maximum_relative_width,
    }
    status = "ready" if all(gate["passed"] for gate in gates) else "review"
    decision_passed = all(
        gate["passed"]
        for gate in gates
        if gate["id"] in {"roi-plausibility", "roi-precision", "predictive-coverage"}
    )
    retry = retry_recommendation(
        contract,
        diagnostics,
        validation_passed,
        decision_passed,
    )
    return {
        "kind": "mcmc",
        "fingerprint": fingerprint,
        "promotedFingerprint": compiled["promotedFingerprint"],
        "promotedId": compiled["promotedId"],
        "cached": False,
        "engine": "PyMC 6.2 · NUTS",
        "inferenceContract": {
            "version": compiled["version"],
            "sharedPosterior": False,
            "sharedPriorContract": True,
            "mapEngine": "Analytic screening MAP / local Laplace",
            "samplingEngine": "PyMC NUTS",
            "sharedRoiTransform": True,
            "responseUncertainty": "Sampled continuous response parameters",
            "planningFactor": (
                "Probabilistic latent basis factor"
                if compiled.get("response", {}).get("planningIntensity")
                else "Not included"
            ),
        },
        "contract": contract,
        "status": status,
        "gates": gates,
        "diagnostics": diagnostics,
        "channels": channels,
        "predictive": {
            "dates": compiled["dates"],
            "actual": [float(value) for value in outcome],
            "map": [float(value) for value in compiled.get("screeningPredicted", [])],
            "median": [float(value) for value in predictive_median],
            "low": [float(value) for value in predictive_low],
            "high": [float(value) for value in predictive_high],
            "meanMedian": [float(value) for value in mean_median],
            "meanLow": [float(value) for value in mean_low],
            "meanHigh": [float(value) for value in mean_high],
        },
        "parameters": parameter_diagnostics,
        "retryRecommendation": retry,
        "artifact": {
            "id": fingerprint,
            "retainedDraws": int(contract["chains"] * contract["draws"]),
            "stored": True,
        },
        "runtimeSeconds": runtime_seconds,
        "runAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def posterior_median_difference(median: float, map_roi: float) -> float:
    return median - map_roi


def retry_recommendation(
    contract: dict[str, Any],
    diagnostics: dict[str, Any],
    validation_passed: bool,
    decision_passed: bool,
) -> dict[str, Any] | None:
    next_contract = dict(contract)
    next_contract["preset"] = "custom"
    if diagnostics["divergences"] > 0:
        next_contract["targetAccept"] = 0.95 if contract["targetAccept"] < 0.95 else 0.99
        next_contract["tune"] = max(int(contract["tune"]), 2000)
        return {
            "title": "Retry with smaller NUTS steps",
            "detail": "Divergences need a higher acceptance target and longer adaptation. Persistent divergences require reparameterization rather than more draws.",
            "contract": next_contract,
        }
    if diagnostics["maxRhat"] > 1.01:
        next_contract["tune"] = max(int(contract["tune"]), 2000)
        next_contract["draws"] = max(int(contract["draws"]), 2000)
        return {
            "title": "Retry with longer adaptation",
            "detail": "The chains have not mixed to the same distribution. Extend warmup and draws, then inspect rank plots if separation remains.",
            "contract": next_contract,
        }
    if diagnostics["minBulkEss"] < 400 or diagnostics["minTailEss"] < 400:
        next_contract["draws"] = min(max(int(contract["draws"]) * 2, 2000), 4000)
        return {
            "title": "Retry with more retained draws",
            "detail": "Chain mixing is acceptable but the effective sample size is low, so additional retained draws are appropriate.",
            "contract": next_contract,
        }
    if diagnostics["treeDepthHits"] > 0:
        next_contract["maxTreeDepth"] = min(max(int(contract["maxTreeDepth"]) + 2, 12), 16)
        return {
            "title": "Retry with a deeper trajectory cap",
            "detail": "Some NUTS trajectories reached the depth limit. Increase it cautiously after confirming there are no divergences.",
            "contract": next_contract,
        }
    if diagnostics["minBfmi"] <= 0.3:
        return {
            "title": "Review parameterization",
            "detail": "Low energy mixing is rarely fixed by simply adding iterations. Rescale or reparameterize the model before retrying.",
            "contract": next_contract,
        }
    if not validation_passed:
        return {
            "title": "Return to validation",
            "detail": "The sampler converged, but the promoted model has an applicable evidence gate that did not pass.",
            "contract": next_contract,
        }
    if not decision_passed:
        return {
            "title": "Return to calibration or specification search",
            "detail": "The sampler converged, but posterior ROI plausibility, identification, or predictive calibration is not decision-ready. More draws will not repair this evidence problem.",
            "contract": next_contract,
        }
    return None


def run_sampling(job: Job, payload: dict[str, Any]) -> None:
    started = time.perf_counter()
    fingerprint = payload["fingerprint"]
    compiled = payload["model"]
    contract = payload["contract"]
    artifact_dir = ARTIFACT_ROOT / fingerprint
    try:
        job.status = "running"
        update_job(
            job,
            stage="compiling",
            completed=0,
            total=int(contract["chains"]) * (int(contract["tune"]) + int(contract["draws"])),
            chain=0,
            detail="Compiling the promoted probabilistic specification.",
        )
        model, initial_values, parameter_names, _, _ = build_model(compiled)
        per_chain = int(contract["tune"]) + int(contract["draws"])
        total = int(contract["chains"]) * per_chain

        def callback(trace: Any = None, draw: Any = None, **_kwargs: Any) -> None:
            del trace
            if draw is None:
                return
            chain = int(getattr(draw, "chain", 0))
            draw_index = int(getattr(draw, "draw_idx", 0))
            tuning = bool(getattr(draw, "tuning", False))
            completed = min(chain * per_chain + draw_index + 1, total)
            update_job(
                job,
                stage="warmup" if tuning else "sampling",
                completed=completed,
                total=total,
                chain=chain + 1,
                detail=(
                    f"Chain {chain + 1}: adapting step size and mass matrix."
                    if tuning
                    else f"Chain {chain + 1}: retaining posterior draws."
                ),
            )

        with model:
            idata = pm.sample(
                draws=int(contract["draws"]),
                tune=int(contract["tune"]),
                chains=int(contract["chains"]),
                cores=1,
                random_seed=int(contract["seed"]),
                initvals=initial_values,
                init="jitter+adapt_full",
                callback=callback,
                progressbar=False,
                compute_convergence_checks=False,
                return_inferencedata=True,
                nuts={
                    "target_accept": float(contract["targetAccept"]),
                    "max_treedepth": int(contract["maxTreeDepth"]),
                },
            )

        update_job(
            job,
            stage="diagnostics",
            completed=total,
            total=total,
            detail="Computing rank-normalized convergence diagnostics.",
        )
        result = sampling_summary(
            job,
            compiled,
            contract,
            fingerprint,
            idata,
            time.perf_counter() - started,
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        beta = extract_beta(idata, len(parameter_names))
        sample_stats = idata.sample_stats
        artifact_payload: dict[str, Any] = {
            "beta": beta,
            "sigma": np.asarray(idata.posterior["sigma"].values),
            "linear_predictor": np.asarray(idata.posterior["linear_predictor"].values),
            "mean_outcome": np.asarray(idata.posterior["mean_outcome"].values),
            "diverging": np.asarray(sample_stats["diverging"].values),
            "tree_depth": (
                np.asarray(sample_stats["tree_depth"].values)
                if "tree_depth" in sample_stats
                else np.zeros((int(contract["chains"]), int(contract["draws"])))
            ),
            "energy": (
                np.asarray(sample_stats["energy"].values)
                if "energy" in sample_stats
                else np.zeros((int(contract["chains"]), int(contract["draws"])))
            ),
            "parameter_names": np.asarray(parameter_names),
        }
        for variable_name in (
            "adstock_decay",
            "weibull_shape",
            "weibull_scale",
            "hill_shape",
            "kernel_bandwidth",
            "planning_deviation_weight",
            "planning_factor",
        ):
            if variable_name in idata.posterior:
                artifact_payload[variable_name] = np.asarray(idata.posterior[variable_name].values)
        for variable_name in idata.posterior.data_vars:
            if variable_name.startswith(
                ("adstock_decay_", "weibull_shape_", "weibull_scale_", "hill_shape_")
            ):
                artifact_payload[variable_name] = np.asarray(idata.posterior[variable_name].values)
        for media_index in range(len(compiled.get("media", []))):
            variable_name = f"media_contribution_{media_index:03d}"
            artifact_payload[variable_name] = np.asarray(idata.posterior[variable_name].values)
        np.savez_compressed(artifact_dir / "posterior.npz", **artifact_payload)
        (artifact_dir / "summary.json").write_text(
            json.dumps(result, separators=(",", ":")),
            encoding="utf-8",
        )
        job.result = result
        job.status = "complete"
        update_job(
            job,
            stage="complete",
            completed=total,
            total=total,
            detail="Posterior sampling and convergence diagnostics complete.",
        )
    except Exception as error:
        job.status = "error"
        job.error = f"{type(error).__name__}: {error}"
        update_job(
            job,
            stage="error",
            detail="Sampling stopped before a valid posterior artifact was produced.",
        )
        traceback.print_exc()


def cached_result(fingerprint: str) -> dict[str, Any] | None:
    summary_path = ARTIFACT_ROOT / fingerprint / "summary.json"
    if not summary_path.exists():
        return None
    try:
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        result["cached"] = True
        return result
    except (OSError, json.JSONDecodeError):
        return None


class Handler(BaseHTTPRequestHandler):
    server_version = "FluxMCMC/1.0"

    def end_headers(self) -> None:
        origin = self.headers.get("Origin", "")
        if re.fullmatch(r"http://(?:localhost|127\.0\.0\.1):\d+", origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError("Sampling payload is empty or exceeds the local limit.")
        return json.loads(self.rfile.read(length))

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self.send_json(
                {
                    "ready": True,
                    "engine": "PyMC 6.2 · NUTS",
                    "contractVersion": SAMPLING_CONTRACT_VERSION,
                }
            )
            return
        match = re.fullmatch(r"/v1/sampling/([a-zA-Z0-9_-]+)", path)
        if match:
            job_id = match.group(1)
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                self.send_json({"error": "Sampling job not found."}, 404)
                return
            self.send_json(
                {
                    "id": job.id,
                    "status": job.status,
                    "progress": job.progress,
                    "result": job.result,
                    "error": job.error,
                }
            )
            return
        self.send_json({"error": "Not found."}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/v1/sampling":
            self.send_json({"error": "Not found."}, 404)
            return
        try:
            payload = self.read_json()
            fingerprint = str(payload.get("fingerprint", ""))
            if not re.fullmatch(r"[a-f0-9]{64}", fingerprint):
                raise ValueError("A valid sampling fingerprint is required.")
            if not payload.get("model") or not payload.get("contract"):
                raise ValueError("Compiled model and sampling contract are required.")
            validate_compiled_model(payload["model"])
            cached = cached_result(fingerprint)
            if cached:
                job = Job(
                    id=fingerprint,
                    fingerprint=fingerprint,
                    status="complete",
                    progress={
                        "stage": "complete",
                        "completed": 1,
                        "total": 1,
                        "chain": int(payload["contract"]["chains"]),
                        "detail": "Identical posterior restored from the artifact cache.",
                    },
                    result=cached,
                )
                with JOBS_LOCK:
                    JOBS[fingerprint] = job
                self.send_json({"id": job.id, "status": job.status, "cached": True})
                return
            with JOBS_LOCK:
                existing = JOBS.get(fingerprint)
                if existing and existing.status in {"queued", "running"}:
                    self.send_json({"id": existing.id, "status": existing.status})
                    return
                job_id = fingerprint or uuid.uuid4().hex
                job = Job(id=job_id, fingerprint=fingerprint)
                JOBS[job_id] = job
            thread = threading.Thread(
                target=run_sampling,
                args=(job, payload),
                daemon=True,
                name=f"flux-mcmc-{job.id[:8]}",
            )
            thread.start()
            self.send_json({"id": job.id, "status": job.status}, 202)
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, 400)

    def log_message(self, format_string: str, *args: Any) -> None:
        if os.environ.get("FLUX_MCMC_VERBOSE") == "1":
            super().log_message(format_string, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flux local MCMC service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8789)
    args = parser.parse_args()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"Flux MCMC service ready at http://{args.host}:{args.port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
