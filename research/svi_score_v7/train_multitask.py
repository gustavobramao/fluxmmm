#!/usr/bin/env python3
"""Deterministic NumPy multi-task V7 economic surrogate."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / ".flux-artifacts/svi-score-v7/dataset.json"
PREDICTIONS_PATH = ROOT / ".flux-artifacts/svi-score-v7/predictions.json"


def softplus(value: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(value))) + np.maximum(value, 0)


def sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -40, 40)
    return 1 / (1 + np.exp(-clipped))


def silu(value: np.ndarray) -> np.ndarray:
    return value * sigmoid(value)


def silu_derivative(value: np.ndarray) -> np.ndarray:
    probability = sigmoid(value)
    return probability + value * probability * (1 - probability)


def inverse_softplus(value: float) -> float:
    safe = max(float(value), 1e-4)
    return safe if safe > 20 else math.log(math.expm1(safe))


def initialize(input_width: int, seed: int, targets: np.ndarray) -> dict[str, np.ndarray]:
    random = np.random.default_rng(seed)
    hidden_one = 48
    hidden_two = 24
    parameters = {
        "w1": random.normal(0, math.sqrt(2 / input_width), (input_width, hidden_one)),
        "b1": np.zeros(hidden_one),
        "w2": random.normal(0, math.sqrt(2 / hidden_one), (hidden_one, hidden_two)),
        "b2": np.zeros(hidden_two),
        "wo": random.normal(0, math.sqrt(1 / hidden_two), (hidden_two, 4)),
        "bo": np.zeros(4),
    }
    means = np.mean(targets, axis=0)
    economic_p90 = np.quantile(targets[:, 0], 0.9)
    parameters["bo"][0] = inverse_softplus(means[0])
    parameters["bo"][1] = inverse_softplus(max(economic_p90 - means[0], 0.1))
    parameters["bo"][2] = inverse_softplus(means[1])
    parameters["bo"][3] = inverse_softplus(means[2])
    return {name: value.astype(np.float64) for name, value in parameters.items()}


def forward(
    features: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    first_linear = features @ parameters["w1"] + parameters["b1"]
    first = silu(first_linear)
    second_linear = first @ parameters["w2"] + parameters["b2"]
    second = silu(second_linear)
    raw = second @ parameters["wo"] + parameters["bo"]
    mean_loss = softplus(raw[:, 0])
    p90_loss = mean_loss + softplus(raw[:, 1])
    roi_error = softplus(raw[:, 2])
    contribution_error = softplus(raw[:, 3])
    output = np.column_stack((mean_loss, p90_loss, roi_error, contribution_error))
    return output, (features, first_linear, first, second_linear, second, raw)


def backward(
    output_gradient: np.ndarray,
    cache: tuple[np.ndarray, ...],
    parameters: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    features, first_linear, first, second_linear, second, raw = cache
    raw_gradient = np.zeros_like(raw)
    raw_gradient[:, 0] = (
        output_gradient[:, 0] + output_gradient[:, 1]
    ) * sigmoid(raw[:, 0])
    raw_gradient[:, 1] = output_gradient[:, 1] * sigmoid(raw[:, 1])
    raw_gradient[:, 2] = output_gradient[:, 2] * sigmoid(raw[:, 2])
    raw_gradient[:, 3] = output_gradient[:, 3] * sigmoid(raw[:, 3])
    gradients: dict[str, np.ndarray] = {}
    gradients["wo"] = second.T @ raw_gradient
    gradients["bo"] = np.sum(raw_gradient, axis=0)
    second_gradient = raw_gradient @ parameters["wo"].T
    second_linear_gradient = second_gradient * silu_derivative(second_linear)
    gradients["w2"] = first.T @ second_linear_gradient
    gradients["b2"] = np.sum(second_linear_gradient, axis=0)
    first_gradient = second_linear_gradient @ parameters["w2"].T
    first_linear_gradient = first_gradient * silu_derivative(first_linear)
    gradients["w1"] = features.T @ first_linear_gradient
    gradients["b1"] = np.sum(first_linear_gradient, axis=0)
    return gradients


def adam_update(
    parameters: dict[str, np.ndarray],
    gradients: dict[str, np.ndarray],
    first_moment: dict[str, np.ndarray],
    second_moment: dict[str, np.ndarray],
    step: int,
    learning_rate: float,
    weight_decay: float,
) -> None:
    beta_one = 0.9
    beta_two = 0.999
    for name, parameter in parameters.items():
        gradient = gradients[name]
        if name.startswith("w"):
            gradient = gradient + weight_decay * parameter
        first_moment[name] = beta_one * first_moment[name] + (1 - beta_one) * gradient
        second_moment[name] = (
            beta_two * second_moment[name] + (1 - beta_two) * gradient * gradient
        )
        corrected_first = first_moment[name] / (1 - beta_one**step)
        corrected_second = second_moment[name] / (1 - beta_two**step)
        parameter -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)


def huber_derivative(error: np.ndarray, delta: float) -> np.ndarray:
    return np.clip(error, -delta, delta)


def training_row_weights(
    economic_target: np.ndarray,
    families: np.ndarray,
) -> np.ndarray:
    tail = economic_target >= np.quantile(economic_target, 0.9) - 1e-12
    family_means = {
        family: float(np.mean(economic_target[families == family]))
        for family in np.unique(families)
    }
    worst_family = sorted(family_means, key=lambda item: (-family_means[item], item))[0]
    return (
        np.ones(economic_target.shape[0])
        + 0.8 * tail.astype(float)
        + 0.7 * (families == worst_family).astype(float)
    )


def hard_negative_pairs(
    business_ids: np.ndarray,
    in_pool: np.ndarray,
    targets: np.ndarray,
    predicted_mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    oracle_indexes: list[int] = []
    challenger_indexes: list[int] = []
    gaps: list[float] = []
    violations: list[float] = []
    for business_id in np.unique(business_ids):
        indexes = np.flatnonzero((business_ids == business_id) & in_pool)
        if indexes.size < 2:
            continue
        oracle = indexes[np.argmin(targets[indexes])]
        candidates = indexes[indexes != oracle]
        candidate_gaps = np.maximum(0, targets[candidates] - targets[oracle])
        candidate_violations = (
            np.log1p(candidate_gaps)
            + predicted_mean[oracle]
            - predicted_mean[candidates]
        )
        selected_position = int(np.argmax(candidate_violations))
        oracle_indexes.append(int(oracle))
        challenger_indexes.append(int(candidates[selected_position]))
        gaps.append(float(candidate_gaps[selected_position]))
        violations.append(float(candidate_violations[selected_position]))
    return (
        np.asarray(oracle_indexes, dtype=int),
        np.asarray(challenger_indexes, dtype=int),
        np.asarray(gaps, dtype=float),
        np.asarray(violations, dtype=float),
    )


def train_model(
    features: np.ndarray,
    targets: np.ndarray,
    business_ids: np.ndarray,
    families: np.ndarray,
    in_pool: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, float]]:
    means = np.mean(features, axis=0)
    scales = np.std(features, axis=0, ddof=1)
    scales = np.maximum(scales, 1e-6)
    standardized = np.clip((features - means) / scales, -8, 8)
    parameters = initialize(features.shape[1], seed, targets)
    first_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    second_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    random = np.random.default_rng(seed + 17)
    row_weights = training_row_weights(targets[:, 0], families)
    step = 0
    batch_size = 512
    start = time.time()
    for epoch in range(24):
        order = random.permutation(standardized.shape[0])
        for offset in range(0, order.size, batch_size):
            indexes = order[offset : offset + batch_size]
            batch_features = standardized[indexes]
            batch_targets = targets[indexes]
            batch_weights = row_weights[indexes]
            normalizer = max(float(np.sum(batch_weights)), 1.0)
            output, cache = forward(batch_features, parameters)
            gradient = np.zeros_like(output)
            gradient[:, 0] = (
                batch_weights
                * huber_derivative(output[:, 0] - batch_targets[:, 0], 2.5)
                / normalizer
            )
            gradient[:, 1] = (
                0.5
                * batch_weights
                * np.where(
                    batch_targets[:, 0] > output[:, 1],
                    -0.9,
                    0.1,
                )
                / normalizer
            )
            gradient[:, 2] = (
                0.15
                * batch_weights
                * huber_derivative(output[:, 2] - batch_targets[:, 1], 0.6)
                / normalizer
            )
            gradient[:, 3] = (
                0.15
                * batch_weights
                * huber_derivative(output[:, 3] - batch_targets[:, 2], 0.6)
                / normalizer
            )
            gradients = backward(gradient, cache, parameters)
            step += 1
            adam_update(
                parameters,
                gradients,
                first_moment,
                second_moment,
                step,
                0.003,
                0.001,
            )

        all_output, _ = forward(standardized, parameters)
        oracle, challenger, gaps, violations = hard_negative_pairs(
            business_ids,
            in_pool,
            targets[:, 0],
            all_output[:, 0],
        )
        if oracle.size:
            pair_features = np.concatenate((standardized[oracle], standardized[challenger]))
            pair_output, pair_cache = forward(pair_features, parameters)
            weights = np.maximum(gaps, 0.05)
            weights /= max(float(np.sum(weights)), 1e-12)
            derivative = 0.45 * weights * sigmoid(violations)
            pair_gradient = np.zeros_like(pair_output)
            pair_gradient[: oracle.size, 0] = derivative
            pair_gradient[oracle.size :, 0] = -derivative
            gradients = backward(pair_gradient, pair_cache, parameters)
            step += 1
            adam_update(
                parameters,
                gradients,
                first_moment,
                second_moment,
                step,
                0.003,
                0.001,
            )
    final_output, _ = forward(standardized, parameters)
    receipt = {
        "seconds": time.time() - start,
        "trainingMeanAbsoluteError": float(np.mean(np.abs(final_output[:, 0] - targets[:, 0]))),
        "trainingP90Coverage": float(np.mean(targets[:, 0] <= final_output[:, 1])),
        "hardNegativeBusinesses": int(np.unique(business_ids).size),
    }
    return parameters, means, scales, receipt


def parameter_hash(parameters: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(parameters):
        digest.update(name.encode())
        digest.update(parameters[name].tobytes())
    return digest.hexdigest()


def main() -> None:
    source = DATASET_PATH.read_bytes()
    dataset = json.loads(source)
    rows = dataset["rows"]
    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    economic = np.asarray([row["targetExcessLoss"] for row in rows], dtype=np.float64)
    roi = np.asarray([row["roiError"] for row in rows], dtype=np.float64)
    contribution = np.asarray([row["contributionError"] for row in rows], dtype=np.float64)
    targets = np.column_stack((economic, roi, contribution))
    business_ids = np.asarray([row["businessId"] for row in rows], dtype=object)
    candidate_ids = np.asarray([row["candidateId"] for row in rows], dtype=object)
    families = np.asarray([row["family"] for row in rows], dtype=object)
    folds = np.asarray([row["fold"] for row in rows], dtype=int)
    regions = np.asarray([row["region"] for row in rows], dtype=int)
    in_pool = np.asarray([row["inSelectionPool"] for row in rows], dtype=bool)
    predictions: list[dict[str, object]] = []
    model_receipts: list[dict[str, object]] = []
    for outer_fold in sorted(np.unique(folds)):
        for candidate_region in sorted(np.unique(regions)):
            training = (folds != outer_fold) & (regions != candidate_region)
            assessment = (folds == outer_fold) & (regions == candidate_region)
            parameters, means, scales, receipt = train_model(
                features[training],
                targets[training],
                business_ids[training],
                families[training],
                in_pool[training],
                seed=7000 + int(outer_fold) * 101 + int(candidate_region) * 17,
            )
            assessment_features = np.clip(
                (features[assessment] - means) / scales,
                -8,
                8,
            )
            output, _ = forward(assessment_features, parameters)
            indexes = np.flatnonzero(assessment)
            for position, row_index in enumerate(indexes):
                mean_loss = float(output[position, 0])
                p90_loss = float(output[position, 1])
                predictions.append({
                    "businessId": str(business_ids[row_index]),
                    "candidateId": str(candidate_ids[row_index]),
                    "mean": mean_loss,
                    "p90": p90_loss,
                    "tailWidth": p90_loss - mean_loss,
                    "risk": 0.65 * mean_loss + 0.35 * p90_loss,
                    "roiError": float(output[position, 2]),
                    "contributionError": float(output[position, 3]),
                    "fold": int(outer_fold),
                    "region": int(candidate_region),
                })
            model_receipts.append({
                "fold": int(outer_fold),
                "region": int(candidate_region),
                "trainingRows": int(np.sum(training)),
                "assessmentRows": int(np.sum(assessment)),
                "parameterHash": parameter_hash(parameters),
                **receipt,
            })
    if len(predictions) != len(rows):
        raise RuntimeError(f"V7 produced {len(predictions)} predictions for {len(rows)} rows")
    output = {
        "artifactId": "flux-svi-score-v7-cross-fitted-predictions-v1",
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "activation": "research-only",
        "provenance": {
            "datasetSha256": hashlib.sha256(source).hexdigest(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "freshValidationAccessed": False,
            "auditAccessed": False,
        },
        "architecture": dataset["contract"]["model"],
        "modelReceipts": model_receipts,
        "predictions": predictions,
    }
    temporary = PREDICTIONS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, separators=(",", ":")))
    temporary.replace(PREDICTIONS_PATH)
    print(json.dumps({
        "path": str(PREDICTIONS_PATH),
        "models": len(model_receipts),
        "predictions": len(predictions),
        "totalTrainingSeconds": sum(float(row["seconds"]) for row in model_receipts),
        "meanTrainingMAE": float(np.mean([row["trainingMeanAbsoluteError"] for row in model_receipts])),
        "meanTrainingP90Coverage": float(np.mean([row["trainingP90Coverage"] for row in model_receipts])),
        "freshValidationAccessed": False,
        "auditAccessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
