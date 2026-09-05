#!/usr/bin/env python3
"""Nested advertiser-grouped V8 selector training using SVI-only labels.

The script deliberately has no dependency on scikit-learn.  Every learner,
split, bootstrap, policy decision, and receipt is deterministic and serialized
for audit.  Candidate identifiers and simulator families are used only for
grouping and reporting; neither enters a model feature vector.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / ".flux-artifacts/svi-score-v8/development-dataset.json"
BASELINE_PATH = ROOT / ".flux-artifacts/svi-score-v8/v5d-svi-baseline-predictions.json"
PREDICTIONS_PATH = ROOT / ".flux-artifacts/svi-score-v8/cross-fitted-predictions.json"
MODEL_PATH = ROOT / ".flux-artifacts/svi-score-v8/frozen-development-selector.json"
ARTIFACT_PATH = ROOT / "research/svi_score_v8/artifacts/svi-score-v8-development.json"

SCENARIOS = (
    "budget-reduction",
    "fixed-budget-mix",
    "budget-growth",
    "economic-ceiling",
)
MEAN_RISK_WEIGHT = 0.65
P90_RISK_WEIGHT = 0.35
ABSTENTION_COST = 0.25
MINIMUM_PROMOTION_COVERAGE = 0.60


@dataclass(frozen=True)
class LearnerConfig:
    id: str
    family: str
    regularization: float
    epochs_or_rounds: int
    learning_rate: float
    complexity: int


@dataclass(frozen=True)
class PolicyConfig:
    uncertainty_penalty: float
    danger_penalty: float
    maximum_danger_probability: float
    maximum_relative_uncertainty: float
    minimum_margin: float

    @property
    def id(self) -> str:
        return (
            f"u{self.uncertainty_penalty:g}-d{self.danger_penalty:g}-"
            f"pd{self.maximum_danger_probability:g}-"
            f"ru{self.maximum_relative_uncertainty:g}-m{self.minimum_margin:g}"
        )


LEARNER_GRID = (
    LearnerConfig("linear-ridge-001", "linear", 0.01, 32, 0.035, 1),
    LearnerConfig("linear-ridge-003", "linear", 0.03, 32, 0.035, 1),
    LearnerConfig("additive-hinge-002", "additive", 0.02, 24, 0.025, 2),
    LearnerConfig("boosted-stumps-24", "boosted", 0.01, 24, 0.06, 3),
)

POLICY_GRID = tuple(
    PolicyConfig(uncertainty, danger, maximum_danger, maximum_relative, margin)
    for uncertainty in (0.0, 0.5)
    for danger in (0.0, 0.5)
    for maximum_danger in (0.55, 0.70)
    for maximum_relative in (0.75, 1.50)
    for margin in (0.0, 0.03)
)

HEAD_COUNT = 7
MEAN_HEAD = 0
P90_HEAD = 1
SCENARIO_SLICE = slice(2, 6)
DANGER_HEAD = 6


def sha256(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


def quantile(values: np.ndarray, probability: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, probability))


def sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -40, 40)
    return 1 / (1 + np.exp(-clipped))


def huber_gradient(error: np.ndarray, delta: float) -> np.ndarray:
    return np.clip(error, -delta, delta)


def softplus(value: np.ndarray | float) -> np.ndarray | float:
    return np.log1p(np.exp(-np.abs(value))) + np.maximum(value, 0)


def grouped_indexes(business_ids: np.ndarray, indexes: np.ndarray) -> list[np.ndarray]:
    return [
        indexes[business_ids[indexes] == business]
        for business in sorted(set(str(value) for value in business_ids[indexes]))
    ]


def bootstrap_multiplicity(
    business_ids: np.ndarray,
    indexes: np.ndarray,
    seed: int,
) -> np.ndarray:
    businesses = np.asarray(sorted(set(str(value) for value in business_ids[indexes])))
    random = np.random.default_rng(seed)
    sampled = random.choice(businesses, size=businesses.size, replace=True)
    counts = {business: int(np.sum(sampled == business)) for business in businesses}
    return np.asarray([counts.get(str(business_ids[index]), 0) for index in indexes], dtype=float)


def standardize_fit(features: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive = np.maximum(weights, 0)
    denominator = max(float(np.sum(positive)), 1.0)
    means = np.sum(features * positive[:, None], axis=0) / denominator
    variances = np.sum((features - means) ** 2 * positive[:, None], axis=0) / denominator
    return means, np.maximum(np.sqrt(variances), 1e-6)


def transform_basis(
    features: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    family: str,
) -> np.ndarray:
    standardized = np.clip((features - means) / scales, -8, 8)
    if family == "linear":
        basis = standardized
    elif family == "additive":
        basis = np.concatenate(
            (
                standardized,
                np.maximum(standardized + 1, 0),
                np.maximum(standardized, 0),
                np.maximum(standardized - 1, 0),
            ),
            axis=1,
        )
    else:
        basis = standardized
    return np.concatenate((np.ones((basis.shape[0], 1)), basis), axis=1)


def oracle_pair_gradient(
    predictions: np.ndarray,
    normalized_gaps: np.ndarray,
    business_ids: np.ndarray,
    indexes: np.ndarray,
    valid: np.ndarray,
    multiplicity: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    gradient = np.zeros(indexes.size, dtype=float)
    risk = MEAN_RISK_WEIGHT * predictions[:, MEAN_HEAD] + P90_RISK_WEIGHT * predictions[:, P90_HEAD]
    weighted_loss = 0.0
    pair_count = 0
    index_position = {int(index): position for position, index in enumerate(indexes)}
    groups = grouped_indexes(business_ids, indexes)
    active_businesses = max(sum(np.max(multiplicity[[index_position[int(i)] for i in group]]) > 0 for group in groups), 1)
    for group in groups:
        positions = np.asarray([index_position[int(index)] for index in group], dtype=int)
        positions = positions[valid[group] & (multiplicity[positions] > 0)]
        if positions.size < 2:
            continue
        absolute = indexes[positions]
        oracle_position = positions[int(np.argmin(normalized_gaps[absolute]))]
        business_weight = float(np.max(multiplicity[positions]))
        for challenger_position in positions:
            if challenger_position == oracle_position:
                continue
            gap = float(normalized_gaps[indexes[challenger_position]])
            if gap <= 1e-12:
                continue
            margin = risk[oracle_position] - risk[challenger_position]
            probability = float(sigmoid(np.asarray([margin]))[0])
            scale = business_weight * gap / active_businesses
            gradient[oracle_position] += scale * probability
            gradient[challenger_position] -= scale * probability
            weighted_loss += scale * float(softplus(margin))
            pair_count += 1
    return gradient, weighted_loss, pair_count


class LinearMultiHead:
    def __init__(self, config: LearnerConfig):
        self.config = config
        self.means: np.ndarray | None = None
        self.scales: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.training_receipt: dict[str, Any] = {}

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        dangerous: np.ndarray,
        normalized_gaps: np.ndarray,
        business_ids: np.ndarray,
        valid: np.ndarray,
        indexes: np.ndarray,
        seed: int,
        bootstrap: bool,
    ) -> None:
        multiplicity = (
            bootstrap_multiplicity(business_ids, indexes, seed)
            if bootstrap
            else np.ones(indexes.size, dtype=float)
        )
        self.means, self.scales = standardize_fit(features[indexes], multiplicity)
        basis = transform_basis(
            features[indexes], self.means, self.scales, self.config.family
        )
        width = basis.shape[1]
        weights = np.zeros((width, HEAD_COUNT), dtype=float)
        active = multiplicity > 0
        weights[0, :6] = np.average(targets[indexes][active], axis=0, weights=multiplicity[active])
        danger_rate = float(np.average(dangerous[indexes][active], weights=multiplicity[active]))
        weights[0, DANGER_HEAD] = math.log(
            max(danger_rate, 1e-4) / max(1 - danger_rate, 1e-4)
        )
        first = np.zeros_like(weights)
        second = np.zeros_like(weights)
        start = time.time()
        pair_count = 0
        final_pair_loss = 0.0
        for epoch in range(self.config.epochs_or_rounds):
            raw = basis @ weights
            prediction = raw.copy()
            prediction[:, :6] = np.maximum(prediction[:, :6], 0)
            gradient = np.zeros_like(prediction)
            row_weight = multiplicity / max(float(np.sum(multiplicity)), 1.0)
            gradient[:, MEAN_HEAD] = row_weight * huber_gradient(
                prediction[:, MEAN_HEAD] - targets[indexes, MEAN_HEAD], 1.5
            )
            p90_error = prediction[:, P90_HEAD] - targets[indexes, P90_HEAD]
            gradient[:, P90_HEAD] = 0.55 * row_weight * np.where(
                p90_error >= 0, 0.1, -0.9
            )
            gradient[:, SCENARIO_SLICE] = (
                0.15
                * row_weight[:, None]
                * huber_gradient(
                    prediction[:, SCENARIO_SLICE] - targets[indexes, SCENARIO_SLICE],
                    1.5,
                )
            )
            gradient[:, DANGER_HEAD] = (
                0.55
                * row_weight
                * (sigmoid(raw[:, DANGER_HEAD]) - dangerous[indexes])
            )
            rank_gradient, final_pair_loss, pair_count = oracle_pair_gradient(
                prediction,
                normalized_gaps,
                business_ids,
                indexes,
                valid,
                multiplicity,
            )
            gradient[:, MEAN_HEAD] += 0.75 * MEAN_RISK_WEIGHT * rank_gradient
            gradient[:, P90_HEAD] += 0.75 * P90_RISK_WEIGHT * rank_gradient
            parameter_gradient = basis.T @ gradient
            parameter_gradient[1:] += self.config.regularization * weights[1:]
            step = epoch + 1
            first = 0.9 * first + 0.1 * parameter_gradient
            second = 0.999 * second + 0.001 * parameter_gradient**2
            corrected_first = first / (1 - 0.9**step)
            corrected_second = second / (1 - 0.999**step)
            rate = self.config.learning_rate / math.sqrt(1 + epoch / 8)
            weights -= rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
            weights = np.clip(weights, -12, 12)
        self.weights = weights
        self.training_receipt = {
            "trainingRows": int(indexes.size),
            "bootstrapBusinesses": int(np.sum(multiplicity > 0) / 48),
            "pairComparisons": int(pair_count),
            "finalWeightedPairLoss": float(final_pair_loss),
            "seconds": time.time() - start,
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert self.means is not None and self.scales is not None and self.weights is not None
        raw = transform_basis(features, self.means, self.scales, self.config.family) @ self.weights
        output = raw.copy()
        output[:, :6] = np.maximum(output[:, :6], 0)
        output[:, DANGER_HEAD] = sigmoid(raw[:, DANGER_HEAD])
        return output

    def serialize(self) -> dict[str, Any]:
        assert self.means is not None and self.scales is not None and self.weights is not None
        return {
            "kind": self.config.family,
            "config": self.config.__dict__,
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "weights": self.weights.tolist(),
            "receipt": self.training_receipt,
        }


def fit_stump(
    features: np.ndarray,
    residual: np.ndarray,
    weights: np.ndarray,
    maximum_features: int = 32,
) -> dict[str, float | int]:
    centered = residual - np.average(residual, weights=np.maximum(weights, 1e-12))
    correlations = np.abs(features.T @ (centered * weights))
    feature_indexes = np.argsort(-correlations, kind="stable")[:maximum_features]
    best: dict[str, float | int] | None = None
    best_error = math.inf
    for feature in feature_indexes:
        values = features[:, feature]
        for threshold in (-1.0, 0.0, 1.0):
            left = values <= threshold
            right = ~left
            left_weight = float(np.sum(weights[left]))
            right_weight = float(np.sum(weights[right]))
            if left_weight <= 1e-9 or right_weight <= 1e-9:
                continue
            left_value = float(np.sum(residual[left] * weights[left]) / left_weight)
            right_value = float(np.sum(residual[right] * weights[right]) / right_weight)
            error = float(
                np.sum(weights[left] * (residual[left] - left_value) ** 2)
                + np.sum(weights[right] * (residual[right] - right_value) ** 2)
            )
            if error < best_error - 1e-12:
                best_error = error
                best = {
                    "feature": int(feature),
                    "threshold": float(threshold),
                    "left": float(np.clip(left_value, -2, 2)),
                    "right": float(np.clip(right_value, -2, 2)),
                }
    return best or {"feature": 0, "threshold": 0.0, "left": 0.0, "right": 0.0}


def stump_predict(features: np.ndarray, stump: dict[str, float | int]) -> np.ndarray:
    feature = int(stump["feature"])
    return np.where(
        features[:, feature] <= float(stump["threshold"]),
        float(stump["left"]),
        float(stump["right"]),
    )


class BoostedMultiHead:
    def __init__(self, config: LearnerConfig):
        self.config = config
        self.means: np.ndarray | None = None
        self.scales: np.ndarray | None = None
        self.initial: np.ndarray | None = None
        self.trees: list[list[dict[str, float | int]]] = [[] for _ in range(HEAD_COUNT)]
        self.training_receipt: dict[str, Any] = {}

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        dangerous: np.ndarray,
        normalized_gaps: np.ndarray,
        business_ids: np.ndarray,
        valid: np.ndarray,
        indexes: np.ndarray,
        seed: int,
        bootstrap: bool,
    ) -> None:
        multiplicity = (
            bootstrap_multiplicity(business_ids, indexes, seed)
            if bootstrap
            else np.ones(indexes.size, dtype=float)
        )
        self.means, self.scales = standardize_fit(features[indexes], multiplicity)
        x = np.clip((features[indexes] - self.means) / self.scales, -8, 8)
        active = multiplicity > 0
        initial = np.zeros(HEAD_COUNT, dtype=float)
        initial[:6] = np.average(targets[indexes][active], axis=0, weights=multiplicity[active])
        danger_rate = float(np.average(dangerous[indexes][active], weights=multiplicity[active]))
        initial[DANGER_HEAD] = math.log(max(danger_rate, 1e-4) / max(1 - danger_rate, 1e-4))
        self.initial = initial
        raw = np.tile(initial, (indexes.size, 1))
        start = time.time()
        pair_count = 0
        final_pair_loss = 0.0
        for _round in range(self.config.epochs_or_rounds):
            prediction = raw.copy()
            prediction[:, :6] = np.maximum(prediction[:, :6], 0)
            gradient = np.zeros_like(raw)
            normalizer = max(float(np.sum(multiplicity)), 1.0)
            row_weight = multiplicity / normalizer
            gradient[:, MEAN_HEAD] = row_weight * huber_gradient(
                prediction[:, MEAN_HEAD] - targets[indexes, MEAN_HEAD], 1.5
            )
            p90_error = prediction[:, P90_HEAD] - targets[indexes, P90_HEAD]
            gradient[:, P90_HEAD] = 0.55 * row_weight * np.where(
                p90_error >= 0, 0.1, -0.9
            )
            gradient[:, SCENARIO_SLICE] = (
                0.15
                * row_weight[:, None]
                * huber_gradient(
                    prediction[:, SCENARIO_SLICE] - targets[indexes, SCENARIO_SLICE],
                    1.5,
                )
            )
            gradient[:, DANGER_HEAD] = (
                0.55
                * row_weight
                * (sigmoid(raw[:, DANGER_HEAD]) - dangerous[indexes])
            )
            rank_gradient, final_pair_loss, pair_count = oracle_pair_gradient(
                prediction,
                normalized_gaps,
                business_ids,
                indexes,
                valid,
                multiplicity,
            )
            gradient[:, MEAN_HEAD] += 0.75 * MEAN_RISK_WEIGHT * rank_gradient
            gradient[:, P90_HEAD] += 0.75 * P90_RISK_WEIGHT * rank_gradient
            # Tree residuals operate on an average-row scale.  The gradients
            # above are normalized for a stable joint objective, so rescale by
            # active row mass before fitting each weak learner.
            residuals = -gradient * normalizer
            for head in range(HEAD_COUNT):
                stump = fit_stump(x, residuals[:, head], multiplicity)
                self.trees[head].append(stump)
                raw[:, head] += self.config.learning_rate * stump_predict(x, stump)
        self.training_receipt = {
            "trainingRows": int(indexes.size),
            "bootstrapBusinesses": int(np.sum(multiplicity > 0) / 48),
            "pairComparisons": int(pair_count),
            "finalWeightedPairLoss": float(final_pair_loss),
            "trees": int(sum(len(value) for value in self.trees)),
            "seconds": time.time() - start,
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert self.means is not None and self.scales is not None and self.initial is not None
        x = np.clip((features - self.means) / self.scales, -8, 8)
        raw = np.tile(self.initial, (features.shape[0], 1))
        for head, trees in enumerate(self.trees):
            for stump in trees:
                raw[:, head] += self.config.learning_rate * stump_predict(x, stump)
        output = raw.copy()
        output[:, :6] = np.maximum(output[:, :6], 0)
        output[:, DANGER_HEAD] = sigmoid(raw[:, DANGER_HEAD])
        return output

    def serialize(self) -> dict[str, Any]:
        assert self.means is not None and self.scales is not None and self.initial is not None
        return {
            "kind": self.config.family,
            "config": self.config.__dict__,
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "initial": self.initial.tolist(),
            "trees": self.trees,
            "receipt": self.training_receipt,
        }


def fit_model(
    config: LearnerConfig,
    features: np.ndarray,
    targets: np.ndarray,
    dangerous: np.ndarray,
    normalized_gaps: np.ndarray,
    business_ids: np.ndarray,
    valid: np.ndarray,
    indexes: np.ndarray,
    seed: int,
    bootstrap: bool,
) -> LinearMultiHead | BoostedMultiHead:
    model: LinearMultiHead | BoostedMultiHead
    model = BoostedMultiHead(config) if config.family == "boosted" else LinearMultiHead(config)
    model.fit(
        features,
        targets,
        dangerous,
        normalized_gaps,
        business_ids,
        valid,
        indexes,
        seed,
        bootstrap,
    )
    return model


def ensemble_predict(
    models: Iterable[LinearMultiHead | BoostedMultiHead],
    features: np.ndarray,
) -> dict[str, np.ndarray]:
    outputs = np.stack([model.predict(features) for model in models], axis=0)
    center = np.mean(outputs, axis=0)
    member_risk = (
        MEAN_RISK_WEIGHT * outputs[:, :, MEAN_HEAD]
        + P90_RISK_WEIGHT * outputs[:, :, P90_HEAD]
    )
    return {
        "heads": center,
        "risk": np.mean(member_risk, axis=0),
        "uncertainty": np.std(member_risk, axis=0, ddof=1) if outputs.shape[0] > 1 else np.zeros(outputs.shape[1]),
    }


def fit_ensemble(
    config: LearnerConfig,
    features: np.ndarray,
    targets: np.ndarray,
    dangerous: np.ndarray,
    normalized_gaps: np.ndarray,
    business_ids: np.ndarray,
    valid: np.ndarray,
    indexes: np.ndarray,
    seeds: Iterable[int],
) -> list[LinearMultiHead | BoostedMultiHead]:
    return [
        fit_model(
            config,
            features,
            targets,
            dangerous,
            normalized_gaps,
            business_ids,
            valid,
            indexes,
            seed,
            bootstrap=True,
        )
        for seed in seeds
    ]


def selection_receipts(
    indexes: np.ndarray,
    business_ids: np.ndarray,
    families: np.ndarray,
    candidate_ids: np.ndarray,
    valid: np.ndarray,
    excess_risk: np.ndarray,
    normalized_gaps: np.ndarray,
    actual_dangerous: np.ndarray,
    predicted: dict[str, np.ndarray],
    policy: PolicyConfig,
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    heads = predicted["heads"]
    risk = predicted["risk"]
    uncertainty = predicted["uncertainty"]
    danger = heads[:, DANGER_HEAD]
    adjusted = risk + policy.uncertainty_penalty * uncertainty + policy.danger_penalty * danger
    position = {int(index): offset for offset, index in enumerate(indexes)}
    for group in grouped_indexes(business_ids, indexes):
        active = group[valid[group]]
        if active.size == 0:
            raise RuntimeError(f"{business_ids[group[0]]} has no valid V8 candidate")
        active_positions = np.asarray([position[int(index)] for index in active], dtype=int)
        order = active_positions[np.argsort(adjusted[active_positions], kind="stable")]
        selected_position = int(order[0])
        selected_index = int(indexes[selected_position])
        second = float(adjusted[order[1]]) if order.size > 1 else math.inf
        relative_uncertainty = float(
            uncertainty[selected_position] / max(abs(risk[selected_position]) + 0.1, 0.1)
        )
        margin = second - float(adjusted[selected_position])
        reasons = []
        if danger[selected_position] > policy.maximum_danger_probability:
            reasons.append("danger-probability")
        if relative_uncertainty > policy.maximum_relative_uncertainty:
            reasons.append("ensemble-uncertainty")
        if margin < policy.minimum_margin:
            reasons.append("indistinguishable-finalists")
        abstained = bool(reasons)
        oracle_index = int(active[np.argmin(excess_risk[active])])
        oracle_position = position[oracle_index]
        indicator_bound = 0.0
        logistic_bound = 0.0
        for challenger_index in active:
            if int(challenger_index) == oracle_index:
                continue
            challenger_position = position[int(challenger_index)]
            gap = float(normalized_gaps[int(challenger_index)])
            difference = float(adjusted[oracle_position] - adjusted[challenger_position])
            if adjusted[challenger_position] <= adjusted[oracle_position] + 1e-12:
                indicator_bound += gap
            logistic_bound += gap * float(softplus(difference)) / math.log(2)
        receipts.append(
            {
                "businessId": str(business_ids[selected_index]),
                "family": str(families[selected_index]),
                "candidateId": str(candidate_ids[selected_index]),
                "abstained": abstained,
                "abstentionReasons": reasons,
                "actualExcessLoss": float(excess_risk[selected_index]),
                "normalizedExcessLoss": float(normalized_gaps[selected_index]),
                "dangerousFalseChampion": bool(actual_dangerous[selected_index]),
                "oracle": bool(excess_risk[selected_index] <= 1e-10),
                "predictedRisk": float(risk[selected_position]),
                "predictedDanger": float(danger[selected_position]),
                "uncertainty": float(uncertainty[selected_position]),
                "adjustedRisk": float(adjusted[selected_position]),
                "runnerUpMargin": margin,
                "policyLoss": ABSTENTION_COST if abstained else float(excess_risk[selected_index]),
                "theoremIndicatorBound": indicator_bound,
                "theoremLogisticBound": logistic_bound,
                "theoremSlack": float(normalized_gaps[selected_index]) - indicator_bound,
            }
        )
    return receipts


def metrics(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    promoted = [row for row in receipts if not row["abstained"]]
    losses = np.asarray([row["actualExcessLoss"] for row in promoted], dtype=float)
    policy_losses = np.asarray([row["policyLoss"] for row in receipts], dtype=float)
    by_family: dict[str, list[float]] = {}
    for row in receipts:
        by_family.setdefault(str(row["family"]), []).append(float(row["policyLoss"]))
    coverage = len(promoted) / max(len(receipts), 1)
    objective = (
        MEAN_RISK_WEIGHT * float(np.mean(policy_losses))
        + P90_RISK_WEIGHT * quantile(policy_losses, 0.9)
    )
    if coverage < MINIMUM_PROMOTION_COVERAGE:
        objective += 100 + 100 * (MINIMUM_PROMOTION_COVERAGE - coverage)
    return {
        "businesses": len(receipts),
        "promotionCoverage": coverage,
        "meanPromotedExcessLoss": float(np.mean(losses)) if losses.size else None,
        "medianPromotedExcessLoss": quantile(losses, 0.5) if losses.size else None,
        "p90PromotedExcessLoss": quantile(losses, 0.9) if losses.size else None,
        "p95PromotedExcessLoss": quantile(losses, 0.95) if losses.size else None,
        "meanPolicyLoss": float(np.mean(policy_losses)),
        "p90PolicyLoss": quantile(policy_losses, 0.9),
        "p95PolicyLoss": quantile(policy_losses, 0.95),
        "selectionObjective": objective,
        "dangerousFalseChampionShare": (
            float(np.mean([row["dangerousFalseChampion"] for row in promoted]))
            if promoted
            else None
        ),
        "oracleRecall": (
            float(np.mean([row["oracle"] for row in promoted])) if promoted else None
        ),
        "familyMeanPolicyLoss": {
            family: float(np.mean(values)) for family, values in sorted(by_family.items())
        },
        "theoremViolations": sum(row["theoremSlack"] > 1e-10 for row in receipts),
        "maximumTheoremSlack": max(0.0, *(float(row["theoremSlack"]) for row in receipts)),
        "meanIndicatorBound": float(np.mean([row["theoremIndicatorBound"] for row in receipts])),
        "meanLogisticBound": float(np.mean([row["theoremLogisticBound"] for row in receipts])),
    }


def baseline_receipts(
    rows: list[dict[str, Any]],
    baseline_predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prediction_by_key = {
        f"{row['businessId']}\0{row['candidateId']}": float(row["risk"])
        for row in baseline_predictions
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["businessId"]), []).append(row)
    output = []
    for business_id, business in sorted(grouped.items()):
        active = [row for row in business if row["valid"]]
        selected = min(
            active,
            key=lambda row: (
                prediction_by_key[f"{business_id}\0{row['candidateId']}"],
                str(row["candidateId"]),
            ),
        )
        loss = float(selected["excessEconomicRisk"])
        output.append(
            {
                "businessId": business_id,
                "family": selected["family"],
                "candidateId": selected["candidateId"],
                "abstained": False,
                "abstentionReasons": [],
                "actualExcessLoss": loss,
                "normalizedExcessLoss": float(selected["normalizedExcessEconomicRisk"]),
                "dangerousFalseChampion": bool(selected["dangerous"]),
                "oracle": loss <= 1e-10,
                "predictedRisk": prediction_by_key[f"{business_id}\0{selected['candidateId']}"],
                "predictedDanger": 0.0,
                "uncertainty": 0.0,
                "adjustedRisk": prediction_by_key[f"{business_id}\0{selected['candidateId']}"],
                "runnerUpMargin": 0.0,
                "policyLoss": loss,
                "theoremIndicatorBound": 0.0,
                "theoremLogisticBound": 0.0,
                "theoremSlack": 0.0,
            }
        )
    result = metrics(output)
    # The baseline did not optimize the V8 theorem surrogate, so those fields
    # are not comparable rather than spuriously reported as zero.
    for key in (
        "theoremViolations",
        "maximumTheoremSlack",
        "meanIndicatorBound",
        "meanLogisticBound",
    ):
        result[key] = None
    return output


def prediction_arrays(
    size: int,
) -> dict[str, np.ndarray]:
    return {
        "heads": np.full((size, HEAD_COUNT), np.nan, dtype=float),
        "risk": np.full(size, np.nan, dtype=float),
        "uncertainty": np.full(size, np.nan, dtype=float),
    }


def assign_predictions(
    destination: dict[str, np.ndarray],
    indexes: np.ndarray,
    source: dict[str, np.ndarray],
) -> None:
    destination["heads"][indexes] = source["heads"]
    destination["risk"][indexes] = source["risk"]
    destination["uncertainty"][indexes] = source["uncertainty"]


def main() -> None:
    started = time.time()
    dataset_source = DATASET_PATH.read_bytes()
    baseline_source = BASELINE_PATH.read_bytes()
    dataset = json.loads(dataset_source)
    baseline = json.loads(baseline_source)
    if (
        dataset.get("activation") != "development-only"
        or dataset.get("provenance", {}).get("auditAccessed")
        or dataset.get("provenance", {}).get("nutsUsed")
        or baseline.get("provenance", {}).get("auditAccessed")
        or baseline.get("provenance", {}).get("nutsUsed")
    ):
        raise RuntimeError("V8 training requires development-only SVI artifacts")
    rows: list[dict[str, Any]] = dataset["rows"]
    if len(rows) != 20_160 or len(dataset["featureNames"]) != 136:
        raise RuntimeError("V8 frozen 420 x 48 x 136 development contract changed")
    forbidden = tuple(dataset["contract"]["tokens"]["forbidden"])
    lowered = [str(name).lower() for name in dataset["featureNames"]]
    if any(any(item in name for item in forbidden) for name in lowered):
        raise RuntimeError("V8 token registry contains a forbidden truth or identity token")

    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    targets = np.asarray(
        [
            [
                row["meanExcessLoss"],
                row["p90ExcessLoss"],
                *(row["scenarioExcessLoss"][scenario] for scenario in SCENARIOS),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    dangerous = np.asarray([row["dangerous"] for row in rows], dtype=float)
    normalized_gaps = np.asarray(
        [row["normalizedExcessEconomicRisk"] for row in rows], dtype=float
    )
    excess_risk = np.asarray([row["excessEconomicRisk"] for row in rows], dtype=float)
    business_ids = np.asarray([row["businessId"] for row in rows], dtype=object)
    candidate_ids = np.asarray([row["candidateId"] for row in rows], dtype=object)
    families = np.asarray([row["family"] for row in rows], dtype=object)
    folds = np.asarray([row["fold"] for row in rows], dtype=int)
    valid = np.asarray([row["valid"] for row in rows], dtype=bool)
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
        raise RuntimeError("V8 features or targets are non-finite")

    outer_predictions = prediction_arrays(len(rows))
    outer_receipts: list[dict[str, Any]] = []
    all_selections: list[dict[str, Any]] = []
    selected_combinations: list[tuple[LearnerConfig, PolicyConfig]] = []
    for outer_fold in sorted(set(int(value) for value in folds)):
        outer_train = np.flatnonzero((folds != outer_fold) & valid)
        outer_assessment = np.flatnonzero((folds == outer_fold) & valid)
        inner_folds = sorted(set(int(value) for value in folds[outer_train]))
        combination_receipts: list[dict[str, Any]] = []
        best: tuple[float, int, str, LearnerConfig, PolicyConfig] | None = None
        for config in LEARNER_GRID:
            inner_predictions = prediction_arrays(len(rows))
            model_seconds = 0.0
            for inner_fold in inner_folds:
                inner_train = outer_train[folds[outer_train] != inner_fold]
                inner_assessment = outer_train[folds[outer_train] == inner_fold]
                models = fit_ensemble(
                    config,
                    features,
                    targets,
                    dangerous,
                    normalized_gaps,
                    business_ids,
                    valid,
                    inner_train,
                    (
                        80_000 + outer_fold * 1_000 + inner_fold * 20,
                        80_001 + outer_fold * 1_000 + inner_fold * 20,
                    ),
                )
                model_seconds += sum(float(model.training_receipt["seconds"]) for model in models)
                assign_predictions(
                    inner_predictions,
                    inner_assessment,
                    ensemble_predict(models, features[inner_assessment]),
                )
            for policy in POLICY_GRID:
                selections = selection_receipts(
                    outer_train,
                    business_ids,
                    families,
                    candidate_ids,
                    valid,
                    excess_risk,
                    normalized_gaps,
                    dangerous.astype(bool),
                    {
                        "heads": inner_predictions["heads"][outer_train],
                        "risk": inner_predictions["risk"][outer_train],
                        "uncertainty": inner_predictions["uncertainty"][outer_train],
                    },
                    policy,
                )
                receipt_metrics = metrics(selections)
                receipt = {
                    "learner": config.id,
                    "policy": policy.id,
                    "metrics": receipt_metrics,
                    "modelTrainingSeconds": model_seconds,
                }
                combination_receipts.append(receipt)
                key = (
                    float(receipt_metrics["selectionObjective"]),
                    config.complexity,
                    f"{config.id}/{policy.id}",
                    config,
                    policy,
                )
                if best is None or key[:3] < best[:3]:
                    best = key
        assert best is not None
        selected_config, selected_policy = best[3], best[4]
        selected_combinations.append((selected_config, selected_policy))
        final_models = fit_ensemble(
            selected_config,
            features,
            targets,
            dangerous,
            normalized_gaps,
            business_ids,
            valid,
            outer_train,
            tuple(90_000 + outer_fold * 100 + seed for seed in range(4)),
        )
        prediction = ensemble_predict(final_models, features[outer_assessment])
        assign_predictions(outer_predictions, outer_assessment, prediction)
        selections = selection_receipts(
            outer_assessment,
            business_ids,
            families,
            candidate_ids,
            valid,
            excess_risk,
            normalized_gaps,
            dangerous.astype(bool),
            prediction,
            selected_policy,
        )
        all_selections.extend(selections)
        outer_receipts.append(
            {
                "fold": outer_fold,
                "trainingBusinesses": len(set(str(value) for value in business_ids[outer_train])),
                "assessmentBusinesses": len(set(str(value) for value in business_ids[outer_assessment])),
                "selectedLearner": selected_config.id,
                "selectedPolicy": selected_policy.id,
                "innerWinnerMetrics": best[0],
                "assessmentMetrics": metrics(selections),
                "candidateConfigurations": combination_receipts,
                "ensembleReceipts": [model.training_receipt for model in final_models],
            }
        )
        print(
            json.dumps(
                {
                    "outerFold": outer_fold,
                    "learner": selected_config.id,
                    "policy": selected_policy.id,
                    "metrics": outer_receipts[-1]["assessmentMetrics"],
                }
            ),
            flush=True,
        )

    valid_indexes = np.flatnonzero(valid)
    if np.any(~np.isfinite(outer_predictions["risk"][valid_indexes])):
        raise RuntimeError("V8 did not produce one cross-fitted prediction per valid row")
    development_metrics = metrics(all_selections)
    baseline_selection = baseline_receipts(rows, baseline["predictions"])
    baseline_metrics = metrics(baseline_selection)
    for key in (
        "theoremViolations",
        "maximumTheoremSlack",
        "meanIndicatorBound",
        "meanLogisticBound",
    ):
        baseline_metrics[key] = None

    # Select the frozen development configuration by frequency across outer
    # folds, then by average inner objective and declared complexity.
    combination_counts: dict[str, int] = {}
    combination_values: dict[str, tuple[LearnerConfig, PolicyConfig]] = {}
    for config, policy in selected_combinations:
        key = f"{config.id}/{policy.id}"
        combination_counts[key] = combination_counts.get(key, 0) + 1
        combination_values[key] = (config, policy)
    frozen_key = sorted(
        combination_counts,
        key=lambda key: (
            -combination_counts[key],
            combination_values[key][0].complexity,
            key,
        ),
    )[0]
    frozen_config, frozen_policy = combination_values[frozen_key]
    final_models = fit_ensemble(
        frozen_config,
        features,
        targets,
        dangerous,
        normalized_gaps,
        business_ids,
        valid,
        valid_indexes,
        tuple(100_000 + seed for seed in range(8)),
    )
    frozen_model = {
        "artifactId": "flux-svi-score-v8-frozen-development-selector-v1",
        "version": dataset["version"],
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "activation": "development-only",
        "provenance": {
            "datasetSha256": sha256(dataset_source),
            "auditAccessed": False,
            "nutsUsed": False,
        },
        "featureNames": dataset["featureNames"],
        "learner": frozen_config.__dict__,
        "policy": frozen_policy.__dict__,
        "models": [model.serialize() for model in final_models],
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = MODEL_PATH.with_suffix(".json.tmp")
    temporary_model.write_text(json.dumps(frozen_model, separators=(",", ":")))
    temporary_model.replace(MODEL_PATH)

    prediction_rows = []
    for index in valid_indexes:
        heads = outer_predictions["heads"][index]
        prediction_rows.append(
            {
                "businessId": str(business_ids[index]),
                "candidateId": str(candidate_ids[index]),
                "fold": int(folds[index]),
                "predictedMean": float(heads[MEAN_HEAD]),
                "predictedP90": float(heads[P90_HEAD]),
                "predictedScenarios": {
                    scenario: float(heads[2 + position])
                    for position, scenario in enumerate(SCENARIOS)
                },
                "predictedDanger": float(heads[DANGER_HEAD]),
                "predictedRisk": float(outer_predictions["risk"][index]),
                "uncertainty": float(outer_predictions["uncertainty"][index]),
            }
        )
    predictions_artifact = {
        "artifactId": "flux-svi-score-v8-cross-fitted-predictions-v1",
        "version": dataset["version"],
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "activation": "development-only",
        "provenance": {
            "datasetSha256": sha256(dataset_source),
            "auditAccessed": False,
            "nutsUsed": False,
        },
        "predictions": prediction_rows,
        "selections": all_selections,
    }
    temporary_predictions = PREDICTIONS_PATH.with_suffix(".json.tmp")
    temporary_predictions.write_text(json.dumps(predictions_artifact, separators=(",", ":")))
    temporary_predictions.replace(PREDICTIONS_PATH)

    checks = {
        "meanPromotedExcessLossImproved": (
            development_metrics["meanPromotedExcessLoss"] is not None
            and development_metrics["meanPromotedExcessLoss"]
            < baseline_metrics["meanPromotedExcessLoss"]
        ),
        "p90PolicyLossImproved": development_metrics["p90PolicyLoss"] < baseline_metrics["p90PolicyLoss"],
        "p95PolicyLossNonInferior": development_metrics["p95PolicyLoss"] <= 1.02 * baseline_metrics["p95PolicyLoss"],
        "dangerousFalseChampionImproved": (
            development_metrics["dangerousFalseChampionShare"] is not None
            and development_metrics["dangerousFalseChampionShare"]
            < baseline_metrics["dangerousFalseChampionShare"]
        ),
        "minimumPromotionCoverage": development_metrics["promotionCoverage"] >= MINIMUM_PROMOTION_COVERAGE,
        "theoremVerified": development_metrics["theoremViolations"] == 0,
        "auditRemainedSealed": True,
        "nutsExcluded": True,
    }
    artifact = {
        "artifactId": "flux-svi-score-v8-selection-aware-development-v1",
        "version": dataset["version"],
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "activation": "development-only",
        "stage": "nested-grouped-development-complete",
        "contract": dataset["contract"],
        "provenance": {
            "datasetSha256": sha256(dataset_source),
            "baselineSha256": sha256(baseline_source),
            "predictionsSha256": sha256(PREDICTIONS_PATH.read_bytes()),
            "frozenModelSha256": sha256(MODEL_PATH.read_bytes()),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "auditAccessed": False,
            "nutsUsed": False,
        },
        "nestedCrossValidation": {
            "outerFolds": 5,
            "innerFoldsPerOuter": 4,
            "learnerGrid": [config.__dict__ for config in LEARNER_GRID],
            "policyGridSize": len(POLICY_GRID),
            "outerReceipts": outer_receipts,
        },
        "riskPreference": {
            "mean": MEAN_RISK_WEIGHT,
            "p90": P90_RISK_WEIGHT,
            "subjectiveBusinessPreference": True,
        },
        "abstention": {
            "cost": ABSTENTION_COST,
            "minimumPromotionCoverage": MINIMUM_PROMOTION_COVERAGE,
            "costIsSubjective": True,
        },
        "development": development_metrics,
        "v5dSviComparatorOnV8Targets": baseline_metrics,
        "checks": checks,
        "conclusion": {
            "allDevelopmentChecksPassed": all(checks.values()),
            "sealedAuditMayOpen": all(checks.values()),
            "sealedAuditAccessed": False,
            "productionActivationPermitted": False,
            "sotaClaimPermitted": False,
        },
        "frozenSelector": {
            "learner": frozen_config.id,
            "policy": frozen_policy.id,
            "outerSelectionFrequency": combination_counts,
            "path": str(MODEL_PATH.relative_to(ROOT)),
        },
        "runtimeSeconds": time.time() - started,
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_artifact = ARTIFACT_PATH.with_suffix(".json.tmp")
    temporary_artifact.write_text(json.dumps(artifact, indent=2))
    temporary_artifact.replace(ARTIFACT_PATH)
    print(
        json.dumps(
            {
                "artifactPath": str(ARTIFACT_PATH),
                "frozenModelPath": str(MODEL_PATH),
                "v8": development_metrics,
                "v5dSvi": baseline_metrics,
                "checks": checks,
                "conclusion": artifact["conclusion"],
                "runtimeSeconds": artifact["runtimeSeconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
