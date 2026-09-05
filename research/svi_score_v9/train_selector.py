#!/usr/bin/env python3
"""Nested, advertiser-grouped V9 DeepSets selector.

The implementation intentionally depends only on NumPy. Candidate order is
never a feature: a shared encoder is mean-pooled within each advertiser and
the decoder receives only permutation-equivariant candidate/context terms.
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
DATASET_PATH = ROOT / ".flux-artifacts/svi-score-v9/development-dataset.json"
V8_PREDICTIONS_PATH = ROOT / ".flux-artifacts/svi-score-v8/cross-fitted-predictions.json"
V8_DEVELOPMENT_PATH = ROOT / "research/svi_score_v8/artifacts/svi-score-v8-development.json"
PREDICTIONS_PATH = ROOT / ".flux-artifacts/svi-score-v9/cross-fitted-predictions.json"
MODEL_PATH = ROOT / ".flux-artifacts/svi-score-v9/development-selector.json"
ARTIFACT_PATH = ROOT / "research/svi_score_v9/artifacts/svi-score-v9-development.json"

SCENARIOS = (
    "budget-reduction",
    "fixed-budget-mix",
    "budget-growth",
    "economic-ceiling",
)
CANDIDATES_PER_BUSINESS = 48
MEAN_RISK_WEIGHT = 0.65
P90_RISK_WEIGHT = 0.35
OPERATIONAL_COVERAGE = 0.70
COVERAGE_LEVELS = tuple(float(value) for value in np.arange(0.60, 1.001, 0.05))

MEAN_HEAD = 0
MEDIAN_HEAD = 1
P90_HEAD = 2
P95_HEAD = 3
CVAR_HEAD = 4
SCENARIO_SLICE = slice(5, 9)
DANGER_HEAD = 9
HEAD_COUNT = 10


@dataclass(frozen=True)
class ModelConfig:
    id: str
    candidate_width: int
    decoder_width: int
    epochs: int
    learning_rate: float
    weight_decay: float
    temperature: float


@dataclass(frozen=True)
class PolicyConfig:
    danger_penalty: float
    uncertainty_penalty: float = 0.25

    @property
    def id(self) -> str:
        return f"danger-{self.danger_penalty:g}-uncertainty-{self.uncertainty_penalty:g}"


MODEL_GRID = (
    ModelConfig("deepsets-compact", 12, 12, 16, 0.0025, 0.001, 0.35),
    ModelConfig("deepsets-balanced", 20, 16, 20, 0.0020, 0.001, 0.35),
)
POLICY_GRID = tuple(PolicyConfig(value) for value in (0.0, 0.25, 0.50))


def sha256(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


def sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -40, 40)
    return 1 / (1 + np.exp(-clipped))


def softplus(value: np.ndarray | float) -> np.ndarray | float:
    return np.log1p(np.exp(-np.abs(value))) + np.maximum(value, 0)


def inverse_softplus(value: float) -> float:
    safe = max(float(value), 1e-4)
    return safe if safe > 20 else math.log(math.expm1(safe))


def silu(value: np.ndarray) -> np.ndarray:
    return value * sigmoid(value)


def silu_derivative(value: np.ndarray) -> np.ndarray:
    probability = sigmoid(value)
    return probability + value * probability * (1 - probability)


def quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability)) if values.size else 0.0


def cvar(values: np.ndarray, probability: float) -> float:
    if not values.size:
        return 0.0
    threshold = quantile(values, probability)
    tail = values[values >= threshold - 1e-12]
    return float(np.mean(tail)) if tail.size else float(np.max(values))


def huber_gradient(error: np.ndarray, delta: float) -> np.ndarray:
    return np.clip(error, -delta, delta)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponential = np.exp(np.clip(shifted, -60, 0))
    return exponential / np.maximum(np.sum(exponential, axis=-1, keepdims=True), 1e-12)


def output_from_raw(raw: np.ndarray) -> np.ndarray:
    output = np.zeros_like(raw)
    output[..., MEAN_HEAD] = softplus(raw[..., MEAN_HEAD])
    output[..., MEDIAN_HEAD] = softplus(raw[..., MEDIAN_HEAD])
    output[..., P90_HEAD] = output[..., MEDIAN_HEAD] + softplus(raw[..., P90_HEAD])
    output[..., P95_HEAD] = output[..., P90_HEAD] + softplus(raw[..., P95_HEAD])
    output[..., CVAR_HEAD] = output[..., P95_HEAD] + softplus(raw[..., CVAR_HEAD])
    output[..., SCENARIO_SLICE] = softplus(raw[..., SCENARIO_SLICE])
    output[..., DANGER_HEAD] = sigmoid(raw[..., DANGER_HEAD])
    return output


def raw_gradient_from_output_gradient(
    output_gradient: np.ndarray,
    raw: np.ndarray,
) -> np.ndarray:
    gradient = np.zeros_like(raw)
    gradient[..., MEAN_HEAD] = (
        output_gradient[..., MEAN_HEAD] * sigmoid(raw[..., MEAN_HEAD])
    )
    gradient[..., MEDIAN_HEAD] = (
        output_gradient[..., MEDIAN_HEAD]
        + output_gradient[..., P90_HEAD]
        + output_gradient[..., P95_HEAD]
        + output_gradient[..., CVAR_HEAD]
    ) * sigmoid(raw[..., MEDIAN_HEAD])
    gradient[..., P90_HEAD] = (
        output_gradient[..., P90_HEAD]
        + output_gradient[..., P95_HEAD]
        + output_gradient[..., CVAR_HEAD]
    ) * sigmoid(raw[..., P90_HEAD])
    gradient[..., P95_HEAD] = (
        output_gradient[..., P95_HEAD] + output_gradient[..., CVAR_HEAD]
    ) * sigmoid(raw[..., P95_HEAD])
    gradient[..., CVAR_HEAD] = (
        output_gradient[..., CVAR_HEAD] * sigmoid(raw[..., CVAR_HEAD])
    )
    gradient[..., SCENARIO_SLICE] = (
        output_gradient[..., SCENARIO_SLICE] * sigmoid(raw[..., SCENARIO_SLICE])
    )
    # The danger slot receives the BCE derivative with respect to its logit.
    gradient[..., DANGER_HEAD] = output_gradient[..., DANGER_HEAD]
    return gradient


class DeepSetSelector:
    def __init__(self, config: ModelConfig, seed: int):
        self.config = config
        self.seed = seed
        self.means: np.ndarray | None = None
        self.scales: np.ndarray | None = None
        self.parameters: dict[str, np.ndarray] = {}
        self.training_receipt: dict[str, Any] = {}

    def initialize(self, input_width: int, targets: np.ndarray, dangerous: np.ndarray) -> None:
        random = np.random.default_rng(self.seed)
        candidate_width = self.config.candidate_width
        decoder_width = self.config.decoder_width
        self.parameters = {
            "w1": random.normal(
                0, math.sqrt(2 / input_width), (input_width, candidate_width)
            ),
            "b1": np.zeros(candidate_width),
            "w2": random.normal(
                0,
                math.sqrt(2 / (4 * candidate_width)),
                (4 * candidate_width, decoder_width),
            ),
            "b2": np.zeros(decoder_width),
            "wo": random.normal(0, math.sqrt(1 / decoder_width), (decoder_width, HEAD_COUNT)),
            "bo": np.zeros(HEAD_COUNT),
        }
        target_center = np.mean(targets.reshape(-1, targets.shape[-1]), axis=0)
        self.parameters["bo"][MEAN_HEAD] = inverse_softplus(target_center[MEAN_HEAD])
        self.parameters["bo"][MEDIAN_HEAD] = inverse_softplus(target_center[MEDIAN_HEAD])
        self.parameters["bo"][P90_HEAD] = inverse_softplus(
            max(target_center[P90_HEAD] - target_center[MEDIAN_HEAD], 0.05)
        )
        self.parameters["bo"][P95_HEAD] = inverse_softplus(
            max(target_center[P95_HEAD] - target_center[P90_HEAD], 0.05)
        )
        self.parameters["bo"][CVAR_HEAD] = inverse_softplus(
            max(target_center[CVAR_HEAD] - target_center[P95_HEAD], 0.05)
        )
        for head in range(SCENARIO_SLICE.start, SCENARIO_SLICE.stop):
            self.parameters["bo"][head] = inverse_softplus(target_center[head])
        danger_rate = float(np.mean(dangerous))
        self.parameters["bo"][DANGER_HEAD] = math.log(
            max(danger_rate, 1e-4) / max(1 - danger_rate, 1e-4)
        )

    def forward(
        self,
        features: np.ndarray,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        parameters = self.parameters
        first_linear = features @ parameters["w1"] + parameters["b1"]
        candidate = silu(first_linear)
        context = np.mean(candidate, axis=1, keepdims=True)
        repeated = np.broadcast_to(context, candidate.shape)
        decoder_input = np.concatenate(
            (candidate, repeated, candidate - repeated, candidate * repeated),
            axis=2,
        )
        second_linear = decoder_input @ parameters["w2"] + parameters["b2"]
        second = silu(second_linear)
        raw = second @ parameters["wo"] + parameters["bo"]
        return output_from_raw(raw), (
            features,
            first_linear,
            candidate,
            context,
            decoder_input,
            second_linear,
            second,
            raw,
        )

    def backward(
        self,
        output_gradient: np.ndarray,
        cache: tuple[np.ndarray, ...],
    ) -> dict[str, np.ndarray]:
        (
            features,
            first_linear,
            candidate,
            context,
            decoder_input,
            second_linear,
            second,
            raw,
        ) = cache
        parameters = self.parameters
        raw_gradient = raw_gradient_from_output_gradient(output_gradient, raw)
        flat_second = second.reshape(-1, second.shape[-1])
        flat_raw_gradient = raw_gradient.reshape(-1, raw_gradient.shape[-1])
        gradients: dict[str, np.ndarray] = {
            "wo": flat_second.T @ flat_raw_gradient,
            "bo": np.sum(flat_raw_gradient, axis=0),
        }
        second_gradient = raw_gradient @ parameters["wo"].T
        second_linear_gradient = second_gradient * silu_derivative(second_linear)
        gradients["w2"] = (
            decoder_input.reshape(-1, decoder_input.shape[-1]).T
            @ second_linear_gradient.reshape(-1, second_linear_gradient.shape[-1])
        )
        gradients["b2"] = np.sum(second_linear_gradient, axis=(0, 1))
        decoder_gradient = second_linear_gradient @ parameters["w2"].T
        width = candidate.shape[-1]
        direct = decoder_gradient[..., :width]
        context_gradient = decoder_gradient[..., width : 2 * width]
        difference = decoder_gradient[..., 2 * width : 3 * width]
        interaction = decoder_gradient[..., 3 * width :]
        candidate_gradient = direct + difference + interaction * context
        context_uses = context_gradient - difference + interaction * candidate
        pooled_gradient = np.sum(context_uses, axis=1, keepdims=True)
        candidate_gradient += pooled_gradient / candidate.shape[1]
        first_gradient = candidate_gradient * silu_derivative(first_linear)
        gradients["w1"] = (
            features.reshape(-1, features.shape[-1]).T
            @ first_gradient.reshape(-1, first_gradient.shape[-1])
        )
        gradients["b1"] = np.sum(first_gradient, axis=(0, 1))
        return gradients

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        dangerous: np.ndarray,
        normalized_gaps: np.ndarray,
        valid: np.ndarray,
        business_indexes: np.ndarray,
        bootstrap: bool,
    ) -> None:
        flat = features[business_indexes].reshape(-1, features.shape[-1])
        self.means = np.mean(flat, axis=0)
        self.scales = np.maximum(np.std(flat, axis=0, ddof=1), 1e-6)
        standardized = np.clip((features - self.means) / self.scales, -8, 8)
        self.initialize(
            features.shape[-1],
            targets[business_indexes],
            dangerous[business_indexes],
        )
        first_moment = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        second_moment = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        random = np.random.default_rng(self.seed + 29)
        sampled = (
            random.choice(business_indexes, size=business_indexes.size, replace=True)
            if bootstrap
            else business_indexes.copy()
        )
        step = 0
        start = time.time()
        final_selection_loss = 0.0
        final_pair_loss = 0.0
        for epoch in range(self.config.epochs):
            order = sampled[random.permutation(sampled.size)]
            for offset in range(0, order.size, 32):
                batch = order[offset : offset + 32]
                output, cache = self.forward(standardized[batch])
                batch_targets = targets[batch]
                batch_dangerous = dangerous[batch]
                batch_gaps = normalized_gaps[batch]
                batch_valid = valid[batch]
                row_weight = batch_valid.astype(float) / max(float(np.sum(batch_valid)), 1.0)
                gradient = np.zeros_like(output)
                gradient[..., MEAN_HEAD] = row_weight * huber_gradient(
                    output[..., MEAN_HEAD] - batch_targets[..., MEAN_HEAD], 1.5
                )
                gradient[..., MEDIAN_HEAD] = 0.15 * row_weight * huber_gradient(
                    output[..., MEDIAN_HEAD] - batch_targets[..., MEDIAN_HEAD], 1.5
                )
                p90_error = output[..., P90_HEAD] - batch_targets[..., P90_HEAD]
                gradient[..., P90_HEAD] = 0.35 * row_weight * np.where(
                    p90_error >= 0, 0.1, -0.9
                )
                p95_error = output[..., P95_HEAD] - batch_targets[..., P95_HEAD]
                gradient[..., P95_HEAD] = 0.20 * row_weight * np.where(
                    p95_error >= 0, 0.05, -0.95
                )
                gradient[..., CVAR_HEAD] = 0.30 * row_weight * huber_gradient(
                    output[..., CVAR_HEAD] - batch_targets[..., CVAR_HEAD], 2.0
                )
                gradient[..., SCENARIO_SLICE] = (
                    0.12
                    * row_weight[..., None]
                    * huber_gradient(
                        output[..., SCENARIO_SLICE] - batch_targets[..., SCENARIO_SLICE],
                        1.5,
                    )
                )
                gradient[..., DANGER_HEAD] = (
                    0.45
                    * row_weight
                    * (output[..., DANGER_HEAD] - batch_dangerous)
                )

                risk = (
                    MEAN_RISK_WEIGHT * output[..., MEAN_HEAD]
                    + P90_RISK_WEIGHT * output[..., P90_HEAD]
                )
                final_selection_loss = 0.0
                final_pair_loss = 0.0
                for position in range(batch.size):
                    active = np.flatnonzero(batch_valid[position])
                    if active.size < 2:
                        continue
                    scores = risk[position, active]
                    gaps = batch_gaps[position, active]
                    probability = stable_softmax(-scores / self.config.temperature)
                    expected_gap = float(np.sum(probability * gaps))
                    direct = probability * (expected_gap - gaps) / self.config.temperature
                    direct /= max(batch.size, 1)
                    gradient[position, active, MEAN_HEAD] += MEAN_RISK_WEIGHT * direct
                    gradient[position, active, P90_HEAD] += P90_RISK_WEIGHT * direct
                    final_selection_loss += expected_gap / max(batch.size, 1)

                    oracle_local = int(np.argmin(gaps))
                    oracle = int(active[oracle_local])
                    for challenger in active:
                        if int(challenger) == oracle:
                            continue
                        gap = float(batch_gaps[position, challenger])
                        if gap <= 1e-12:
                            continue
                        margin = float(risk[position, oracle] - risk[position, challenger])
                        derivative = 0.25 * gap * float(sigmoid(np.asarray([margin]))[0])
                        derivative /= max(batch.size, 1)
                        gradient[position, oracle, MEAN_HEAD] += MEAN_RISK_WEIGHT * derivative
                        gradient[position, oracle, P90_HEAD] += P90_RISK_WEIGHT * derivative
                        gradient[position, challenger, MEAN_HEAD] -= MEAN_RISK_WEIGHT * derivative
                        gradient[position, challenger, P90_HEAD] -= P90_RISK_WEIGHT * derivative
                        final_pair_loss += (
                            0.25 * gap * float(softplus(margin)) / max(batch.size, 1)
                        )

                gradients = self.backward(gradient, cache)
                for name, value in gradients.items():
                    if name.startswith("w"):
                        gradients[name] = value + self.config.weight_decay * self.parameters[name]
                norm = math.sqrt(sum(float(np.sum(value * value)) for value in gradients.values()))
                if norm > 5:
                    gradients = {name: value * (5 / norm) for name, value in gradients.items()}
                step += 1
                self._adam_update(gradients, first_moment, second_moment, step)
        self.training_receipt = {
            "trainingBusinesses": int(business_indexes.size),
            "bootstrap": bootstrap,
            "epochs": self.config.epochs,
            "finalBatchDirectSelectionLoss": final_selection_loss,
            "finalBatchTheoremSurrogateLoss": final_pair_loss,
            "seconds": time.time() - start,
        }

    def _adam_update(
        self,
        gradients: dict[str, np.ndarray],
        first_moment: dict[str, np.ndarray],
        second_moment: dict[str, np.ndarray],
        step: int,
    ) -> None:
        for name, parameter in self.parameters.items():
            gradient = gradients[name]
            first_moment[name] = 0.9 * first_moment[name] + 0.1 * gradient
            second_moment[name] = 0.999 * second_moment[name] + 0.001 * gradient * gradient
            corrected_first = first_moment[name] / (1 - 0.9**step)
            corrected_second = second_moment[name] / (1 - 0.999**step)
            rate = self.config.learning_rate / math.sqrt(1 + step / 1_000)
            parameter -= rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert self.means is not None and self.scales is not None
        standardized = np.clip((features - self.means) / self.scales, -8, 8)
        return self.forward(standardized)[0]

    def serialize(self) -> dict[str, Any]:
        assert self.means is not None and self.scales is not None
        return {
            "kind": "deepsets",
            "config": self.config.__dict__,
            "seed": self.seed,
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "parameters": {name: value.tolist() for name, value in self.parameters.items()},
            "receipt": self.training_receipt,
        }


def fit_ensemble(
    config: ModelConfig,
    features: np.ndarray,
    targets: np.ndarray,
    dangerous: np.ndarray,
    normalized_gaps: np.ndarray,
    valid: np.ndarray,
    business_indexes: np.ndarray,
    seeds: Iterable[int],
    bootstrap: bool,
) -> list[DeepSetSelector]:
    models = []
    for seed in seeds:
        model = DeepSetSelector(config, seed)
        model.fit(
            features,
            targets,
            dangerous,
            normalized_gaps,
            valid,
            business_indexes,
            bootstrap,
        )
        models.append(model)
    return models


def ensemble_predict(models: list[DeepSetSelector], features: np.ndarray) -> dict[str, np.ndarray]:
    members = np.stack([model.predict(features) for model in models], axis=0)
    heads = np.mean(members, axis=0)
    risks = (
        MEAN_RISK_WEIGHT * members[..., MEAN_HEAD]
        + P90_RISK_WEIGHT * members[..., P90_HEAD]
    )
    return {
        "heads": heads,
        "risk": np.mean(risks, axis=0),
        "uncertainty": (
            np.std(risks, axis=0, ddof=1)
            if members.shape[0] > 1
            else np.zeros(risks.shape[1:])
        ),
    }


def selection_receipts(
    business_ids: np.ndarray,
    families: np.ndarray,
    candidate_ids: np.ndarray,
    valid: np.ndarray,
    excess_risk: np.ndarray,
    normalized_gaps: np.ndarray,
    dangerous: np.ndarray,
    predicted: dict[str, np.ndarray],
    policy: PolicyConfig,
) -> list[dict[str, Any]]:
    receipts = []
    heads = predicted["heads"]
    risk = predicted["risk"]
    uncertainty = predicted["uncertainty"]
    danger = heads[..., DANGER_HEAD]
    adjusted = risk + policy.danger_penalty * danger + policy.uncertainty_penalty * uncertainty
    for business in range(business_ids.size):
        active = np.flatnonzero(valid[business])
        order = active[np.argsort(adjusted[business, active], kind="stable")]
        selected = int(order[0])
        runner_up = float(adjusted[business, order[1]]) if order.size > 1 else math.inf
        margin = runner_up - float(adjusted[business, selected])
        relative_uncertainty = float(
            uncertainty[business, selected] / max(abs(risk[business, selected]) + 0.1, 0.1)
        )
        ambiguity = math.exp(
            -max(margin, 0) / max(abs(float(adjusted[business, selected])) + 0.1, 0.1)
        )
        confidence_risk = (
            0.4 * float(danger[business, selected])
            + 0.3 * min(relative_uncertainty, 1)
            + 0.3 * ambiguity
        )
        oracle = int(active[np.argmin(excess_risk[business, active])])
        indicator_bound = 0.0
        logistic_bound = 0.0
        for challenger in active:
            if int(challenger) == oracle:
                continue
            gap = float(normalized_gaps[business, challenger])
            difference = float(adjusted[business, oracle] - adjusted[business, challenger])
            if adjusted[business, challenger] <= adjusted[business, oracle] + 1e-12:
                indicator_bound += gap
            logistic_bound += gap * float(softplus(difference)) / math.log(2)
        receipts.append(
            {
                "businessId": str(business_ids[business]),
                "family": str(families[business]),
                "candidateId": str(candidate_ids[business, selected]),
                "actualExcessLoss": float(excess_risk[business, selected]),
                "normalizedExcessLoss": float(normalized_gaps[business, selected]),
                "dangerousFalseChampion": bool(dangerous[business, selected]),
                "oracle": bool(excess_risk[business, selected] <= 1e-10),
                "predictedRisk": float(risk[business, selected]),
                "predictedDanger": float(danger[business, selected]),
                "uncertainty": float(uncertainty[business, selected]),
                "adjustedRisk": float(adjusted[business, selected]),
                "runnerUpMargin": margin,
                "confidenceRisk": confidence_risk,
                "theoremIndicatorBound": indicator_bound,
                "theoremLogisticBound": logistic_bound,
                "theoremSlack": float(normalized_gaps[business, selected]) - indicator_bound,
            }
        )
    return receipts


def promoted_metrics(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    losses = np.asarray([float(row["actualExcessLoss"]) for row in receipts])
    normalized = np.asarray([float(row["normalizedExcessLoss"]) for row in receipts])
    return {
        "businesses": len(receipts),
        "meanPromotedExcessLoss": float(np.mean(losses)) if losses.size else None,
        "medianPromotedExcessLoss": quantile(losses, 0.5) if losses.size else None,
        "p90PromotedExcessLoss": quantile(losses, 0.9) if losses.size else None,
        "p95PromotedExcessLoss": quantile(losses, 0.95) if losses.size else None,
        "meanNormalizedExcessLoss": float(np.mean(normalized)) if normalized.size else None,
        "selectionObjective": (
            MEAN_RISK_WEIGHT * float(np.mean(losses))
            + P90_RISK_WEIGHT * quantile(losses, 0.9)
            if losses.size
            else None
        ),
        "dangerousFalseChampionShare": (
            float(np.mean([row["dangerousFalseChampion"] for row in receipts]))
            if receipts
            else None
        ),
        "oracleRecall": (
            float(np.mean([row["oracle"] for row in receipts])) if receipts else None
        ),
        "theoremViolations": sum(float(row["theoremSlack"]) > 1e-10 for row in receipts),
        "maximumTheoremSlack": max(
            0.0, *(float(row["theoremSlack"]) for row in receipts)
        ),
    }


def at_coverage(receipts: list[dict[str, Any]], coverage: float) -> dict[str, Any]:
    count = max(1, min(len(receipts), int(math.ceil(coverage * len(receipts) - 1e-12))))
    promoted = sorted(
        receipts,
        key=lambda row: (float(row["confidenceRisk"]), str(row["businessId"])),
    )[:count]
    output = promoted_metrics(promoted)
    output["requestedCoverage"] = coverage
    output["promotionCoverage"] = count / max(len(receipts), 1)
    return output


def risk_coverage_curve(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    points = [at_coverage(receipts, coverage) for coverage in COVERAGE_LEVELS]
    x = np.asarray([point["promotionCoverage"] for point in points], dtype=float)
    selection = np.asarray([point["selectionObjective"] for point in points], dtype=float)
    mean = np.asarray([point["meanPromotedExcessLoss"] for point in points], dtype=float)
    scale = max(float(x[-1] - x[0]), 1e-12)
    return {
        "range": [float(x[0]), float(x[-1])],
        "points": points,
        "areaUnderSelectionRisk": float(np.trapezoid(selection, x) / scale),
        "areaUnderMeanRisk": float(np.trapezoid(mean, x) / scale),
    }


def v8_receipts(source: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for row in source["selections"]:
        relative_uncertainty = float(row["uncertainty"]) / max(
            abs(float(row["predictedRisk"])) + 0.1, 0.1
        )
        ambiguity = math.exp(
            -max(float(row["runnerUpMargin"]), 0)
            / max(abs(float(row["adjustedRisk"])) + 0.1, 0.1)
        )
        output.append(
            {
                **row,
                "confidenceRisk": (
                    0.4 * float(row["predictedDanger"])
                    + 0.3 * min(relative_uncertainty, 1)
                    + 0.3 * ambiguity
                ),
            }
        )
    return output


def prepare_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["businessId"]), []).append(row)
    businesses = sorted(grouped)
    ordered = []
    families = []
    folds = []
    for business in businesses:
        candidates = sorted(grouped[business], key=lambda row: str(row["candidateId"]))
        if len(candidates) != CANDIDATES_PER_BUSINESS:
            raise RuntimeError(f"V9 expected 48 candidates for {business}")
        if len({int(row["fold"]) for row in candidates}) != 1:
            raise RuntimeError(f"V9 found multiple folds for {business}")
        ordered.append(candidates)
        families.append(str(candidates[0]["family"]))
        folds.append(int(candidates[0]["fold"]))
    features = np.asarray(
        [[row["features"] for row in candidates] for candidates in ordered], dtype=np.float64
    )
    scenario = np.asarray(
        [
            [
                [row["scenarioLoss"][name] for name in SCENARIOS]
                for row in candidates
            ]
            for candidates in ordered
        ],
        dtype=np.float64,
    )
    summary = np.zeros((len(businesses), CANDIDATES_PER_BUSINESS, 5), dtype=float)
    summary[..., MEAN_HEAD] = np.mean(scenario, axis=2)
    summary[..., MEDIAN_HEAD] = np.quantile(scenario, 0.5, axis=2)
    summary[..., P90_HEAD] = np.quantile(scenario, 0.9, axis=2)
    summary[..., P95_HEAD] = np.quantile(scenario, 0.95, axis=2)
    for business in range(len(businesses)):
        for candidate in range(CANDIDATES_PER_BUSINESS):
            summary[business, candidate, CVAR_HEAD] = cvar(scenario[business, candidate], 0.9)
    targets = np.concatenate((summary, scenario), axis=2)
    return {
        "features": features,
        "targets": targets,
        "dangerous": np.asarray(
            [[row["dangerous"] for row in candidates] for candidates in ordered], dtype=float
        ),
        "normalizedGaps": np.asarray(
            [
                [row["normalizedExcessEconomicRisk"] for row in candidates]
                for candidates in ordered
            ],
            dtype=float,
        ),
        "excessRisk": np.asarray(
            [[row["excessEconomicRisk"] for row in candidates] for candidates in ordered],
            dtype=float,
        ),
        "valid": np.asarray(
            [[row["valid"] for row in candidates] for candidates in ordered], dtype=bool
        ),
        "businessIds": np.asarray(businesses, dtype=object),
        "families": np.asarray(families, dtype=object),
        "folds": np.asarray(folds, dtype=int),
        "candidateIds": np.asarray(
            [[row["candidateId"] for row in candidates] for candidates in ordered],
            dtype=object,
        ),
    }


def development_objective(receipts: list[dict[str, Any]]) -> float:
    full = promoted_metrics(receipts)
    curve = risk_coverage_curve(receipts)
    return 0.5 * float(full["selectionObjective"]) + 0.5 * float(
        curve["areaUnderSelectionRisk"]
    )


def main() -> None:
    started = time.time()
    dataset_source = DATASET_PATH.read_bytes()
    v8_prediction_source = V8_PREDICTIONS_PATH.read_bytes()
    v8_development_source = V8_DEVELOPMENT_PATH.read_bytes()
    dataset = json.loads(dataset_source)
    v8_prediction_artifact = json.loads(v8_prediction_source)
    v8_development = json.loads(v8_development_source)
    if (
        dataset.get("activation") != "development-only"
        or dataset.get("provenance", {}).get("v8AuditDereferenced")
        or dataset.get("provenance", {}).get("nutsUsed")
        or len(dataset.get("rows", [])) != 20_160
        or len(dataset.get("featureNames", [])) != 232
    ):
        raise RuntimeError("V9 requires the frozen 420 x 48 x 232 development contract")
    forbidden = tuple(dataset["contract"]["tokens"]["forbidden"])
    if any(
        any(term in str(name).lower() for term in forbidden)
        for name in dataset["featureNames"]
    ):
        raise RuntimeError("V9 token registry contains truth or identity")
    arrays = prepare_arrays(dataset["rows"])
    features = arrays["features"]
    targets = arrays["targets"]
    dangerous = arrays["dangerous"]
    normalized_gaps = arrays["normalizedGaps"]
    valid = arrays["valid"]
    folds = arrays["folds"]
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
        raise RuntimeError("V9 inputs are not finite")

    outer_heads = np.full((features.shape[0], features.shape[1], HEAD_COUNT), np.nan)
    outer_risk = np.full(features.shape[:2], np.nan)
    outer_uncertainty = np.full(features.shape[:2], np.nan)
    outer_receipts = []
    selections = []
    chosen: list[tuple[ModelConfig, PolicyConfig]] = []
    for outer_fold in sorted(np.unique(folds)):
        outer_train = np.flatnonzero(folds != outer_fold)
        outer_assessment = np.flatnonzero(folds == outer_fold)
        inner_folds = sorted(np.unique(folds[outer_train]))
        best: tuple[float, int, str, ModelConfig, PolicyConfig] | None = None
        candidates = []
        for config in MODEL_GRID:
            inner_heads = np.full_like(outer_heads, np.nan)
            inner_risk = np.full_like(outer_risk, np.nan)
            for inner_fold in inner_folds:
                inner_train = outer_train[folds[outer_train] != inner_fold]
                inner_assessment = outer_train[folds[outer_train] == inner_fold]
                model = fit_ensemble(
                    config,
                    features,
                    targets,
                    dangerous,
                    normalized_gaps,
                    valid,
                    inner_train,
                    (90_000 + int(outer_fold) * 1_000 + int(inner_fold),),
                    bootstrap=False,
                )[0]
                prediction = ensemble_predict([model], features[inner_assessment])
                inner_heads[inner_assessment] = prediction["heads"]
                inner_risk[inner_assessment] = prediction["risk"]
            for policy in POLICY_GRID:
                predicted = {
                    "heads": inner_heads[outer_train],
                    "risk": inner_risk[outer_train],
                    "uncertainty": np.zeros_like(inner_risk[outer_train]),
                }
                receipt = selection_receipts(
                    arrays["businessIds"][outer_train],
                    arrays["families"][outer_train],
                    arrays["candidateIds"][outer_train],
                    valid[outer_train],
                    arrays["excessRisk"][outer_train],
                    normalized_gaps[outer_train],
                    dangerous[outer_train].astype(bool),
                    predicted,
                    policy,
                )
                objective = development_objective(receipt)
                candidates.append(
                    {"model": config.id, "policy": policy.id, "objective": objective}
                )
                key = (objective, config.candidate_width, f"{config.id}/{policy.id}", config, policy)
                if best is None or key[:3] < best[:3]:
                    best = key
        assert best is not None
        config, policy = best[3], best[4]
        chosen.append((config, policy))
        models = fit_ensemble(
            config,
            features,
            targets,
            dangerous,
            normalized_gaps,
            valid,
            outer_train,
            tuple(100_000 + int(outer_fold) * 100 + seed for seed in range(3)),
            bootstrap=True,
        )
        prediction = ensemble_predict(models, features[outer_assessment])
        outer_heads[outer_assessment] = prediction["heads"]
        outer_risk[outer_assessment] = prediction["risk"]
        outer_uncertainty[outer_assessment] = prediction["uncertainty"]
        receipt = selection_receipts(
            arrays["businessIds"][outer_assessment],
            arrays["families"][outer_assessment],
            arrays["candidateIds"][outer_assessment],
            valid[outer_assessment],
            arrays["excessRisk"][outer_assessment],
            normalized_gaps[outer_assessment],
            dangerous[outer_assessment].astype(bool),
            prediction,
            policy,
        )
        selections.extend(receipt)
        fold_result = {
            "fold": int(outer_fold),
            "selectedModel": config.id,
            "selectedPolicy": policy.id,
            "innerObjective": best[0],
            "assessmentAt100Percent": promoted_metrics(receipt),
            "assessmentAt70Percent": at_coverage(receipt, OPERATIONAL_COVERAGE),
            "candidateConfigurations": candidates,
            "ensembleReceipts": [model.training_receipt for model in models],
        }
        outer_receipts.append(fold_result)
        print(json.dumps(fold_result), flush=True)

    if not np.all(np.isfinite(outer_risk)):
        raise RuntimeError("V9 failed to cross-fit every candidate")
    v9_full = promoted_metrics(selections)
    v9_curve = risk_coverage_curve(selections)
    v9_operational = at_coverage(selections, OPERATIONAL_COVERAGE)
    v8_selection = v8_receipts(v8_prediction_artifact)
    v8_full = promoted_metrics(v8_selection)
    v8_curve = risk_coverage_curve(v8_selection)
    v8_operational = at_coverage(v8_selection, OPERATIONAL_COVERAGE)
    v8_native_coverage = float(v8_development["development"]["promotionCoverage"])
    matched_v9 = at_coverage(selections, v8_native_coverage)
    matched_v8 = at_coverage(v8_selection, v8_native_coverage)

    combination_counts: dict[str, int] = {}
    combination_values: dict[str, tuple[ModelConfig, PolicyConfig]] = {}
    for config, policy in chosen:
        key = f"{config.id}/{policy.id}"
        combination_counts[key] = combination_counts.get(key, 0) + 1
        combination_values[key] = (config, policy)
    frozen_key = sorted(
        combination_counts,
        key=lambda key: (
            -combination_counts[key],
            combination_values[key][0].candidate_width,
            key,
        ),
    )[0]
    frozen_config, frozen_policy = combination_values[frozen_key]
    all_businesses = np.arange(features.shape[0])
    final_models = fit_ensemble(
        frozen_config,
        features,
        targets,
        dangerous,
        normalized_gaps,
        valid,
        all_businesses,
        tuple(110_000 + seed for seed in range(5)),
        bootstrap=True,
    )
    model_artifact = {
        "artifactId": "flux-svi-score-v9-development-selector-v1",
        "version": dataset["version"],
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "activation": "development-only",
        "provenance": {
            "datasetSha256": sha256(dataset_source),
            "v8AuditDereferenced": False,
            "nutsUsed": False,
        },
        "featureNames": dataset["featureNames"],
        "model": frozen_config.__dict__,
        "policy": frozen_policy.__dict__,
        "models": [model.serialize() for model in final_models],
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = MODEL_PATH.with_suffix(".json.tmp")
    temporary_model.write_text(json.dumps(model_artifact, separators=(",", ":")))
    temporary_model.replace(MODEL_PATH)

    prediction_rows = []
    for business in range(features.shape[0]):
        for candidate in range(features.shape[1]):
            heads = outer_heads[business, candidate]
            prediction_rows.append(
                {
                    "businessId": str(arrays["businessIds"][business]),
                    "candidateId": str(arrays["candidateIds"][business, candidate]),
                    "fold": int(folds[business]),
                    "predictedMean": float(heads[MEAN_HEAD]),
                    "predictedMedian": float(heads[MEDIAN_HEAD]),
                    "predictedP90": float(heads[P90_HEAD]),
                    "predictedP95": float(heads[P95_HEAD]),
                    "predictedCvar90": float(heads[CVAR_HEAD]),
                    "predictedScenarios": {
                        name: float(heads[SCENARIO_SLICE.start + index])
                        for index, name in enumerate(SCENARIOS)
                    },
                    "predictedDanger": float(heads[DANGER_HEAD]),
                    "predictedRisk": float(outer_risk[business, candidate]),
                    "uncertainty": float(outer_uncertainty[business, candidate]),
                }
            )
    predictions_artifact = {
        "artifactId": "flux-svi-score-v9-cross-fitted-predictions-v1",
        "version": dataset["version"],
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "activation": "development-only",
        "provenance": {
            "datasetSha256": sha256(dataset_source),
            "v8AuditDereferenced": False,
            "nutsUsed": False,
        },
        "predictions": prediction_rows,
        "selections": selections,
    }
    temporary_predictions = PREDICTIONS_PATH.with_suffix(".json.tmp")
    temporary_predictions.write_text(json.dumps(predictions_artifact, separators=(",", ":")))
    temporary_predictions.replace(PREDICTIONS_PATH)

    checks = {
        "fullCoverageMeanImproved": (
            v9_full["meanPromotedExcessLoss"] < v8_full["meanPromotedExcessLoss"]
        ),
        "fullCoverageP90NonInferior": (
            v9_full["p90PromotedExcessLoss"] <= 1.02 * v8_full["p90PromotedExcessLoss"]
        ),
        "fullCoverageP95NonInferior": (
            v9_full["p95PromotedExcessLoss"] <= 1.02 * v8_full["p95PromotedExcessLoss"]
        ),
        "matchedCoverageSelectionObjectiveImproved": (
            matched_v9["selectionObjective"] < matched_v8["selectionObjective"]
        ),
        "selectiveRiskAreaImproved": (
            v9_curve["areaUnderSelectionRisk"] < v8_curve["areaUnderSelectionRisk"]
        ),
        "dangerousFalseChampionImprovedAtOperationalCoverage": (
            v9_operational["dangerousFalseChampionShare"]
            < v8_operational["dangerousFalseChampionShare"]
        ),
        "minimumOperationalCoverage": v9_operational["promotionCoverage"] >= 0.70,
        "theoremVerified": v9_full["theoremViolations"] == 0,
        "v8AuditUntouched": True,
        "nutsExcluded": True,
    }
    artifact = {
        "artifactId": "flux-svi-score-v9-setwise-distributional-development-v1",
        "version": dataset["version"],
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "activation": "development-only",
        "stage": "nested-grouped-development-complete",
        "contract": dataset["contract"],
        "provenance": {
            "datasetSha256": sha256(dataset_source),
            "v8PredictionsSha256": sha256(v8_prediction_source),
            "v8DevelopmentSha256": sha256(v8_development_source),
            "predictionsSha256": sha256(PREDICTIONS_PATH.read_bytes()),
            "modelSha256": sha256(MODEL_PATH.read_bytes()),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "v8AuditDereferenced": False,
            "nutsUsed": False,
        },
        "nestedCrossValidation": {
            "outerFolds": 5,
            "innerFoldsPerOuter": 4,
            "modelGrid": [config.__dict__ for config in MODEL_GRID],
            "policyGrid": [policy.__dict__ for policy in POLICY_GRID],
            "outerReceipts": outer_receipts,
        },
        "v9": {
            "at100PercentCoverage": v9_full,
            "atOperationalCoverage": v9_operational,
            "riskCoverageCurve": v9_curve,
        },
        "v8Comparator": {
            "at100PercentCoverage": v8_full,
            "atOperationalCoverage": v8_operational,
            "atNativeCoverage": matched_v8,
            "riskCoverageCurve": v8_curve,
        },
        "matchedCoverage": {
            "coverage": v8_native_coverage,
            "v9": matched_v9,
            "v8": matched_v8,
        },
        "v5dSviComparatorAt100PercentCoverage": v8_development[
            "v5dSviComparatorOnV8Targets"
        ],
        "checks": checks,
        "conclusion": {
            "allDevelopmentChecksPassed": all(checks.values()),
            "eligibleToFreezeForNewAudit": all(checks.values()),
            "v8AuditReused": False,
            "newSealedAuditRequired": True,
            "productionActivationPermitted": False,
            "sotaClaimPermitted": False,
        },
        "developmentSelector": {
            "model": frozen_config.id,
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
                "v9": artifact["v9"],
                "v8": artifact["v8Comparator"],
                "matchedCoverage": artifact["matchedCoverage"],
                "checks": checks,
                "conclusion": artifact["conclusion"],
                "runtimeSeconds": artifact["runtimeSeconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
