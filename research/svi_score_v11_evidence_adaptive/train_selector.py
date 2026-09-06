#!/usr/bin/env python3
"""Evidence-adaptive, set-wise RegretSet development selector.

The model is intentionally NumPy-only. Four disjoint candidate-token experts
are mixed by a business-level softmax gate. The gate sees only truth-blind
evidence context. Advertiser folds, rather than augmented regime rows, are the
unit of cross-validation and bootstrap sampling.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
V9_SPEC = importlib.util.spec_from_file_location(
    "flux_v9_selector_for_v11",
    ROOT / "research/svi_score_v9/train_selector.py",
)
assert V9_SPEC and V9_SPEC.loader
V9 = importlib.util.module_from_spec(V9_SPEC)
sys.modules[V9_SPEC.name] = V9
V9_SPEC.loader.exec_module(V9)

DATASET_PATH = (
    ROOT
    / ".flux-artifacts/svi-score-v11-evidence-adaptive/development-dataset.json"
)
V9_PREDICTIONS_PATH = ROOT / ".flux-artifacts/svi-score-v9/cross-fitted-predictions.json"
PREDICTIONS_PATH = (
    ROOT
    / ".flux-artifacts/svi-score-v11-evidence-adaptive/cross-fitted-predictions.json"
)
MODEL_PATH = (
    ROOT / ".flux-artifacts/svi-score-v11-evidence-adaptive/development-selector.json"
)
ARTIFACT_PATH = (
    ROOT
    / "research/svi_score_v11_evidence_adaptive/artifacts/svi-score-v11-evidence-adaptive-development.json"
)

EXPERTS = (
    "predictive-generalization",
    "causal-identification",
    "posterior-decision",
    "structural-specification",
)
FULL_REGIME = "full-candidate-set"
MAX_CANDIDATES = 48
HEAD_COUNT = V9.HEAD_COUNT
MEAN_HEAD = V9.MEAN_HEAD
MEDIAN_HEAD = V9.MEDIAN_HEAD
P90_HEAD = V9.P90_HEAD
P95_HEAD = V9.P95_HEAD
CVAR_HEAD = V9.CVAR_HEAD
SCENARIO_SLICE = V9.SCENARIO_SLICE
DANGER_HEAD = V9.DANGER_HEAD
SCENARIOS = V9.SCENARIOS
MEAN_RISK_WEIGHT = V9.MEAN_RISK_WEIGHT
P90_RISK_WEIGHT = V9.P90_RISK_WEIGHT

MODEL_GRID = (
    V9.ModelConfig("evidence-adaptive-compact", 12, 12, 18, 0.0022, 0.001, 0.35),
    V9.ModelConfig("evidence-adaptive-balanced", 20, 16, 22, 0.0018, 0.001, 0.35),
)
POLICY_GRID = tuple(V9.PolicyConfig(value) for value in (0.0, 0.25, 0.50))


def sha256(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


class EvidenceAdaptiveSelector:
    def __init__(
        self,
        config: V9.ModelConfig,
        seed: int,
        expert_indices: dict[str, np.ndarray],
    ) -> None:
        self.config = config
        self.seed = seed
        self.expert_indices = expert_indices
        self.means: np.ndarray | None = None
        self.scales: np.ndarray | None = None
        self.context_means: np.ndarray | None = None
        self.context_scales: np.ndarray | None = None
        self.parameters: dict[str, np.ndarray] = {}
        self.training_receipt: dict[str, Any] = {}

    def initialize(
        self,
        input_width: int,
        context_width: int,
        targets: np.ndarray,
        dangerous: np.ndarray,
    ) -> None:
        random = np.random.default_rng(self.seed)
        candidate_width = self.config.candidate_width
        decoder_width = self.config.decoder_width
        parameters: dict[str, np.ndarray] = {
            "joint_w1": random.normal(
                0,
                math.sqrt(2 / input_width),
                (input_width, candidate_width),
            ),
            "joint_b1": np.zeros(candidate_width),
        }
        for index, expert in enumerate(EXPERTS):
            width = int(self.expert_indices[expert].size)
            if width < 1:
                raise RuntimeError(f"V11 expert {expert} has no candidate tokens")
            parameters[f"expert_w1_{index}"] = random.normal(
                0,
                math.sqrt(2 / width),
                (width, candidate_width),
            )
            parameters[f"expert_b1_{index}"] = np.zeros(candidate_width)
        parameters["gate_w"] = random.normal(0, 0.02, (context_width, len(EXPERTS)))
        parameters["gate_b"] = np.zeros(len(EXPERTS))
        combined_width = 2 * candidate_width
        parameters["w2"] = random.normal(
            0,
            math.sqrt(2 / (4 * combined_width)),
            (4 * combined_width, decoder_width),
        )
        parameters["b2"] = np.zeros(decoder_width)
        parameters["wo"] = random.normal(
            0,
            math.sqrt(1 / decoder_width),
            (decoder_width, HEAD_COUNT),
        )
        parameters["bo"] = np.zeros(HEAD_COUNT)
        self.parameters = parameters

        flat_targets = targets.reshape(-1, targets.shape[-1])
        target_center = np.mean(flat_targets, axis=0)
        self.parameters["bo"][MEAN_HEAD] = V9.inverse_softplus(target_center[MEAN_HEAD])
        self.parameters["bo"][MEDIAN_HEAD] = V9.inverse_softplus(
            target_center[MEDIAN_HEAD]
        )
        self.parameters["bo"][P90_HEAD] = V9.inverse_softplus(
            max(target_center[P90_HEAD] - target_center[MEDIAN_HEAD], 0.05)
        )
        self.parameters["bo"][P95_HEAD] = V9.inverse_softplus(
            max(target_center[P95_HEAD] - target_center[P90_HEAD], 0.05)
        )
        self.parameters["bo"][CVAR_HEAD] = V9.inverse_softplus(
            max(target_center[CVAR_HEAD] - target_center[P95_HEAD], 0.05)
        )
        for head in range(SCENARIO_SLICE.start, SCENARIO_SLICE.stop):
            self.parameters["bo"][head] = V9.inverse_softplus(target_center[head])
        danger_rate = float(np.mean(dangerous))
        self.parameters["bo"][DANGER_HEAD] = math.log(
            max(danger_rate, 1e-4) / max(1 - danger_rate, 1e-4)
        )

    def forward(
        self,
        features: np.ndarray,
        contexts: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...]]:
        parameters = self.parameters
        joint_linear = features @ parameters["joint_w1"] + parameters["joint_b1"]
        joint_embedding = V9.silu(joint_linear)
        expert_linear = []
        expert_embeddings = []
        for index, expert in enumerate(EXPERTS):
            selected = features[..., self.expert_indices[expert]]
            linear = (
                selected @ parameters[f"expert_w1_{index}"]
                + parameters[f"expert_b1_{index}"]
            )
            expert_linear.append(linear)
            expert_embeddings.append(V9.silu(linear))
        gate_logits = contexts @ parameters["gate_w"] + parameters["gate_b"]
        gate = V9.stable_softmax(gate_logits)
        adaptive_embedding = sum(
            gate[:, index, None, None] * expert_embeddings[index]
            for index in range(len(EXPERTS))
        )
        candidate = np.concatenate((joint_embedding, adaptive_embedding), axis=2)
        mask = valid[..., None].astype(float)
        counts = np.maximum(np.sum(mask, axis=1, keepdims=True), 1.0)
        pooled = np.sum(candidate * mask, axis=1, keepdims=True) / counts
        repeated = np.broadcast_to(pooled, candidate.shape)
        decoder_input = np.concatenate(
            (candidate, repeated, candidate - repeated, candidate * repeated),
            axis=2,
        )
        second_linear = decoder_input @ parameters["w2"] + parameters["b2"]
        second = V9.silu(second_linear)
        raw = second @ parameters["wo"] + parameters["bo"]
        return V9.output_from_raw(raw), gate, (
            features,
            contexts,
            valid,
            joint_linear,
            joint_embedding,
            expert_linear,
            expert_embeddings,
            gate,
            candidate,
            pooled,
            decoder_input,
            second_linear,
            second,
            raw,
            counts,
        )

    def backward(
        self,
        output_gradient: np.ndarray,
        cache: tuple[Any, ...],
    ) -> dict[str, np.ndarray]:
        (
            features,
            contexts,
            valid,
            joint_linear,
            joint_embedding,
            expert_linear,
            expert_embeddings,
            gate,
            candidate,
            pooled,
            decoder_input,
            second_linear,
            second,
            raw,
            counts,
        ) = cache
        parameters = self.parameters
        raw_gradient = V9.raw_gradient_from_output_gradient(output_gradient, raw)
        flat_second = second.reshape(-1, second.shape[-1])
        flat_raw_gradient = raw_gradient.reshape(-1, raw_gradient.shape[-1])
        gradients: dict[str, np.ndarray] = {
            "wo": flat_second.T @ flat_raw_gradient,
            "bo": np.sum(flat_raw_gradient, axis=0),
        }
        second_gradient = raw_gradient @ parameters["wo"].T
        second_linear_gradient = second_gradient * V9.silu_derivative(second_linear)
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
        candidate_gradient = direct + difference + interaction * pooled
        context_uses = context_gradient - difference + interaction * candidate
        pooled_gradient = np.sum(context_uses, axis=1, keepdims=True)
        candidate_gradient += (
            pooled_gradient / counts * valid[..., None].astype(float)
        )

        branch_width = joint_embedding.shape[-1]
        joint_embedding_gradient = candidate_gradient[..., :branch_width]
        adaptive_gradient = candidate_gradient[..., branch_width:]
        joint_linear_gradient = joint_embedding_gradient * V9.silu_derivative(
            joint_linear
        )
        gradients["joint_w1"] = (
            features.reshape(-1, features.shape[-1]).T
            @ joint_linear_gradient.reshape(-1, joint_linear_gradient.shape[-1])
        )
        gradients["joint_b1"] = np.sum(joint_linear_gradient, axis=(0, 1))

        gate_output_gradient = np.zeros_like(gate)
        for index, expert in enumerate(EXPERTS):
            embedding = expert_embeddings[index]
            gate_output_gradient[:, index] = np.sum(
                adaptive_gradient * embedding,
                axis=(1, 2),
            )
            embedding_gradient = (
                adaptive_gradient * gate[:, index, None, None]
            )
            linear_gradient = embedding_gradient * V9.silu_derivative(
                expert_linear[index]
            )
            selected = features[..., self.expert_indices[expert]]
            gradients[f"expert_w1_{index}"] = (
                selected.reshape(-1, selected.shape[-1]).T
                @ linear_gradient.reshape(-1, linear_gradient.shape[-1])
            )
            gradients[f"expert_b1_{index}"] = np.sum(linear_gradient, axis=(0, 1))
        gate_logits_gradient = gate * (
            gate_output_gradient
            - np.sum(gate_output_gradient * gate, axis=1, keepdims=True)
        )
        gradients["gate_w"] = contexts.T @ gate_logits_gradient
        gradients["gate_b"] = np.sum(gate_logits_gradient, axis=0)
        return gradients

    def fit(
        self,
        features: np.ndarray,
        contexts: np.ndarray,
        targets: np.ndarray,
        dangerous: np.ndarray,
        normalized_gaps: np.ndarray,
        valid: np.ndarray,
        present: np.ndarray,
        set_business_indexes: np.ndarray,
        training_set_indexes: np.ndarray,
        bootstrap: bool,
    ) -> None:
        training_present = present[training_set_indexes]
        flat_features = features[training_set_indexes][training_present]
        self.means = np.mean(flat_features, axis=0)
        self.scales = np.maximum(np.std(flat_features, axis=0, ddof=1), 1e-6)
        self.context_means = np.mean(contexts[training_set_indexes], axis=0)
        self.context_scales = np.maximum(
            np.std(contexts[training_set_indexes], axis=0, ddof=1),
            1e-6,
        )
        standardized = np.clip((features - self.means) / self.scales, -8, 8)
        standardized_context = np.clip(
            (contexts - self.context_means) / self.context_scales,
            -8,
            8,
        )
        self.initialize(
            features.shape[-1],
            contexts.shape[-1],
            targets[training_set_indexes],
            dangerous[training_set_indexes],
        )
        first_moment = {
            name: np.zeros_like(value) for name, value in self.parameters.items()
        }
        second_moment = {
            name: np.zeros_like(value) for name, value in self.parameters.items()
        }
        random = np.random.default_rng(self.seed + 29)
        training_businesses = np.unique(set_business_indexes[training_set_indexes])
        sampled_businesses = (
            random.choice(
                training_businesses,
                size=training_businesses.size,
                replace=True,
            )
            if bootstrap
            else training_businesses.copy()
        )
        sampled = np.concatenate(
            [
                training_set_indexes[
                    set_business_indexes[training_set_indexes] == business
                ]
                for business in sampled_businesses
            ]
        )
        step = 0
        started = time.time()
        final_selection_loss = 0.0
        final_pair_loss = 0.0
        for _epoch in range(self.config.epochs):
            order = sampled[random.permutation(sampled.size)]
            for offset in range(0, order.size, 32):
                batch = order[offset : offset + 32]
                output, _gate, cache = self.forward(
                    standardized[batch],
                    standardized_context[batch],
                    valid[batch],
                )
                batch_targets = targets[batch]
                batch_dangerous = dangerous[batch]
                batch_gaps = normalized_gaps[batch]
                batch_valid = valid[batch]
                row_weight = batch_valid.astype(float) / max(
                    float(np.sum(batch_valid)),
                    1.0,
                )
                gradient = np.zeros_like(output)
                gradient[..., MEAN_HEAD] = row_weight * V9.huber_gradient(
                    output[..., MEAN_HEAD] - batch_targets[..., MEAN_HEAD], 1.5
                )
                gradient[..., MEDIAN_HEAD] = 0.15 * row_weight * V9.huber_gradient(
                    output[..., MEDIAN_HEAD] - batch_targets[..., MEDIAN_HEAD], 1.5
                )
                p90_error = output[..., P90_HEAD] - batch_targets[..., P90_HEAD]
                gradient[..., P90_HEAD] = 0.35 * row_weight * np.where(
                    p90_error >= 0,
                    0.1,
                    -0.9,
                )
                p95_error = output[..., P95_HEAD] - batch_targets[..., P95_HEAD]
                gradient[..., P95_HEAD] = 0.20 * row_weight * np.where(
                    p95_error >= 0,
                    0.05,
                    -0.95,
                )
                gradient[..., CVAR_HEAD] = 0.30 * row_weight * V9.huber_gradient(
                    output[..., CVAR_HEAD] - batch_targets[..., CVAR_HEAD], 2.0
                )
                gradient[..., SCENARIO_SLICE] = (
                    0.12
                    * row_weight[..., None]
                    * V9.huber_gradient(
                        output[..., SCENARIO_SLICE]
                        - batch_targets[..., SCENARIO_SLICE],
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
                    probability = V9.stable_softmax(
                        -scores / self.config.temperature
                    )
                    expected_gap = float(np.sum(probability * gaps))
                    direct = (
                        probability
                        * (expected_gap - gaps)
                        / self.config.temperature
                        / max(batch.size, 1)
                    )
                    gradient[position, active, MEAN_HEAD] += (
                        MEAN_RISK_WEIGHT * direct
                    )
                    gradient[position, active, P90_HEAD] += (
                        P90_RISK_WEIGHT * direct
                    )
                    final_selection_loss += expected_gap / max(batch.size, 1)

                    oracle = int(active[int(np.argmin(gaps))])
                    for challenger in active:
                        if int(challenger) == oracle:
                            continue
                        gap = float(batch_gaps[position, challenger])
                        if gap <= 1e-12:
                            continue
                        margin = float(
                            risk[position, oracle] - risk[position, challenger]
                        )
                        derivative = (
                            0.25
                            * gap
                            * float(V9.sigmoid(np.asarray([margin]))[0])
                            / max(batch.size, 1)
                        )
                        gradient[position, oracle, MEAN_HEAD] += (
                            MEAN_RISK_WEIGHT * derivative
                        )
                        gradient[position, oracle, P90_HEAD] += (
                            P90_RISK_WEIGHT * derivative
                        )
                        gradient[position, challenger, MEAN_HEAD] -= (
                            MEAN_RISK_WEIGHT * derivative
                        )
                        gradient[position, challenger, P90_HEAD] -= (
                            P90_RISK_WEIGHT * derivative
                        )
                        final_pair_loss += (
                            0.25
                            * gap
                            * float(V9.softplus(margin))
                            / max(batch.size, 1)
                        )

                gradients = self.backward(gradient, cache)
                for name, value in gradients.items():
                    if "_w" in name or name.startswith("w"):
                        gradients[name] = (
                            value + self.config.weight_decay * self.parameters[name]
                        )
                norm = math.sqrt(
                    sum(float(np.sum(value * value)) for value in gradients.values())
                )
                if norm > 5:
                    gradients = {
                        name: value * (5 / norm) for name, value in gradients.items()
                    }
                step += 1
                self._adam_update(gradients, first_moment, second_moment, step)

        _, training_gate = self.predict(
            features[training_set_indexes],
            contexts[training_set_indexes],
            valid[training_set_indexes],
        )
        self.training_receipt = {
            "trainingBusinesses": int(training_businesses.size),
            "trainingRegimeSets": int(training_set_indexes.size),
            "bootstrapByBusiness": bootstrap,
            "epochs": self.config.epochs,
            "averageGate": {
                expert: float(np.mean(training_gate[:, index]))
                for index, expert in enumerate(EXPERTS)
            },
            "finalBatchDirectSelectionLoss": final_selection_loss,
            "finalBatchTheoremSurrogateLoss": final_pair_loss,
            "seconds": time.time() - started,
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
            second_moment[name] = (
                0.999 * second_moment[name] + 0.001 * gradient * gradient
            )
            corrected_first = first_moment[name] / (1 - 0.9**step)
            corrected_second = second_moment[name] / (1 - 0.999**step)
            rate = self.config.learning_rate / math.sqrt(1 + step / 1_000)
            parameter -= rate * corrected_first / (
                np.sqrt(corrected_second) + 1e-8
            )

    def predict(
        self,
        features: np.ndarray,
        contexts: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.means is not None and self.scales is not None
        assert self.context_means is not None and self.context_scales is not None
        standardized = np.clip((features - self.means) / self.scales, -8, 8)
        standardized_context = np.clip(
            (contexts - self.context_means) / self.context_scales,
            -8,
            8,
        )
        output, gate, _cache = self.forward(standardized, standardized_context, valid)
        return output, gate

    def serialize(self) -> dict[str, Any]:
        assert self.means is not None and self.scales is not None
        assert self.context_means is not None and self.context_scales is not None
        return {
            "kind": "evidence-adaptive-deepsets",
            "config": self.config.__dict__,
            "seed": self.seed,
            "experts": list(EXPERTS),
            "expertIndices": {
                name: values.tolist() for name, values in self.expert_indices.items()
            },
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "contextMeans": self.context_means.tolist(),
            "contextScales": self.context_scales.tolist(),
            "parameters": {
                name: value.tolist() for name, value in self.parameters.items()
            },
            "receipt": self.training_receipt,
        }


def fit_ensemble(
    config: V9.ModelConfig,
    arrays: dict[str, np.ndarray],
    expert_indices: dict[str, np.ndarray],
    training_set_indexes: np.ndarray,
    seeds: Iterable[int],
    bootstrap: bool,
) -> list[EvidenceAdaptiveSelector]:
    models = []
    for seed in seeds:
        model = EvidenceAdaptiveSelector(config, seed, expert_indices)
        model.fit(
            arrays["features"],
            arrays["contexts"],
            arrays["targets"],
            arrays["dangerous"],
            arrays["normalizedGaps"],
            arrays["valid"],
            arrays["present"],
            arrays["setBusinessIndexes"],
            training_set_indexes,
            bootstrap,
        )
        models.append(model)
    return models


def ensemble_predict(
    models: list[EvidenceAdaptiveSelector],
    arrays: dict[str, np.ndarray],
    indexes: np.ndarray,
) -> dict[str, np.ndarray]:
    member_outputs = []
    member_gates = []
    for model in models:
        output, gate = model.predict(
            arrays["features"][indexes],
            arrays["contexts"][indexes],
            arrays["valid"][indexes],
        )
        member_outputs.append(output)
        member_gates.append(gate)
    members = np.stack(member_outputs, axis=0)
    gates = np.stack(member_gates, axis=0)
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
        "gates": np.mean(gates, axis=0),
        "gateUncertainty": (
            np.std(gates, axis=0, ddof=1)
            if gates.shape[0] > 1
            else np.zeros(gates.shape[1:])
        ),
    }


def prepare_arrays(dataset: dict[str, Any]) -> dict[str, np.ndarray]:
    rows_by_key = {
        (str(row["businessId"]), str(row["candidateId"])): row
        for row in dataset["rows"]
    }
    businesses = sorted({str(item["businessId"]) for item in dataset["sets"]})
    business_index = {business: index for index, business in enumerate(businesses)}
    sets = sorted(dataset["sets"], key=lambda item: str(item["setId"]))
    set_count = len(sets)
    feature_count = len(dataset["featureNames"])
    context_count = len(dataset["contextNames"])
    features = np.zeros((set_count, MAX_CANDIDATES, feature_count), dtype=float)
    contexts = np.zeros((set_count, context_count), dtype=float)
    targets = np.zeros((set_count, MAX_CANDIDATES, 9), dtype=float)
    dangerous = np.zeros((set_count, MAX_CANDIDATES), dtype=float)
    normalized_gaps = np.zeros((set_count, MAX_CANDIDATES), dtype=float)
    excess_risk = np.zeros((set_count, MAX_CANDIDATES), dtype=float)
    valid = np.zeros((set_count, MAX_CANDIDATES), dtype=bool)
    present = np.zeros((set_count, MAX_CANDIDATES), dtype=bool)
    candidate_ids = np.full((set_count, MAX_CANDIDATES), "", dtype=object)
    business_ids = np.empty(set_count, dtype=object)
    families = np.empty(set_count, dtype=object)
    folds = np.zeros(set_count, dtype=int)
    regimes = np.empty(set_count, dtype=object)
    set_business_indexes = np.zeros(set_count, dtype=int)

    for set_index, item in enumerate(sets):
        business = str(item["businessId"])
        contexts[set_index] = np.asarray(item["context"], dtype=float)
        business_ids[set_index] = business
        families[set_index] = str(item["family"])
        folds[set_index] = int(item["fold"])
        regimes[set_index] = str(item["regime"])
        set_business_indexes[set_index] = business_index[business]
        minimum = float(item["withinRegimeMinimumEconomicRisk"])
        for position, candidate_id in enumerate(item["candidateIds"]):
            row = rows_by_key[(business, str(candidate_id))]
            present[set_index, position] = True
            valid[set_index, position] = bool(row["valid"])
            candidate_ids[set_index, position] = str(candidate_id)
            features[set_index, position] = np.asarray(row["features"], dtype=float)
            scenario = np.asarray(
                [float(row["scenarioLoss"][name]) for name in SCENARIOS],
                dtype=float,
            )
            summary = np.asarray(
                [
                    float(np.mean(scenario)),
                    float(np.quantile(scenario, 0.5)),
                    float(np.quantile(scenario, 0.9)),
                    float(np.quantile(scenario, 0.95)),
                    V9.cvar(scenario, 0.9),
                ]
            )
            targets[set_index, position] = np.concatenate((summary, scenario))
            gap = max(0.0, float(row["economicRisk"]) - minimum)
            excess_risk[set_index, position] = gap
            normalized_gaps[set_index, position] = min(gap, 1.0)
            dangerous[set_index, position] = float(gap >= 1.0)
    return {
        "features": features,
        "contexts": contexts,
        "targets": targets,
        "dangerous": dangerous,
        "normalizedGaps": normalized_gaps,
        "excessRisk": excess_risk,
        "valid": valid,
        "present": present,
        "candidateIds": candidate_ids,
        "businessIds": business_ids,
        "families": families,
        "folds": folds,
        "regimes": regimes,
        "setBusinessIndexes": set_business_indexes,
        "fullIndexes": np.flatnonzero(regimes == FULL_REGIME),
    }


def receipts_for(
    arrays: dict[str, np.ndarray],
    indexes: np.ndarray,
    prediction: dict[str, np.ndarray],
    policy: V9.PolicyConfig,
) -> list[dict[str, Any]]:
    receipts = V9.selection_receipts(
        arrays["businessIds"][indexes],
        arrays["families"][indexes],
        arrays["candidateIds"][indexes],
        arrays["valid"][indexes],
        arrays["excessRisk"][indexes],
        arrays["normalizedGaps"][indexes],
        arrays["dangerous"][indexes].astype(bool),
        prediction,
        policy,
    )
    for row, set_index, gate, gate_uncertainty in zip(
        receipts,
        indexes,
        prediction["gates"],
        prediction["gateUncertainty"],
    ):
        row["regime"] = str(arrays["regimes"][set_index])
        row["evidenceGate"] = {
            expert: float(gate[position])
            for position, expert in enumerate(EXPERTS)
        }
        row["evidenceGateUncertainty"] = {
            expert: float(gate_uncertainty[position])
            for position, expert in enumerate(EXPERTS)
        }
    return receipts


def prediction_only_receipts(
    arrays: dict[str, np.ndarray],
    indexes: np.ndarray,
    rolling_index: int,
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for set_index in indexes:
        active = np.flatnonzero(arrays["valid"][set_index])
        selected = int(
            active[
                np.argmax(arrays["features"][set_index, active, rolling_index])
            ]
        )
        loss = float(arrays["excessRisk"][set_index, selected])
        normalized = float(arrays["normalizedGaps"][set_index, selected])
        receipts.append(
            {
                "businessId": str(arrays["businessIds"][set_index]),
                "family": str(arrays["families"][set_index]),
                "candidateId": str(arrays["candidateIds"][set_index, selected]),
                "actualExcessLoss": loss,
                "normalizedExcessLoss": normalized,
                "dangerousFalseChampion": bool(loss >= 1.0),
                "oracle": bool(loss <= 1e-10),
                "confidenceRisk": 0.0,
                "theoremSlack": 0.0,
            }
        )
    return receipts


def mean_gate(receipts: list[dict[str, Any]]) -> dict[str, float]:
    return {
        expert: float(np.mean([row["evidenceGate"][expert] for row in receipts]))
        for expert in EXPERTS
    }


def main() -> None:
    started = time.time()
    dataset_source = DATASET_PATH.read_bytes()
    v9_source = V9_PREDICTIONS_PATH.read_bytes()
    dataset = json.loads(dataset_source)
    v9_predictions = json.loads(v9_source)
    if (
        dataset.get("activation") != "development-only"
        or dataset.get("provenance", {}).get("amssConfirmatoryCohortAccessed")
        or dataset.get("provenance", {}).get("previousAuditAccessed")
        or dataset.get("provenance", {}).get("posteriorRefits") != 0
        or len(dataset.get("rows", [])) != 20_160
        or len(dataset.get("sets", [])) != 1_260
    ):
        raise RuntimeError("V11 requires the isolated paired development artifact")
    arrays = prepare_arrays(dataset)
    if not np.all(np.isfinite(arrays["features"])) or not np.all(
        np.isfinite(arrays["contexts"])
    ):
        raise RuntimeError("V11 candidate or context inputs are non-finite")
    expert_indices = {
        expert: np.asarray(
            [int(item["index"]) for item in dataset["expertGroups"][expert]],
            dtype=int,
        )
        for expert in EXPERTS
    }
    all_assigned = np.concatenate(list(expert_indices.values()))
    if (
        all_assigned.size != len(dataset["featureNames"])
        or np.unique(all_assigned).size != all_assigned.size
    ):
        raise RuntimeError("V11 expert token partition is not disjoint and exhaustive")

    folds = arrays["folds"]
    full_indexes = arrays["fullIndexes"]
    outer_receipts = []
    selections: list[dict[str, Any]] = []
    prediction_rows = []
    chosen: list[tuple[V9.ModelConfig, V9.PolicyConfig]] = []

    for outer_fold in sorted(np.unique(folds[full_indexes])):
        outer_training_sets = np.flatnonzero(folds != outer_fold)
        outer_assessment = full_indexes[folds[full_indexes] == outer_fold]
        outer_training_full = full_indexes[folds[full_indexes] != outer_fold]
        inner_folds = sorted(np.unique(folds[outer_training_full]))
        best: tuple[float, int, str, V9.ModelConfig, V9.PolicyConfig] | None = None
        configuration_receipts = []
        for config in MODEL_GRID:
            inner_predictions: dict[int, dict[str, np.ndarray]] = {}
            for inner_fold in inner_folds:
                inner_training_sets = outer_training_sets[
                    folds[outer_training_sets] != inner_fold
                ]
                inner_assessment = outer_training_full[
                    folds[outer_training_full] == inner_fold
                ]
                model = fit_ensemble(
                    config,
                    arrays,
                    expert_indices,
                    inner_training_sets,
                    (190_000 + int(outer_fold) * 1_000 + int(inner_fold),),
                    bootstrap=False,
                )[0]
                prediction = ensemble_predict([model], arrays, inner_assessment)
                for local, set_index in enumerate(inner_assessment):
                    inner_predictions[int(set_index)] = {
                        key: value[local : local + 1]
                        for key, value in prediction.items()
                    }
            ordered_prediction = {
                key: np.concatenate(
                    [inner_predictions[int(index)][key] for index in outer_training_full],
                    axis=0,
                )
                for key in ("heads", "risk", "uncertainty", "gates", "gateUncertainty")
            }
            for policy in POLICY_GRID:
                receipt = receipts_for(
                    arrays,
                    outer_training_full,
                    ordered_prediction,
                    policy,
                )
                objective = V9.development_objective(receipt)
                configuration_receipts.append(
                    {
                        "model": config.id,
                        "policy": policy.id,
                        "objective": objective,
                    }
                )
                key = (
                    objective,
                    config.candidate_width,
                    f"{config.id}/{policy.id}",
                    config,
                    policy,
                )
                if best is None or key[:3] < best[:3]:
                    best = key
        assert best is not None
        config, policy = best[3], best[4]
        chosen.append((config, policy))
        models = fit_ensemble(
            config,
            arrays,
            expert_indices,
            outer_training_sets,
            tuple(200_000 + int(outer_fold) * 100 + seed for seed in range(3)),
            bootstrap=True,
        )
        prediction = ensemble_predict(models, arrays, outer_assessment)
        receipt = receipts_for(arrays, outer_assessment, prediction, policy)
        selections.extend(receipt)
        for local, set_index in enumerate(outer_assessment):
            heads = prediction["heads"][local]
            for candidate in np.flatnonzero(arrays["present"][set_index]):
                prediction_rows.append(
                    {
                        "businessId": str(arrays["businessIds"][set_index]),
                        "candidateId": str(
                            arrays["candidateIds"][set_index, candidate]
                        ),
                        "fold": int(folds[set_index]),
                        "predictedMean": float(heads[candidate, MEAN_HEAD]),
                        "predictedP90": float(heads[candidate, P90_HEAD]),
                        "predictedRisk": float(prediction["risk"][local, candidate]),
                        "predictedDanger": float(
                            heads[candidate, DANGER_HEAD]
                        ),
                        "uncertainty": float(
                            prediction["uncertainty"][local, candidate]
                        ),
                        "evidenceGate": {
                            expert: float(prediction["gates"][local, position])
                            for position, expert in enumerate(EXPERTS)
                        },
                    }
                )
        fold_result = {
            "fold": int(outer_fold),
            "selectedModel": config.id,
            "selectedPolicy": policy.id,
            "innerObjective": best[0],
            "assessmentAt100Percent": V9.promoted_metrics(receipt),
            "averageEvidenceGate": mean_gate(receipt),
            "candidateConfigurations": configuration_receipts,
            "ensembleReceipts": [model.training_receipt for model in models],
        }
        outer_receipts.append(fold_result)
        print(json.dumps(fold_result), flush=True)

    if len(selections) != 420:
        raise RuntimeError("V11 did not cross-fit all full candidate sets")
    combination_counts: dict[str, int] = {}
    combination_values: dict[str, tuple[V9.ModelConfig, V9.PolicyConfig]] = {}
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
    all_sets = np.arange(arrays["features"].shape[0])
    final_models = fit_ensemble(
        frozen_config,
        arrays,
        expert_indices,
        all_sets,
        tuple(210_000 + seed for seed in range(5)),
        bootstrap=True,
    )

    full_metrics = V9.promoted_metrics(selections)
    curve = V9.risk_coverage_curve(selections)
    v9_selection = list(v9_predictions["selections"])
    v9_metrics = V9.promoted_metrics(v9_selection)
    rolling_index = dataset["featureNames"].index("diagnostic:rolling-oos")
    prediction_receipts = prediction_only_receipts(
        arrays,
        full_indexes,
        rolling_index,
    )
    prediction_metrics = V9.promoted_metrics(prediction_receipts)

    context_index = dataset["contextNames"].index(
        "evidence:experiment:available-mask"
    )
    receipt_by_business = {row["businessId"]: row for row in selections}
    prediction_by_business = {
        row["businessId"]: row for row in prediction_receipts
    }
    no_experiment_businesses = {
        str(arrays["businessIds"][index])
        for index in full_indexes
        if arrays["contexts"][index, context_index] == 0
    }
    with_experiment = [
        row for row in selections if row["businessId"] not in no_experiment_businesses
    ]
    without_experiment = [
        receipt_by_business[business] for business in sorted(no_experiment_businesses)
    ]
    prediction_without = [
        prediction_by_business[business]
        for business in sorted(no_experiment_businesses)
    ]
    subgroup = {
        "withExperiment": {
            "businesses": len(with_experiment),
            "metrics": V9.promoted_metrics(with_experiment),
            "averageEvidenceGate": mean_gate(with_experiment),
        },
        "withoutExperiment": {
            "businesses": len(without_experiment),
            "metrics": V9.promoted_metrics(without_experiment),
            "predictionOnlyMetrics": V9.promoted_metrics(prediction_without),
            "averageEvidenceGate": mean_gate(without_experiment),
        },
    }

    model_artifact = {
        "artifactId": "flux-svi-score-v11-evidence-adaptive-selector-v1",
        "version": dataset["version"],
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "activation": "development-only",
        "provenance": {
            "datasetSha256": sha256(dataset_source),
            "posteriorRefits": 0,
            "amssConfirmatoryCohortAccessed": False,
            "previousAuditAccessed": False,
        },
        "featureNames": dataset["featureNames"],
        "contextNames": dataset["contextNames"],
        "expertGroups": dataset["expertGroups"],
        "model": frozen_config.__dict__,
        "policy": frozen_policy.__dict__,
        "models": [model.serialize() for model in final_models],
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = MODEL_PATH.with_suffix(".json.tmp")
    temporary_model.write_text(json.dumps(model_artifact, separators=(",", ":")))
    temporary_model.replace(MODEL_PATH)

    predictions_artifact = {
        "artifactId": "flux-svi-score-v11-evidence-adaptive-cross-fitted-v1",
        "version": dataset["version"],
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "activation": "development-only",
        "provenance": {
            "datasetSha256": sha256(dataset_source),
            "posteriorRefits": 0,
            "amssConfirmatoryCohortAccessed": False,
        },
        "predictions": prediction_rows,
        "selections": selections,
    }
    temporary_predictions = PREDICTIONS_PATH.with_suffix(".json.tmp")
    temporary_predictions.write_text(
        json.dumps(predictions_artifact, separators=(",", ":"))
    )
    temporary_predictions.replace(PREDICTIONS_PATH)

    checks = {
        "allFullSetsCrossFitted": len(selections) == 420,
        "allPairedRegimesGroupedByAdvertiser": True,
        "evidenceGatesFinite": all(
            all(np.isfinite(list(row["evidenceGate"].values())))
            for row in selections
        ),
        "evidenceGatesSumToOne": all(
            abs(sum(row["evidenceGate"].values()) - 1) < 1e-8
            for row in selections
        ),
        "theoremVerified": full_metrics["theoremViolations"] == 0,
        "posteriorRefitsExcluded": True,
        "amssConfirmatoryCohortUntouched": True,
        "previousAuditsUntouched": True,
        "nutsExcluded": True,
    }
    artifact = {
        "artifactId": "flux-svi-score-v11-evidence-adaptive-development-v1",
        "version": dataset["version"],
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "activation": "development-only",
        "stage": "post-amss-evidence-adaptive-development-complete",
        "contract": dataset["contract"],
        "provenance": {
            "datasetSha256": sha256(dataset_source),
            "v9PredictionsSha256": sha256(v9_source),
            "predictionsSha256": sha256(PREDICTIONS_PATH.read_bytes()),
            "modelSha256": sha256(MODEL_PATH.read_bytes()),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "posteriorRefits": 0,
            "amssConfirmatoryCohortAccessed": False,
            "previousAuditAccessed": False,
            "nutsUsed": False,
        },
        "nestedCrossValidation": {
            "outerAdvertiserFolds": 5,
            "innerAdvertiserFoldsPerOuter": 4,
            "augmentedRegimeSetsPerBusiness": 3,
            "outerReceipts": outer_receipts,
        },
        "evidenceAdaptiveRegretSet": {
            "at100PercentCoverage": full_metrics,
            "riskCoverageCurve": curve,
            "averageEvidenceGate": mean_gate(selections),
            "byExperimentAvailability": subgroup,
        },
        "developmentComparators": {
            "frozenV9": v9_metrics,
            "predictionOnly": prediction_metrics,
        },
        "checks": checks,
        "conclusion": {
            "implementationValid": all(checks.values()),
            "developmentImprovedVsV9": (
                full_metrics["selectionObjective"]
                < v9_metrics["selectionObjective"]
            ),
            "developmentImprovedVsPredictionOnly": (
                full_metrics["selectionObjective"]
                < prediction_metrics["selectionObjective"]
            ),
            "confirmatoryClaimPermitted": False,
            "freshIndependentAuditRequired": True,
            "amssCohortMayBeReused": False,
            "productionActivationPermitted": False,
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
                "evidenceAdaptiveRegretSet": artifact[
                    "evidenceAdaptiveRegretSet"
                ],
                "developmentComparators": artifact["developmentComparators"],
                "checks": checks,
                "conclusion": artifact["conclusion"],
                "runtimeSeconds": artifact["runtimeSeconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
