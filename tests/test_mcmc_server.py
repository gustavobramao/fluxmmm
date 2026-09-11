from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pymc as pm


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "flux_mcmc_server_tested",
    ROOT / "scripts" / "mcmc_server.py",
)
assert SPEC and SPEC.loader
MCMC = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MCMC
SPEC.loader.exec_module(MCMC)


def synthetic_contract(planning: bool = False) -> tuple[dict, float]:
    rng = np.random.default_rng(41)
    row_count = 64
    time = np.linspace(0, 1, row_count)
    planning_proxy = np.sin(2 * np.pi * time)
    spend = np.maximum(rng.gamma(2, 2, row_count) - 1, 0)
    carried: list[float] = []
    for value in spend:
        carried.append(value + 0.4 * (carried[-1] if carried else 0))
    carried_array = np.asarray(carried)
    half = float(np.median(carried_array[carried_array > 0]))
    transformed = carried_array**1.25 / (carried_array**1.25 + half**1.25)
    media_coefficient = 4.0
    outcome = (
        9
        + (0.8 * planning_proxy if planning else 0)
        + media_coefficient * transformed
        + rng.normal(0, 0.55, row_count)
    )
    columns = [np.ones(row_count)]
    names = ["Intercept"]
    priors = [
        {
            "kind": "normal",
            "mean": 9,
            "standardDeviation": 10,
            "initialValue": 9,
        }
    ]
    planning_index = None
    if planning:
        planning_index = len(columns)
        columns.append(planning_proxy)
        names.append("Planning intensity")
        priors.append(
            {
                "kind": "normal",
                "mean": 0,
                "standardDeviation": 3,
                "initialValue": 0.8,
            }
        )
    media_index = len(columns)
    columns.append(transformed)
    names.append("Paid search effect")
    priors.append(
        {
            "kind": "log-normal",
            "mean": 4,
            "standardDeviation": 2,
            "initialValue": 4,
        }
    )
    matrix = np.column_stack(columns)
    contract = {
        "matrix": matrix.tolist(),
        "target": outcome.tolist(),
        "outcome": outcome.tolist(),
        "parameterNames": names,
        "priors": priors,
        "media": [
            {
                "channel": "paid_search",
                "indexes": [media_index],
                "spend": float(spend.sum()),
                "rawSpend": spend.tolist(),
                "halfSaturation": half,
                "screeningRoi": 1.0,
                "screeningLow": 0.2,
                "screeningHigh": 3.0,
                "plausibleUpperRoi": 8.0,
                "roiWeights": [1.0],
            }
        ],
        "calibrations": [],
        "response": {
            "adstockType": "geometric",
            "adstockDecay": 0.4,
            "weibullShape": 2.5,
            "weibullScale": 4,
            "hillShape": 1.25,
            "timeVarying": False,
            "kernelKnots": 1,
            "kernelBandwidth": 0.18,
            "planningIntensity": planning,
            "planningParameterIndex": planning_index,
            "planningInitial": planning_proxy.tolist() if planning else [],
        },
        "likelihood": "gaussian",
        "studentTDegreesFreedom": 4,
        "screeningPredicted": outcome.tolist(),
        "dates": [f"2025-01-{(index % 28) + 1:02d}" for index in range(row_count)],
        "validationEligible": True,
        "promotedFingerprint": "synthetic",
        "promotedId": "synthetic",
        "version": MCMC.SAMPLING_CONTRACT_VERSION,
    }
    return contract, media_coefficient


class McmcContractTests(unittest.TestCase):
    def test_svi_cache_reuses_only_inference_equivalent_candidates(self) -> None:
        contract = {"method": "fullrank-advi"}
        model = {"promotedId": "C18-experiment", "matrix": [[1.0]], "target": [2.0]}
        equivalent = {**model, "promotedId": "C18-benchmark"}
        different = {**model, "target": [3.0]}
        with tempfile.TemporaryDirectory() as temporary_directory:
            previous_root = MCMC.SVI_ARTIFACT_ROOT
            MCMC.SVI_ARTIFACT_ROOT = Path(temporary_directory)
            try:
                artifact = MCMC.SVI_ARTIFACT_ROOT / "legacy"
                artifact.mkdir()
                (artifact / "payload.json").write_text(
                    json.dumps({"model": model, "contract": contract}),
                    encoding="utf-8",
                )
                (artifact / "summary.json").write_text(
                    json.dumps({"promotedId": model["promotedId"], "channels": []}),
                    encoding="utf-8",
                )
                reused = MCMC.cached_svi_result("0" * 64, equivalent, contract)
                self.assertIsNotNone(reused)
                self.assertTrue(reused["cacheEquivalent"])
                self.assertIsNone(
                    MCMC.cached_svi_result("1" * 64, different, contract)
                )
            finally:
                MCMC.SVI_ARTIFACT_ROOT = previous_root

    def test_v11_fullrank_advi_contract_is_exact_and_frozen(self) -> None:
        contract = {
            "method": "fullrank-advi",
            "iterations": 5000,
            "draws": 256,
            "primarySeeds": [30071, 81119],
            "adjudicationSeed": 190081,
            "learningRate": 0.001,
            "initialScale": 0.01,
            "gradientNorm": 10,
            "elboWindow": 250,
            "maximumElboDrift": 0.05,
            "maximumSeedLogRoiDifference": 0.25,
        }
        MCMC.validate_svi_contract(contract)
        changed = {**contract, "iterations": 4999}
        with self.assertRaisesRegex(ValueError, "frozen FullRankADVI"):
            MCMC.validate_svi_contract(changed)

    def test_stale_contract_fails_before_sampling(self) -> None:
        contract, _ = synthetic_contract()
        del contract["media"][0]["rawSpend"]
        with self.assertRaisesRegex(ValueError, "raw spend history is missing"):
            MCMC.validate_compiled_model(contract)

    def test_channel_specific_contract_builds_channel_response_parameters(self) -> None:
        contract, _ = synthetic_contract()
        contract["response"]["channelSpecific"] = True
        contract["media"][0]["response"] = {
            "adstockType": "geometric",
            "adstock": 0.4,
            "weibullShape": 2.5,
            "weibullScale": 4.0,
            "saturation": 1.25,
            "halfSaturationQuantile": 0.5,
            "kernelNormalization": "peak",
        }
        model, *_ = MCMC.build_model(contract)
        self.assertIn("adstock_decay_000", model.named_vars)
        self.assertIn("hill_shape_000", model.named_vars)
        self.assertNotIn("adstock_decay", model.named_vars)
        self.assertNotIn("hill_shape", model.named_vars)

    def test_channel_specific_contract_rejects_missing_channel_response(self) -> None:
        contract, _ = synthetic_contract()
        contract["response"]["channelSpecific"] = True
        with self.assertRaisesRegex(ValueError, "channel response contract"):
            MCMC.validate_compiled_model(contract)

    def test_experiment_calibration_uses_declared_spend_and_outcome_windows(self) -> None:
        contract, _ = synthetic_contract()
        spend_rows = list(range(10, 18))
        outcome_rows = list(range(10, 26))
        spend = float(
            np.asarray(contract["media"][0]["rawSpend"], dtype=float)[
                spend_rows
            ].sum()
        )
        calibration = {
            "label": "Paid search lift",
            "channel": "paid_search",
            "route": "prior",
            "basis": "experiment-window",
            "indexes": contract["media"][0]["indexes"],
            "weights": [1.0],
            "observedRoi": 1.4,
            "standardError": 0.3,
            "rows": outcome_rows,
            "spendRows": spend_rows,
            "outcomeRows": outcome_rows,
            "spend": spend,
        }
        contract["calibrations"] = [calibration]
        contract["media"][0]["priorEvidence"] = {
            "mean": 1.4,
            "standardDeviation": 0.3,
            "source": "experiment",
            "label": "Paid search lift",
            "rows": outcome_rows,
        }
        prior_model, *_ = MCMC.build_model(contract)
        self.assertIn("calibration_prior_000", prior_model.named_vars)
        self.assertNotIn("roi_prior_000", prior_model.named_vars)

        contract["calibrations"][0]["route"] = "likelihood"
        likelihood_model, *_ = MCMC.build_model(contract)
        self.assertIn("calibration_000", likelihood_model.named_vars)

    def test_hdi_is_shortest_interval_not_equal_tail_alias(self) -> None:
        values = np.random.default_rng(7).exponential(size=20_000)
        low, high = MCMC.hdi_bounds(values)
        equal_low, equal_high = np.quantile(values, [0.025, 0.975])
        self.assertLess(high - low, equal_high - equal_low)
        self.assertLess(low, equal_low)

    def test_posterior_predictive_contains_observation_noise(self) -> None:
        linear = np.zeros((4_000, 3))
        sigma = np.ones(4_000)
        draws = MCMC.posterior_predictive_draws(
            linear,
            sigma,
            "gaussian",
            4,
            np.random.default_rng(9),
        )
        self.assertGreater(float(draws.std()), 0.9)
        self.assertLess(float(draws.std()), 1.1)

    def test_decision_gates_reject_converged_but_unidentified_roi(self) -> None:
        metrics = MCMC.posterior_decision_metrics(
            [{"implausibleProbability": 0.35, "relativeIntervalWidth": 4.2}],
            np.asarray([1.0, 2.0]),
            np.asarray([0.0, 1.0]),
            np.asarray([2.0, 3.0]),
        )
        self.assertFalse(metrics["plausibilityPassed"])
        self.assertFalse(metrics["precisionPassed"])
        self.assertTrue(metrics["predictivePassed"])

    def test_nuts_recovers_media_and_samples_response_parameters(self) -> None:
        contract, true_coefficient = synthetic_contract()
        model, initial_values, *_ = MCMC.build_model(contract)
        with model:
            idata = pm.sample(
                draws=250,
                tune=300,
                chains=2,
                cores=1,
                random_seed=202603,
                initvals=initial_values,
                init="jitter+adapt_diag",
                progressbar=False,
                compute_convergence_checks=False,
                return_inferencedata=True,
                nuts={"target_accept": 0.95},
            )
        coefficient = np.asarray(idata.posterior["beta_001"]).reshape(-1)
        low, high = MCMC.hdi_bounds(coefficient)
        self.assertLess(low, true_coefficient)
        self.assertGreater(high, true_coefficient)
        self.assertEqual(int(np.asarray(idata.sample_stats.diverging).sum()), 0)
        self.assertIn("adstock_decay", idata.posterior)
        self.assertIn("hill_shape", idata.posterior)
        self.assertIn("media_contribution_000", idata.posterior)
        summary = MCMC.sampling_summary(
            MCMC.Job(id="synthetic", fingerprint="synthetic"),
            contract,
            {
                "chains": 2,
                "draws": 250,
                "tune": 300,
                "seed": 202603,
                "targetAccept": 0.95,
                "maxTreeDepth": 12,
            },
            "synthetic",
            idata,
            1.0,
        )
        gate_ids = {gate["id"] for gate in summary["gates"]}
        self.assertTrue(
            {"roi-plausibility", "roi-precision", "predictive-coverage"}.issubset(
                gate_ids
            )
        )
        predictive_width = np.mean(
            np.asarray(summary["predictive"]["high"])
            - np.asarray(summary["predictive"]["low"])
        )
        response_width = np.mean(
            np.asarray(summary["predictive"]["meanHigh"])
            - np.asarray(summary["predictive"]["meanLow"])
        )
        self.assertGreater(predictive_width, response_width)
        self.assertFalse(summary["inferenceContract"]["sharedPosterior"])
        self.assertGreater(len(summary["decisionDraws"]), 0)
        self.assertEqual(
            len(summary["decisionDraws"][0]["channels"]),
            len(contract["media"]),
        )
        self.assertIn("response", summary["decisionDraws"][0]["channels"][0])
        parameter_labels = {parameter["label"] for parameter in summary["parameters"]}
        self.assertIn("Geometric decay", parameter_labels)
        self.assertIn("Hill shape", parameter_labels)
        self.assertIn("Residual scale", parameter_labels)

    def test_planning_intensity_is_a_probabilistic_latent_path(self) -> None:
        contract, _ = synthetic_contract(planning=True)
        model, initial_values, *_ = MCMC.build_model(contract)
        self.assertIn("planning_deviation_weight", model.named_vars)
        self.assertIn("planning_factor", model.named_vars)
        self.assertNotIn("planning_proxy_measurement", model.named_vars)
        with model:
            idata = pm.sample(
                draws=100,
                tune=150,
                chains=2,
                cores=1,
                random_seed=202604,
                initvals=initial_values,
                init="jitter+adapt_diag",
                progressbar=False,
                compute_convergence_checks=False,
                return_inferencedata=True,
                nuts={"target_accept": 0.95},
            )
        self.assertEqual(int(np.asarray(idata.sample_stats.diverging).sum()), 0)


if __name__ == "__main__":
    unittest.main()
