#!/usr/bin/env python3
"""Run one Flux FullRankADVI payload for unbiased offline score research."""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pymc as pm
import xarray as xr
from xarray import DataTree


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "flux_svi_batch_model",
    ROOT / "scripts" / "mcmc_server.py",
)
if not SPEC or not SPEC.loader:
    raise RuntimeError("Unable to load the Flux posterior model.")
MCMC = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MCMC
SPEC.loader.exec_module(MCMC)


def finite_number(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else fallback
    except (TypeError, ValueError):
        return fallback


def execution_backend_label(backend: str | None) -> str:
    return "cvm-retry" if backend == "c" else "numba-default"


def elbo_drift(history: np.ndarray, window: int) -> float:
    values = np.asarray(history, dtype=float)
    if values.size < 4 or not np.isfinite(values).all():
        return float("inf")
    width = max(2, min(int(window), values.size // 2))
    prior = float(np.median(values[-2 * width : -width]))
    recent = float(np.median(values[-width:]))
    return abs(recent - prior) / max(abs(recent), 1.0)


def initialize_full_rank_scale(inference: Any, scale: float) -> None:
    dimension = int(inference.approx.ddim)
    rows, columns = np.tril_indices(dimension)
    packed = np.zeros(len(rows), dtype=float)
    # FullRank stores diagonal standard deviations through softplus(rho).
    packed[rows == columns] = math.log(math.expm1(max(scale, 1e-6)))
    inference.approx.params_dict["L_tril"].set_value(packed)


def fit_seed(
    model: Any,
    initial_values: dict[str, Any],
    contract: dict[str, Any],
    seed: int,
    backend: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
    inference = pm.FullRankADVI(
        model=model,
        random_seed=seed,
        start=initial_values,
    )
    initialize_full_rank_scale(inference, finite_number(contract.get("initialScale"), 0.01))
    with model:
        approximation = inference.fit(
            n=int(contract["iterations"]),
            obj_optimizer=pm.adam(
                learning_rate=finite_number(contract.get("learningRate"), 0.001)
            ),
            total_grad_norm_constraint=finite_number(contract.get("gradientNorm"), 10),
            progressbar=False,
            backend=backend,
        )
        idata = approximation.sample(
            draws=int(contract["draws"]),
            random_seed=seed + 1_009,
            return_inferencedata=True,
        )
    history = np.asarray(approximation.hist, dtype=float)
    drift = elbo_drift(history, int(contract.get("elboWindow", 250)))
    return idata, {
        "seed": seed,
        "iterations": int(contract["iterations"]),
        "finalLoss": finite_number(np.median(history[-min(len(history), 50) :])),
        "elboDrift": finite_number(drift, 1e9),
        "finite": bool(np.isfinite(history).all()),
        "runtimeSeconds": time.perf_counter() - started,
    }


def roi_medians(idata: Any, compiled: dict[str, Any]) -> dict[str, float]:
    sigma = np.asarray(idata.posterior["sigma"].values, dtype=float).reshape(-1)
    linear = np.asarray(
        idata.posterior["linear_predictor"].values,
        dtype=float,
    ).reshape(len(sigma), -1)
    mean = np.asarray(
        idata.posterior["mean_outcome"].values,
        dtype=float,
    ).reshape(len(sigma), -1)
    result: dict[str, float] = {}
    for media_index, media in enumerate(compiled["media"]):
        contribution = np.asarray(
            idata.posterior[f"media_contribution_{media_index:03d}"].values,
            dtype=float,
        ).reshape(len(sigma), -1)
        spend = max(finite_number(media.get("spend"), 1.0), 1.0)
        if compiled["likelihood"] == "log-normal":
            without = np.exp(
                np.clip(
                    linear - contribution + 0.5 * sigma[:, None] ** 2,
                    -30,
                    30,
                )
            )
            roi = np.sum(mean - without, axis=1) / spend
        else:
            roi = np.sum(contribution, axis=1) / spend
        result[str(media["channel"])] = float(np.median(roi))
    return result


def maximum_seed_difference(seed_rois: list[dict[str, float]]) -> float:
    if len(seed_rois) < 2:
        return float("inf")
    channels = sorted(set.intersection(*(set(item) for item in seed_rois)))
    differences: list[float] = []
    for channel in channels:
        transformed = np.log1p(
            np.maximum([finite_number(item[channel]) for item in seed_rois], 0)
        )
        center = float(np.median(transformed))
        differences.extend(abs(float(value) - center) for value in transformed)
    return max(differences, default=float("inf"))


def combined_idata(idatas: list[Any]) -> Any:
    posterior = xr.concat(
        [idata.posterior.to_dataset() for idata in idatas],
        dim="chain",
    )
    chain_count = int(posterior.sizes["chain"])
    draw_count = int(posterior.sizes["draw"])
    # The production summary function needs sample_stats to construct its
    # display payload. These placeholders are never presented as NUTS
    # diagnostics and are replaced by explicit SVI diagnostics below.
    energy = np.tile(np.linspace(-1, 1, draw_count), (chain_count, 1))
    sample_stats = xr.Dataset(
        {
            "diverging": (
                ("chain", "draw"),
                np.zeros((chain_count, draw_count), dtype=bool),
            ),
            "tree_depth": (
                ("chain", "draw"),
                np.zeros((chain_count, draw_count), dtype=int),
            ),
            "energy": (("chain", "draw"), energy),
        }
    )
    return DataTree.from_dict({"/posterior": posterior, "/sample_stats": sample_stats})


def run(payload: dict[str, Any], backend: str | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    fingerprint = str(payload["fingerprint"])
    compiled = payload["model"]
    contract = payload["contract"]
    model, initial_values, _, _, _ = MCMC.build_model(compiled)
    idatas: list[Any] = []
    diagnostics: list[dict[str, Any]] = []
    for seed in contract["primarySeeds"]:
        idata, diagnostic = fit_seed(
            model,
            initial_values,
            contract,
            int(seed),
            backend,
        )
        idatas.append(idata)
        diagnostics.append(diagnostic)

    seed_rois = [roi_medians(idata, compiled) for idata in idatas]
    maximum_difference = maximum_seed_difference(seed_rois)
    maximum_drift = max(item["elboDrift"] for item in diagnostics)
    needs_adjudication = (
        maximum_difference > finite_number(contract["maximumSeedLogRoiDifference"], 0.25)
        or maximum_drift > finite_number(contract["maximumElboDrift"], 0.05)
    )
    if needs_adjudication:
        idata, diagnostic = fit_seed(
            model,
            initial_values,
            contract,
            int(contract["adjudicationSeed"]),
            backend,
        )
        idatas.append(idata)
        diagnostics.append(diagnostic)
        seed_rois.append(roi_medians(idata, compiled))
        maximum_difference = maximum_seed_difference(seed_rois)
        maximum_drift = max(item["elboDrift"] for item in diagnostics)

    idata = combined_idata(idatas)
    summary_contract = {
        "preset": "custom",
        "chains": len(idatas),
        "tune": int(contract["iterations"]),
        "draws": int(contract["draws"]),
        "targetAccept": 0.9,
        "maxTreeDepth": 10,
        "seed": int(contract["primarySeeds"][0]),
    }
    summary = MCMC.sampling_summary(
        MCMC.Job(id=fingerprint, fingerprint=fingerprint),
        compiled,
        summary_contract,
        fingerprint,
        idata,
        time.perf_counter() - started,
    )
    finite = bool(
        all(item["finite"] for item in diagnostics)
        and summary.get("decisionDraws")
        and all(
            math.isfinite(finite_number(channel.get("posteriorMedian"), float("nan")))
            for channel in summary["channels"]
        )
    )
    elbo_stable = maximum_drift <= finite_number(contract["maximumElboDrift"], 0.05)
    seed_agreement = maximum_difference <= finite_number(
        contract["maximumSeedLogRoiDifference"], 0.25
    )
    diagnostics_payload = {
        "finite": finite,
        "elboStable": elbo_stable,
        "maximumElboDrift": maximum_drift,
        "seedAgreement": seed_agreement,
        "maximumSeedLogRoiDifference": maximum_difference,
        "adjudicationUsed": needs_adjudication,
        "predictiveCoverage": finite_number(
            summary["diagnostics"].get("predictiveCoverage")
        ),
        "maximumImplausibleProbability": finite_number(
            summary["diagnostics"].get("maxImplausibleProbability")
        ),
        "maximumRelativeRoiWidth": finite_number(
            summary["diagnostics"].get("maxRelativeRoiWidth")
        ),
    }
    retained_draws = summary["decisionDraws"]
    if len(retained_draws) > 128:
        retained_draws = [
            retained_draws[
                min(
                    len(retained_draws) - 1,
                    int(((index + 0.5) / 128) * len(retained_draws)),
                )
            ]
            for index in range(128)
        ]
    compact_channels = []
    for channel in summary["channels"]:
        compact = dict(channel)
        samples = compact.get("posteriorSamples") or []
        if len(samples) > 128:
            compact["posteriorSamples"] = [
                samples[
                    min(
                        len(samples) - 1,
                        int(((index + 0.5) / 128) * len(samples)),
                    )
                ]
                for index in range(128)
            ]
        compact_channels.append(compact)
    return {
        "kind": "svi",
        "fingerprint": fingerprint,
        "promotedFingerprint": compiled["promotedFingerprint"],
        "promotedId": compiled["promotedId"],
        "engine": "PyMC 6.2 · FullRankADVI",
        "executionBackend": execution_backend_label(backend),
        "contract": contract,
        "status": "labelled" if finite and elbo_stable and seed_agreement else "review",
        "diagnostics": diagnostics_payload,
        "seeds": diagnostics,
        "channels": compact_channels,
        "decisionDraws": retained_draws,
        "runtimeSeconds": time.perf_counter() - started,
        "runAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--backend",
        choices=("default", "c"),
        default="default",
        help="PyTensor execution backend. The C/VM backend is the deterministic fallback for Numba compilation limits.",
    )
    args = parser.parse_args()
    payload_path = Path(args.payload).resolve()
    output_path = Path(args.output).resolve()
    if payload_path.suffix == ".gz":
        with gzip.open(payload_path, "rt", encoding="utf-8") as source:
            payload = json.load(source)
    else:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    result = run(payload, None if args.backend == "default" else args.backend)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".gz":
        with gzip.open(output_path, "wt", encoding="utf-8") as destination:
            json.dump(result, destination)
    else:
        output_path.write_text(json.dumps(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "fingerprint": result["fingerprint"],
                "status": result["status"],
                "runtimeSeconds": result["runtimeSeconds"],
                "diagnostics": result["diagnostics"],
            }
        )
    )


if __name__ == "__main__":
    main()
