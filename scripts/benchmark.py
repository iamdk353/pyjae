#!/usr/bin/env python
"""Honest denoising benchmark for JAE1.

Runs the non-gameable evaluation from ``pyjae.eval``: held-out test split, matched
latent dimension across methods, per-channel VAF, a mean-predictor floor, PCA and
Factor Analysis linear baselines, a matched-capacity denoising autoencoder, and
phase-shuffle / noise-only negative controls.

Expected outcome (this is the pass condition, and it is regime-specific):
  - NONLINEAR data: JAE1 > PCA and > FA, with the margin growing as noise rises.
  - LINEAR data: PCA / FA >= JAE1 (a JAE win here would be a red flag).
  - Every method beats the mean-predictor floor on real data.
  - Negative controls: JAE1 collapses to the mean-predictor floor (no false signal).

Usage:
  uv run python scripts/benchmark.py --quick
  uv run python scripts/benchmark.py            # fuller sweep
"""

import argparse

from pyjae.eval import aggregate, evaluate_cell, negative_control
from pyjae.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Small fast configuration.")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if args.quick:
        snrs = [0.0, 10.0]
        seeds = args.seeds or 2
        epochs = args.epochs or 150
        n_samples, n_channels, n_timepoints, latent_dim = 300, 48, 64, 6
    else:
        snrs = [-5.0, 0.0, 5.0, 10.0, 15.0]
        seeds = args.seeds or 3
        epochs = args.epochs or 250
        n_samples, n_channels, n_timepoints, latent_dim = 400, 64, 96, 6

    set_seed(0)
    results = []
    for nonlinear in (False, True):
        for snr in snrs:
            for seed in range(seeds):
                res = evaluate_cell(
                    snr_db=snr,
                    nonlinear=nonlinear,
                    seed=seed,
                    n_samples=n_samples,
                    n_channels=n_channels,
                    n_timepoints=n_timepoints,
                    latent_dim=latent_dim,
                    epochs=epochs,
                )
                results.append(res)
                tag = "nonlin" if nonlinear else "linear"
                s = res.scores
                print(
                    f"[{tag} snr={snr:+.0f} seed={seed}] "
                    f"noisy={s['noisy']:.3f} mean={s['mean']:.3f} "
                    f"pca={s['pca']:.3f} fa={s['fa']:.3f} dae={s['dae']:.3f} jae1={s['jae1']:.3f}",
                    flush=True,
                )

    print("\n===== AGGREGATE (mean +/- std over seeds) =====", flush=True)
    summary = aggregate(results)
    print(f"{'regime':>8} {'snr':>5}  {'pca':>14} {'fa':>14} {'dae':>14} {'jae1':>14}", flush=True)
    for (snr, nonlinear), bucket in sorted(summary.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        tag = "nonlin" if nonlinear else "linear"

        def cell(name):
            m, sd = bucket[name]
            return f"{m:.3f}+/-{sd:.3f}"

        winner = "JAE1" if bucket["jae1"][0] > max(bucket["pca"][0], bucket["fa"][0]) else "LINEAR"
        print(
            f"{tag:>8} {snr:>+5.0f}  {cell('pca'):>14} {cell('fa'):>14} "
            f"{cell('dae'):>14} {cell('jae1'):>14}   winner={winner}",
            flush=True,
        )

    print("\n===== NEGATIVE CONTROLS (JAE1 must not invent signal) =====", flush=True)
    ps = negative_control("phase_shuffle", seed=0, latent_dim=latent_dim, epochs=epochs)
    gap = ps["jae1"] - ps["mean"]
    flag = "OK" if gap < 0.05 else "SUSPICIOUS"
    print(
        f"  phase_shuffle: jae1={ps['jae1']:+.3f} mean_floor={ps['mean']:+.3f} "
        f"gap={gap:+.3f} [{flag}]  (JAE1 should sit at the mean floor)",
        flush=True,
    )
    no = negative_control("noise_only", seed=0, latent_dim=latent_dim, epochs=epochs)
    gap = no["jae1_recon"] - no["pca_recon"]
    flag = "OK" if gap < 0.05 else "SUSPICIOUS"
    print(
        f"     noise_only: jae1_recon={no['jae1_recon']:+.3f} pca_recon={no['pca_recon']:+.3f} "
        f"gap={gap:+.3f} [{flag}]  (JAE1 must not reconstruct noise better than PCA)",
        flush=True,
    )


if __name__ == "__main__":
    main()
