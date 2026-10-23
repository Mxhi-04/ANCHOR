"""Rebuild every ANCHOR output from data/:  python -m anchor.run"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import active, report, validate
from .data import MONOMERS, load_excluded, load_homopolymers, load_training


def main() -> dict:
    t0 = time.time()
    train, excluded = load_training(), load_excluded()
    tg_map = load_homopolymers()
    tg_homo = np.array([tg_map[m] for m in MONOMERS])
    report.OUT.mkdir(exist_ok=True)

    print(f"Data: {len(train)} rows, {train['system'].nunique()} systems, {train['source_id'].nunique()} sources")
    pred, curves = validate.logo_predictions(train, tg_homo)
    summary = validate.summarise(pred)
    by_system = validate.per_system(pred)
    gt = validate.gordon_taylor_by_system(train, tg_map)
    external = validate.external_test(train, excluded, tg_homo)
    relevance = validate.feature_relevance(train, tg_homo)
    print("Held-out validation done")

    lc = active.learning_curves(train, tg_homo)
    trials = active.trials_to_target(train, tg_homo)
    trial_summary = active.summarise_trials(trials)
    print("Active-learning replay done")

    metrics = {
        "data": {"n_rows": int(len(train)), "n_systems": int(train["system"].nunique()),
                 "n_sources": int(train["source_id"].nunique()),
                 "systems": train["system"].value_counts().sort_index().to_dict()},
        "validation": "leave-one-monomer-system-out; CV+ (K=5) normalised conformal intervals",
        "held_out": summary,
        "calibration": {name: {str(k): v for k, v in c.items()} for name, c in curves.items()},
        "per_system": by_system.to_dict(orient="records"),
        "gordon_taylor_in_sample": gt.to_dict(orient="records"),
        "external": external,
        "anchor_length_scales": relevance,
        "tolerance_K": active.TOLERANCE_K,
        "trials_to_target": trial_summary,
        "homopolymer_tg_K": tg_map,
    }

    pred.to_csv(report.OUT / "predictions.csv", index=False)
    report.write_json(report.OUT / "metrics.json", metrics)
    report.write_metrics_md(report.OUT / "metrics.md", metrics)

    report.parity(pred, summary)
    report.calibration(curves)
    report.trials_figure(lc, trials, trial_summary)
    report.residuals_figure(by_system, gt)
    report.deck_figures(pred, summary, curves, trial_summary)
    report.gif_hold_out(pred)
    biggest_single_source = (train.groupby("system")["source_id"].nunique() == 1) & (train.groupby("system").size() >= 6)
    report.gif_learning(train, tg_homo, biggest_single_source[biggest_single_source].index[0])

    readme = [
        ("What this is", "ANCHOR results: physics-anchored, uncertainty-aware copolymer Tg prediction."),
        ("Reproduce", "python -m anchor.run  (from the repo root, with src/ on PYTHONPATH)"),
        ("data_training", "The 39 measured rows Mahi cleared for training, with canonical system labels."),
        ("data_excluded", "12 Taheri resins (acrylic acid not measured; low Mn): external stress test only."),
        ("homopolymers", "Homopolymer Tg used by the Fox prior, all from one curated experimental dataset."),
        ("predictions", "Held-out (leave-one-system-out) predictions and 90% intervals for every row."),
        ("metrics", "Overall held-out metrics per model."),
        ("per_system", "Held-out error per monomer system, and Fox bias (measured - Fox)."),
        ("gordon_taylor", "Gordon-Taylor k fitted within each system (in-sample, descriptive)."),
        ("calibration", "Observed coverage at each nominal confidence level."),
        ("trials", "Active-learning replay: trials to hit a target Tg within +/-5 K."),
        ("learning_curves", "Active-learning replay: MAE on unmeasured rows vs rows measured."),
        ("sources", "Citations for every source_id."),
        ("Note", "All numbers come from literature measurements; no values were assumed or generated."),
    ]
    flat = [{"model": k, **v} for k, v in summary.items()]
    cal = [{"model": n, "nominal": float(k), "observed": v} for n, c in curves.items() for k, v in c.items()]
    report.write_excel(report.OUT / "results.xlsx", {
        "data_training": train, "data_excluded": excluded,
        "homopolymers": pd.read_csv(report.ROOT / "data" / "homopolymer_tg.csv"),
        "predictions": pred, "metrics": pd.DataFrame(flat), "per_system": by_system,
        "gordon_taylor": gt, "calibration": pd.DataFrame(cal), "trials": trials,
        "learning_curves": lc, "sources": report.sources_table(),
    }, readme)

    print(open(report.OUT / "metrics.md").read())
    print(f"Done in {time.time() - t0:.0f} s")
    return metrics


if __name__ == "__main__":
    main()
