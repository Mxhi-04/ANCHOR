"""Active-learning replay on literature data (no new experiments).

Two questions, both asked one held-out system at a time, starting from a
model trained only on the other systems:

1. Learning curve: if each "experiment" reveals one measured row, how fast
   does error on the rest of the system fall when ANCHOR picks the most
   uncertain row, compared with random picks?
2. Trials to target: a formulator wants Tg within +/-5 K of a target. How
   many rows must be "made" before one hits it, when ANCHOR picks the row most
   likely to hit, compared with Fox-guided and random picking?

Every row's measured Tg serves once as the target, so no target is
hand-picked. Rules fixed before running: tolerance 5 K, 20 random seeds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data import weights
from .models import Anchor
from .physics import fox_tg

TOLERANCE_K = 5.0
SEEDS = range(20)


def _fit_predict(w_tr, y_tr, w_pool, tg_homo):
    m = Anchor(tg_homo).fit(w_tr, y_tr)
    return m.predict(w_pool, return_std=True)


def learning_curves(df: pd.DataFrame, tg_homo: np.ndarray) -> pd.DataFrame:
    rows = []
    for system in sorted(df["system"].unique()):
        base = df[df["system"] != system]
        pool = df[df["system"] == system]
        w_b, y_b = weights(base), base["Tg_exp_K"].to_numpy()
        w_p, y_p = weights(pool), pool["Tg_exp_K"].to_numpy()
        n = len(pool)
        strategies = {"ANCHOR (most uncertain)": [-1], "Random": list(SEEDS)}
        for strategy, seeds in strategies.items():
            for seed in seeds:
                rng = np.random.default_rng(max(seed, 0))
                taken: list[int] = []
                for step in range(n):
                    w_tr = np.vstack([w_b, w_p[taken]]) if taken else w_b
                    y_tr = np.concatenate([y_b, y_p[taken]]) if taken else y_b
                    rest = [i for i in range(n) if i not in taken]
                    mu, sd = _fit_predict(w_tr, y_tr, w_p[rest], tg_homo)
                    rows.append({"system": system, "strategy": strategy, "seed": seed, "measured": step,
                                 "MAE_K": float(np.mean(np.abs(mu - y_p[rest])))})
                    if step == n - 1:
                        break
                    pick = rest[int(np.argmax(sd))] if seed < 0 else int(rng.choice(rest))
                    taken.append(pick)
    return pd.DataFrame(rows)


def trials_to_target(df: pd.DataFrame, tg_homo: np.ndarray) -> pd.DataFrame:
    rows = []
    for system in sorted(df["system"].unique()):
        base = df[df["system"] != system]
        pool = df[df["system"] == system]
        w_b, y_b = weights(base), base["Tg_exp_K"].to_numpy()
        w_p, y_p = weights(pool), pool["Tg_exp_K"].to_numpy()
        fox_p = fox_tg(w_p, tg_homo)
        n = len(pool)
        for t_idx in range(n):
            target = y_p[t_idx]
            hit = np.abs(y_p - target) <= TOLERANCE_K

            # ANCHOR: refit after every result, pick the row most likely to hit.
            taken: list[int] = []
            while True:
                rest = [i for i in range(n) if i not in taken]
                w_tr = np.vstack([w_b, w_p[taken]]) if taken else w_b
                y_tr = np.concatenate([y_b, y_p[taken]]) if taken else y_b
                mu, sd = _fit_predict(w_tr, y_tr, w_p[rest], tg_homo)
                p_hit = norm.cdf((target + TOLERANCE_K - mu) / sd) - norm.cdf((target - TOLERANCE_K - mu) / sd)
                taken.append(rest[int(np.argmax(p_hit))])
                if hit[taken[-1]]:
                    break
            rows.append({"system": system, "target_row": t_idx, "strategy": "ANCHOR", "seed": -1, "trials": len(taken)})

            # Fox: rank rows by |Fox - target|, no learning.
            order = np.argsort(np.abs(fox_p - target), kind="stable")
            rows.append({"system": system, "target_row": t_idx, "strategy": "Fox", "seed": -1,
                         "trials": int(np.argmax(hit[order]) + 1)})

            for seed in SEEDS:
                order = np.random.default_rng(seed).permutation(n)
                rows.append({"system": system, "target_row": t_idx, "strategy": "Random", "seed": seed,
                             "trials": int(np.argmax(hit[order]) + 1)})
    return pd.DataFrame(rows)


def summarise_trials(trials: pd.DataFrame) -> dict:
    out = {}
    for strategy, g in trials.groupby("strategy"):
        out[strategy] = {"mean_trials": float(g["trials"].mean()),
                         "median_trials": float(g["trials"].median()),
                         "first_try_hit_rate": float((g["trials"] == 1).mean())}
    return out
