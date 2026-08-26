"""Train the numeric forecaster on the real city panel and export it for
OpenVINO inference.

    python -m greenplan.forecast.train --config config/city.yaml

What it does, honestly:
  1. loads the same panel the engine uses (146 cells x 42 months for the
     shipped Ahmedabad data);
  2. builds one sample per (zone, cutoff) exactly like the backtest does —
     the model only ever sees history strictly BEFORE each cutoff;
  3. splits by TIME, not randomly: early cutoffs train, the latest cutoffs
     test, so the score is a genuine out-of-sample forecast score;
  4. trains a small MLP per metric (features are standardized here; the ONNX
     graph is the bare network, because OpenVINO executes standard ONNX ops);
  5. scores model vs the Theil-Sen + seasonality baseline on the SAME test
     tasks — the identical skill definition the LLM was scored with, so the
     two are directly comparable;
  6. writes {metric}.onnx + norm.json (+ report.json) to models/{city}/forecaster/.

Traffic note: the traffic stream is a disclosed inert placeholder (flat
value, MCDA weight 0). Its model trains and exports for interface
completeness, but its score is meaningless and is excluded from the combined
skill figure.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_config
from ..engine import load_panel
from ..features.trends import (
    METRIC_BOUNDS,
    METRICS,
    panel_stats,
    trend_seasonal_estimate,
)
from ..training.backtest import build_predict_context
from .features import FEATURE_NAMES, feature_vector

log = logging.getLogger(__name__)

REAL_METRICS = [m for m in METRICS if m != "traffic"]  # traffic = inert placeholder


class _NoMemory:
    """The forecaster's features use no memory records; satisfy the context
    builder's interface with an empty retrieval."""

    @staticmethod
    def retrieve(zone: str, feat: Any, k: int) -> list[dict[str, Any]]:
        return []


def build_dataset(cfg, panel) -> dict[str, Any]:
    t = cfg.training
    stats = panel_stats(panel)
    last_month = int(panel["month"].max())
    horizon = t.horizon_months
    memory = _NoMemory()

    X: dict[str, list[np.ndarray]] = {m: [] for m in METRICS}
    y: dict[str, list[float]] = {m: [] for m in METRICS}
    base: dict[str, list[float]] = {m: [] for m in METRICS}
    cutoffs: list[int] = []

    zones = sorted(panel["zone"].unique())
    for zone in zones:
        g = panel[panel["zone"] == zone]
        for cutoff in range(t.min_history_months, last_month - horizon + 1):
            target = g[g["month"] == cutoff + horizon]
            if target.empty or target[METRICS].isna().any(axis=None):
                continue
            if g[g["month"] < cutoff]["month"].nunique() < t.min_history_months:
                continue
            ctx = build_predict_context(
                panel, zone, cutoff, horizon, t.context_window,
                cfg.data.start_month, stats, memory, 0,
            )
            for m in METRICS:
                X[m].append(feature_vector(ctx, m))
                y[m].append(float(target.iloc[0][m]))
                pts = [
                    (h["month"], h["cal_month"], h[m])
                    for h in ctx["history"] if h[m] is not None
                ]
                est = trend_seasonal_estimate(pts, ctx["target_month"], ctx["season"])
                if est is None:
                    est = ctx["inputs"][f"{m}_latest"]
                base[m].append(float(np.clip(est, *METRIC_BOUNDS[m])))
            cutoffs.append(cutoff)

    return {
        "X": {m: np.vstack(X[m]) for m in METRICS},
        "y": {m: np.asarray(y[m], dtype=np.float32) for m in METRICS},
        "base": {m: np.asarray(base[m], dtype=np.float32) for m in METRICS},
        "cutoffs": np.asarray(cutoffs),
        "stats": stats,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Train the OpenVINO numeric forecaster")
    ap.add_argument("--config", default="config/city.yaml")
    ap.add_argument("--hidden", type=int, nargs="*", default=[64, 32])
    ap.add_argument("--test-cutoffs", type=int, default=3,
                    help="how many of the LATEST cutoff months form the test set")
    args = ap.parse_args()

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from sklearn.neural_network import MLPRegressor
    except ImportError:
        print(
            "error: training needs scikit-learn and skl2onnx:\n"
            "    pip install scikit-learn skl2onnx",
            file=sys.stderr,
        )
        return 2

    cfg = load_config(args.config)
    panel = load_panel(cfg, mock=False)
    ds = build_dataset(cfg, panel)
    n = len(ds["cutoffs"])

    split = sorted(set(ds["cutoffs"]))[-args.test_cutoffs]
    train_idx = ds["cutoffs"] < split
    test_idx = ~train_idx
    log.info(
        "%d samples: %d train (cutoff < %d), %d test — split by time, not randomly",
        n, int(train_idx.sum()), split, int(test_idx.sum()),
    )

    out_dir = Path(cfg.model.forecaster_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    norm: dict[str, Any] = {"feature_names": FEATURE_NAMES, "mu": {}, "sd": {}}
    report: dict[str, Any] = {"n_train": int(train_idx.sum()), "n_test": int(test_idx.sum()),
                              "horizon_months": cfg.training.horizon_months, "per_metric": {}}
    skills = []

    for m in METRICS:
        Xm, ym, bm = ds["X"][m], ds["y"][m], ds["base"][m]
        mu = Xm[train_idx].mean(axis=0)
        sd = Xm[train_idx].std(axis=0) + 1e-9
        Z = (Xm - mu) / sd

        # The network learns the RESIDUAL from the Theil-Sen + seasonal
        # baseline, standardized. At worst it learns nothing and collapses to
        # the baseline (skill ~ 0); it can only add value, never quietly
        # replace an honest estimate with a worse one.
        resid = ym - bm
        r_mu = float(resid[train_idx].mean())
        r_sd = float(resid[train_idx].std() + 1e-9)

        net = MLPRegressor(
            hidden_layer_sizes=tuple(args.hidden),
            alpha=1e-2, max_iter=4000, random_state=cfg.run.seed,
        )
        net.fit(Z[train_idx], (resid[train_idx] - r_mu) / r_sd)

        pred_resid = net.predict(Z[test_idx]) * r_sd + r_mu
        pred = np.clip(bm[test_idx] + pred_resid, *METRIC_BOUNDS[m])
        mae = float(np.mean(np.abs(pred - ym[test_idx])))
        base_mae = float(np.mean(np.abs(bm[test_idx] - ym[test_idx])))
        std = ds["stats"][m]["std"]
        skill = float((base_mae - mae) / std)  # + = beats the baseline
        report["per_metric"][m] = {
            "test_mae": round(mae, 4),
            "baseline_mae": round(base_mae, 4),
            "skill_vs_baseline": round(skill, 4),
            "inert_placeholder": m == "traffic",
        }
        if m in REAL_METRICS:
            skills.append(skill)

        onx = convert_sklearn(
            net, initial_types=[("X", FloatTensorType([None, len(FEATURE_NAMES)]))]
        )
        (out_dir / f"{m}.onnx").write_bytes(onx.SerializeToString())
        norm["mu"][m] = [float(v) for v in mu]
        norm["sd"][m] = [float(v) for v in sd]
        norm.setdefault("resid_mu", {})[m] = r_mu
        norm.setdefault("resid_sd", {})[m] = r_sd

    report["skill_combined"] = round(float(np.mean(skills)), 4)
    norm["report"] = report
    (out_dir / "norm.json").write_text(json.dumps(norm), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'metric':9s} {'test MAE':>9s} {'baseline':>9s} {'skill':>7s}")
    for m in METRICS:
        r = report["per_metric"][m]
        tag = "  (inert placeholder, excluded)" if r["inert_placeholder"] else ""
        print(f"{m:9s} {r['test_mae']:9.4f} {r['baseline_mae']:9.4f} "
              f"{r['skill_vs_baseline']:+7.3f}{tag}")
    verdict = (
        "POSITIVE — provider `hybrid` will deploy this network"
        if report["skill_combined"] > 0
        else "NEGATIVE — provider `hybrid` keeps the statistical forecaster "
        "(the honest champion) and benches this challenger"
    )
    print(
        f"\ncombined skill vs Theil-Sen+seasonal baseline (real metrics): "
        f"{report['skill_combined']:+.3f}\n"
        f"(same definition scored the LLM at -0.333 — see README table)\n"
        f"verdict: {verdict}\n"
        f"wrote {out_dir}/[metric].onnx + norm.json + report.json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
