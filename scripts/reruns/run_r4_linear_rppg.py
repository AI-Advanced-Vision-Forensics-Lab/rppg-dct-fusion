#!/usr/bin/env python3
"""
run_r4_linear_rppg.py — R4: rPPG-only using FusionMLP's 10K linear encoder

The FusionMLP rPPG branch is: Linear(160→64) → BN → ReLU → Dropout(0.3) → Linear(64→1)
This is a standalone rPPG classifier that matches the rPPG branch used inside
all fusion models in Table I.

Runs both:
  R4-iso:  isolated per-method (same as Table I setup but using the linear encoder)
  R4-comb: combined all 7 methods (same as R2 but linear encoder)

Purpose: establishes a fair baseline for the rPPG contribution inside fusion
(the rPPG-only row in Table I currently uses ResNet1D, not the linear encoder
that's actually inside the fusion models).

Output: <out-dir>/r4_linear_rppg/
  config.json, r4_iso_results.json, r4_comb_results.json, run.log

Writes only to <out-dir>.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (SPLIT_FILE, DEVICE, SEEDS, METHODS,
                    LinearRPPG, load_waveforms, build_arrays,
                    run_experiment, make_config, DCT_FILE)

from common import OUT_ROOT
OUT_DIR = OUT_ROOT / "r4_linear_rppg"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "run.log"


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    t0 = time.time()
    log(f"=== R4: rPPG-only LinearRPPG (10K, dropout=0.3) | device={DEVICE} ===")

    df = pd.read_csv(SPLIT_FILE)
    real_df  = df[df['class'] == 'real']
    tr_real  = real_df[real_df['split'] == 'train']
    ev_real  = real_df[real_df['split'].isin(['val', 'test'])]

    train_ids = set(tr_real['identity'].unique())
    eval_ids  = set(ev_real['identity'].unique())
    assert len(tr_real) == 1675
    assert len(ev_real) == 696
    assert len(train_ids & eval_ids) == 0
    log("Assertions PASSED")

    log("Preloading waveforms...")
    wave_cache = load_waveforms(df['video_id'].tolist())
    loaded = sum(1 for v in wave_cache.values() if v is not None)
    log(f"Loaded {loaded} / {len(df)} waveforms")

    tr_real_w, _, tr_real_l, _ = build_arrays(tr_real, wave_cache, {})
    ev_real_w, _, ev_real_l, _ = build_arrays(ev_real, wave_cache, {})

    # ── R4 isolated ───────────────────────────────────────────────────────────
    log("\n=== R4-iso: Isolated per-method LinearRPPG ===")
    r4_iso = {}
    for method in METHODS:
        t_m = time.time()
        fake_mask = df['video_id'].str.startswith(method + '__')
        tr_fk_df = df[fake_mask & (df['split'] == 'train')]
        ev_fk_df = df[fake_mask & df['split'].isin(['val', 'test'])]

        tr_fk_w, _, tr_fk_l, _ = build_arrays(tr_fk_df, wave_cache, {})
        ev_fk_w, _, ev_fk_l, _ = build_arrays(ev_fk_df, wave_cache, {})

        wave_tr  = np.concatenate([tr_fk_w, tr_real_w])
        label_tr = np.concatenate([tr_fk_l, tr_real_l])
        wave_ev  = np.concatenate([ev_fk_w, ev_real_w])
        label_ev = np.concatenate([ev_fk_l, ev_real_l])

        dct_dummy_tr = np.zeros((len(wave_tr), 64), dtype=np.float32)
        dct_dummy_ev = np.zeros((len(wave_ev), 64), dtype=np.float32)

        result = run_experiment(
            LinearRPPG, {"dropout": 0.3},
            wave_tr, dct_dummy_tr, label_tr,
            wave_ev, dct_dummy_ev, label_ev,
            SEEDS, use_dct=False,
            lr=1e-3, wd=5e-4, epochs=30, batch=64)

        result.update({
            "n_train_real": len(tr_real_w), "n_train_fake": int(label_tr.sum()),
            "n_eval_real": len(ev_real_w), "n_eval_fake": int(label_ev.sum()),
            "model_params": LinearRPPG.param_count(),
        })
        r4_iso[method] = result
        log(f"  [{method}] AUC={result['mean_auc']:.4f}±{result['std_auc']:.4f}  ({time.time()-t_m:.0f}s)")

    # ── R4 combined ───────────────────────────────────────────────────────────
    log("\n=== R4-comb: Combined LinearRPPG (all 7 methods) ===")
    t_m = time.time()
    all_fake_df  = df[df['class'] == 'fake']
    tr_all_fk_df = all_fake_df[all_fake_df['split'] == 'train']
    ev_all_fk_df = all_fake_df[all_fake_df['split'].isin(['val', 'test'])]

    tr_fk_w, _, tr_fk_l, _ = build_arrays(tr_all_fk_df, wave_cache, {})
    ev_fk_w, _, ev_fk_l, _ = build_arrays(ev_all_fk_df, wave_cache, {})

    wave_tr  = np.concatenate([tr_fk_w, tr_real_w])
    label_tr = np.concatenate([tr_fk_l, tr_real_l])
    wave_ev  = np.concatenate([ev_fk_w, ev_real_w])
    label_ev = np.concatenate([ev_fk_l, ev_real_l])

    dct_dummy_tr = np.zeros((len(wave_tr), 64), dtype=np.float32)
    dct_dummy_ev = np.zeros((len(wave_ev), 64), dtype=np.float32)

    r4_comb = run_experiment(
        LinearRPPG, {"dropout": 0.3},
        wave_tr, dct_dummy_tr, label_tr,
        wave_ev, dct_dummy_ev, label_ev,
        SEEDS, use_dct=False,
        lr=1e-3, wd=5e-4, epochs=30, batch=64)

    r4_comb.update({
        "n_train_real": len(tr_real_w), "n_train_fake": int(label_tr.sum()),
        "n_eval_real": len(ev_real_w), "n_eval_fake": int(label_ev.sum()),
        "model_params": LinearRPPG.param_count(),
    })
    log(f"  Combined AUC={r4_comb['mean_auc']:.4f}±{r4_comb['std_auc']:.4f}  ({time.time()-t_m:.0f}s)")

    # ── Save ──────────────────────────────────────────────────────────────────
    config = make_config(
        "LinearRPPG",
        {"dropout": 0.3},
        {"n_train_real": 1675, "n_eval_real": 696},
        extra={
            "run_id": "R4",
            "arch": "Linear(160,64)→BN→ReLU→Dropout(0.3)→Linear(64,1) — 10,432 params",
            "note": "Same rPPG arch as FusionMLP.rppg_enc. Provides fair rPPG-only "
                    "baseline matching the rPPG branch inside the fusion models.",
        }
    )
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2))
    (OUT_DIR / "r4_iso_results.json").write_text(json.dumps(r4_iso, indent=2))
    (OUT_DIR / "r4_comb_results.json").write_text(json.dumps(r4_comb, indent=2))

    log(f"\n=== R4 DONE in {(time.time()-t0)/60:.1f} min ===")

    log("\n--- R4 Summary ---")
    for m in METHODS:
        log(f"  {m:<18} iso={r4_iso[m]['mean_auc']:.4f}±{r4_iso[m]['std_auc']:.4f}")
    log(f"  {'Combined':<18} {r4_comb['mean_auc']:.4f}±{r4_comb['std_auc']:.4f}")


if __name__ == "__main__":
    main()
