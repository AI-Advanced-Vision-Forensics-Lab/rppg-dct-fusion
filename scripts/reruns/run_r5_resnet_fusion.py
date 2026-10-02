#!/usr/bin/env python3
"""
run_r5_resnet_fusion.py — R5 (conditional): Concat-fusion with ResNet1D trunk

Replaces FusionMLP's 10K linear rPPG encoder with the full 240K ResNet1D trunk
(GAP output → 128-dim) and a matching fusion head.  Architecture (ResNetFusion):
  rPPG: ResNet1D stem+stage1+stage2+stage3+GAP → 128-dim
  DCT:  Linear(64→64)→BN→ReLU→Dropout(0.3) → 64-dim
  Fusion: cat(128,64)→Linear(192→64)→ReLU→Dropout→Linear(64,1)
  Total: ~263K params

Runs:
  R5-iso:  isolated per-method (replaces Table I Concat column with ResNet rPPG)
  R5-comb: combined all 7 methods (replaces Table I Concat Overall)

Output: <out-dir>/r5_resnet_fusion/
  config.json, r5_iso_results.json, r5_comb_results.json, run.log

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
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import (SPLIT_FILE, DEVICE, SEEDS, METHODS,
                    ResNetFusion, load_waveforms, build_arrays,
                    run_experiment, make_config, DCT_FILE)

from common import OUT_ROOT
OUT_DIR = OUT_ROOT / "r5_resnet_fusion"
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
    log(f"=== R5: Concat with ResNet1D trunk | device={DEVICE} ===")
    log(f"  ResNetFusion params: {ResNetFusion.param_count():,}")

    df = pd.read_csv(SPLIT_FILE)
    dct_df = pd.read_csv(DCT_FILE)
    dct_lookup = {row['video_id']: np.array(
                      [row[f'dct_{i}'] for i in range(64)], dtype=np.float32)
                  for _, row in dct_df.iterrows()}

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

    tr_real_w, tr_real_d, tr_real_l, _ = build_arrays(tr_real, wave_cache, dct_lookup)
    ev_real_w, ev_real_d, ev_real_l, _ = build_arrays(ev_real, wave_cache, dct_lookup)

    # ── R5 isolated ───────────────────────────────────────────────────────────
    log("\n=== R5-iso: Isolated per-method ResNetFusion ===")
    r5_iso = {}
    for method in METHODS:
        t_m = time.time()
        fake_mask = df['video_id'].str.startswith(method + '__')
        tr_fk_df = df[fake_mask & (df['split'] == 'train')]
        ev_fk_df = df[fake_mask & df['split'].isin(['val', 'test'])]

        tr_fk_w, tr_fk_d, tr_fk_l, _ = build_arrays(tr_fk_df, wave_cache, dct_lookup)
        ev_fk_w, ev_fk_d, ev_fk_l, _ = build_arrays(ev_fk_df, wave_cache, dct_lookup)

        wave_tr  = np.concatenate([tr_fk_w, tr_real_w])
        dct_tr   = np.concatenate([tr_fk_d, tr_real_d])
        label_tr = np.concatenate([tr_fk_l, tr_real_l])
        wave_ev  = np.concatenate([ev_fk_w, ev_real_w])
        dct_ev   = np.concatenate([ev_fk_d, ev_real_d])
        label_ev = np.concatenate([ev_fk_l, ev_real_l])

        scaler = StandardScaler()
        dct_tr_sc = scaler.fit_transform(dct_tr).astype(np.float32)
        dct_ev_sc = scaler.transform(dct_ev).astype(np.float32)

        result = run_experiment(
            ResNetFusion, {"dropout": 0.3},
            wave_tr, dct_tr_sc, label_tr,
            wave_ev, dct_ev_sc,  label_ev,
            SEEDS, use_dct=True,
            lr=1e-3, wd=5e-4, epochs=30, batch=64)

        result.update({
            "n_train_real": len(tr_real_w), "n_train_fake": int(label_tr.sum()),
            "n_eval_real": len(ev_real_w), "n_eval_fake": int(label_ev.sum()),
            "model_params": ResNetFusion.param_count(),
        })
        r5_iso[method] = result
        log(f"  [{method}] AUC={result['mean_auc']:.4f}±{result['std_auc']:.4f}  ({time.time()-t_m:.0f}s)")

    # ── R5 combined ───────────────────────────────────────────────────────────
    log("\n=== R5-comb: Combined ResNetFusion (all 7 methods) ===")
    t_m = time.time()
    all_fake_df  = df[df['class'] == 'fake']
    tr_all_fk_df = all_fake_df[all_fake_df['split'] == 'train']
    ev_all_fk_df = all_fake_df[all_fake_df['split'].isin(['val', 'test'])]

    tr_fk_w, tr_fk_d, tr_fk_l, _ = build_arrays(tr_all_fk_df, wave_cache, dct_lookup)
    ev_fk_w, ev_fk_d, ev_fk_l, _ = build_arrays(ev_all_fk_df, wave_cache, dct_lookup)

    wave_tr  = np.concatenate([tr_fk_w, tr_real_w])
    dct_tr   = np.concatenate([tr_fk_d, tr_real_d])
    label_tr = np.concatenate([tr_fk_l, tr_real_l])
    wave_ev  = np.concatenate([ev_fk_w, ev_real_w])
    dct_ev   = np.concatenate([ev_fk_d, ev_real_d])
    label_ev = np.concatenate([ev_fk_l, ev_real_l])

    scaler = StandardScaler()
    dct_tr_sc = scaler.fit_transform(dct_tr).astype(np.float32)
    dct_ev_sc = scaler.transform(dct_ev).astype(np.float32)

    r5_comb = run_experiment(
        ResNetFusion, {"dropout": 0.3},
        wave_tr, dct_tr_sc, label_tr,
        wave_ev, dct_ev_sc,  label_ev,
        SEEDS, use_dct=True,
        lr=1e-3, wd=5e-4, epochs=30, batch=64)

    r5_comb.update({
        "n_train_real": len(tr_real_w), "n_train_fake": int(label_tr.sum()),
        "n_eval_real": len(ev_real_w), "n_eval_fake": int(label_ev.sum()),
        "model_params": ResNetFusion.param_count(),
    })
    log(f"  Combined AUC={r5_comb['mean_auc']:.4f}±{r5_comb['std_auc']:.4f}  ({time.time()-t_m:.0f}s)")

    # ── Save ──────────────────────────────────────────────────────────────────
    config = make_config(
        "ResNetFusion",
        {"dropout": 0.3},
        {"n_train_real": 1675, "n_eval_real": 696},
        extra={
            "run_id": "R5",
            "arch": "ResNet1D trunk (128-dim GAP) + DCT MLP (64-dim) → fusion Linear(192,64) → Linear(64,1)",
            "model_params": ResNetFusion.param_count(),
            "note": "Architecturally consistent fusion: same ResNet rPPG backbone in "
                    "both standalone (R1) and fusion (R5) settings.",
        }
    )
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2))
    (OUT_DIR / "r5_iso_results.json").write_text(json.dumps(r5_iso, indent=2))
    (OUT_DIR / "r5_comb_results.json").write_text(json.dumps(r5_comb, indent=2))

    log(f"\n=== R5 DONE in {(time.time()-t0)/60:.1f} min ===")
    log("\n--- R5 Summary ---")
    for m in METHODS:
        log(f"  {m:<18} iso={r5_iso[m]['mean_auc']:.4f}±{r5_iso[m]['std_auc']:.4f}")
    log(f"  {'Combined':<18} {r5_comb['mean_auc']:.4f}±{r5_comb['std_auc']:.4f}")


if __name__ == "__main__":
    main()
