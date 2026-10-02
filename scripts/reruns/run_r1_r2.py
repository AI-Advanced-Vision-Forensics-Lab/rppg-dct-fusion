#!/usr/bin/env python3
"""
run_r1_r2.py — R1 + R2: rPPG-only 1D ResNet (240K, dropout=0.5)

R1: Isolated per-method training (one model per TF method).
    Train: train-split real + train-split fakes of that method.
    Eval:  val+test-split real + val+test-split fakes of that method.
    → Replaces Table I rPPG column (which used OLD corpus / 16-ID eval).

R2: Combined training (all 7 TF methods' fakes + all real).
    → Replaces Table I rPPG Overall row.

Output:  <out-dir>/r1_r2_rppg_resnet/
  config.json      — full config + corpus counts
  r1_results.json  — per-method AUC mean/std/per-seed
  r2_results.json  — combined AUC mean/std/per-seed
  run.log          — stdout copy

Assertions before training:
  - n_train_real == 1675
  - n_eval_real  == 696
  - n_eval_ids   == 18
  - zero identity overlap between train and eval sets

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

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).parent))
from common import (SPLIT_FILE, DEVICE, SEEDS, METHODS,
                    ResNet1D, load_split, load_waveforms, build_arrays,
                    run_experiment, make_config, DCT_FILE)

from common import OUT_ROOT
OUT_DIR = OUT_ROOT / "r1_r2_rppg_resnet"
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
    log(f"=== R1 + R2: rPPG-only 1D ResNet | device={DEVICE} ===")
    log(f"SPLIT_FILE: {SPLIT_FILE}")

    # ── Load full split ───────────────────────────────────────────────────────
    df = pd.read_csv(SPLIT_FILE)
    log(f"Total split rows: {len(df)}")

    # ── Assertions on corpus ──────────────────────────────────────────────────
    real_df  = df[df['class'] == 'real']
    tr_real  = real_df[real_df['split'] == 'train']
    ev_real  = real_df[real_df['split'].isin(['val', 'test'])]

    n_train_real = len(tr_real)
    n_eval_real  = len(ev_real)
    train_ids    = set(tr_real['identity'].unique())
    eval_ids     = set(ev_real['identity'].unique())
    n_eval_ids   = len(eval_ids)
    overlap      = train_ids & eval_ids

    log(f"ASSERT n_train_real == 1675: got {n_train_real}")
    assert n_train_real == 1675, f"Expected 1675, got {n_train_real}"
    log(f"ASSERT n_eval_real  == 696:  got {n_eval_real}")
    assert n_eval_real  == 696,  f"Expected 696, got {n_eval_real}"
    log(f"ASSERT n_eval_ids   == 18:   got {n_eval_ids}")
    assert n_eval_ids   == 18,   f"Expected 18, got {n_eval_ids}"
    log(f"ASSERT zero identity overlap: {overlap}")
    assert len(overlap) == 0,    f"Identity overlap found: {overlap}"
    log("All assertions PASSED")

    # ── Preload all waveforms ─────────────────────────────────────────────────
    log("Preloading waveforms...")
    all_vids = df['video_id'].tolist()
    wave_cache = load_waveforms(all_vids)
    loaded = sum(1 for v in wave_cache.values() if v is not None)
    log(f"Loaded {loaded} / {len(all_vids)} waveforms")

    # Build real arrays (shared across methods)
    tr_real_w, _, tr_real_l, _ = build_arrays(tr_real, wave_cache, {}, require_dct=False)
    ev_real_w, _, ev_real_l, _ = build_arrays(ev_real, wave_cache, {}, require_dct=False)
    log(f"Real arrays: train {tr_real_w.shape}, eval {ev_real_w.shape}")

    # ── R1: isolated per-method ───────────────────────────────────────────────
    log("\n=== R1: Isolated per-method rPPG ResNet ===")
    r1_results = {}

    for method in METHODS:
        t_m = time.time()
        fake_mask = df['video_id'].str.startswith(method + '__')
        tr_fake_df = df[fake_mask & (df['split'] == 'train')]
        ev_fake_df = df[fake_mask & (df['split'].isin(['val', 'test']))]

        tr_fk_w, _, tr_fk_l, _ = build_arrays(tr_fake_df, wave_cache, {}, require_dct=False)
        ev_fk_w, _, ev_fk_l, _ = build_arrays(ev_fake_df, wave_cache, {}, require_dct=False)

        wave_tr = np.concatenate([tr_fk_w, tr_real_w])
        label_tr = np.concatenate([tr_fk_l, tr_real_l])
        wave_ev  = np.concatenate([ev_fk_w, ev_real_w])
        label_ev = np.concatenate([ev_fk_l, ev_real_l])

        n_tr_fake = int(label_tr.sum())
        n_ev_fake = int(label_ev.sum())

        # Per-method identity assertions for eval
        ev_fake_ids = set(df.loc[fake_mask & df['split'].isin(['val','test']), 'identity'])
        tr_fake_ids = set(df.loc[fake_mask & (df['split'] == 'train'), 'identity'])
        id_overlap = eval_ids & train_ids  # already asserted 0 above
        assert len(id_overlap) == 0, f"{method}: identity overlap in eval"

        log(f"\n  [{method}] train: {n_train_real}r+{n_tr_fake}f  eval: {n_eval_real}r+{n_ev_fake}f")

        dct_dummy = np.zeros((len(wave_tr), 64), dtype=np.float32)
        dct_ev_dummy = np.zeros((len(wave_ev), 64), dtype=np.float32)

        result = run_experiment(
            model_factory=ResNet1D,
            model_kwargs={"dropout": 0.5},
            wave_tr=wave_tr, dct_tr=dct_dummy, label_tr=label_tr,
            wave_ev=wave_ev, dct_ev=dct_ev_dummy, label_ev=label_ev,
            seeds=SEEDS, use_dct=False,
            lr=1e-3, wd=5e-4, epochs=30, batch=64,
        )
        result["n_train_real"] = n_train_real
        result["n_train_fake"] = n_tr_fake
        result["n_eval_real"]  = n_eval_real
        result["n_eval_fake"]  = n_ev_fake
        result["model_params"] = ResNet1D.param_count()

        r1_results[method] = result
        log(f"  [{method}] AUC = {result['mean_auc']:.4f} ± {result['std_auc']:.4f}  "
            f"({time.time()-t_m:.0f}s)")

    # ── R2: combined (all 7 methods) ──────────────────────────────────────────
    log("\n=== R2: Combined rPPG ResNet (all 7 TF methods) ===")
    t_m = time.time()

    all_fake_df  = df[(df['class'] == 'fake')]
    tr_all_fk_df = all_fake_df[all_fake_df['split'] == 'train']
    ev_all_fk_df = all_fake_df[all_fake_df['split'].isin(['val', 'test'])]

    tr_fk_w, _, tr_fk_l, _ = build_arrays(tr_all_fk_df, wave_cache, {}, require_dct=False)
    ev_fk_w, _, ev_fk_l, _ = build_arrays(ev_all_fk_df, wave_cache, {}, require_dct=False)

    wave_tr  = np.concatenate([tr_fk_w, tr_real_w])
    label_tr = np.concatenate([tr_fk_l, tr_real_l])
    wave_ev  = np.concatenate([ev_fk_w, ev_real_w])
    label_ev = np.concatenate([ev_fk_l, ev_real_l])

    n_tr_fake = int(label_tr.sum())
    n_ev_fake = int(label_ev.sum())
    log(f"  Combined train: {n_train_real}r+{n_tr_fake}f  eval: {n_eval_real}r+{n_ev_fake}f")

    dct_dummy    = np.zeros((len(wave_tr), 64), dtype=np.float32)
    dct_ev_dummy = np.zeros((len(wave_ev), 64), dtype=np.float32)

    r2_result = run_experiment(
        model_factory=ResNet1D,
        model_kwargs={"dropout": 0.5},
        wave_tr=wave_tr, dct_tr=dct_dummy, label_tr=label_tr,
        wave_ev=wave_ev, dct_ev=dct_ev_dummy, label_ev=label_ev,
        seeds=SEEDS, use_dct=False,
        lr=1e-3, wd=5e-4, epochs=30, batch=64,
    )
    r2_result["n_train_real"] = n_train_real
    r2_result["n_train_fake"] = n_tr_fake
    r2_result["n_eval_real"]  = n_eval_real
    r2_result["n_eval_fake"]  = n_ev_fake
    r2_result["model_params"] = ResNet1D.param_count()
    log(f"  Combined AUC = {r2_result['mean_auc']:.4f} ± {r2_result['std_auc']:.4f}  "
        f"({time.time()-t_m:.0f}s)")

    # ── Save outputs ──────────────────────────────────────────────────────────
    corpus_info = {"n_train_real": n_train_real, "n_eval_real": n_eval_real}
    config = make_config(
        "ResNet1D",
        {"dropout": 0.5},
        corpus_info,
        extra={
            "run_ids": ["R1", "R2"],
            "eval_identities": sorted(eval_ids),
            "train_identity_count": len(train_ids),
        }
    )

    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2))
    (OUT_DIR / "r1_results.json").write_text(json.dumps(r1_results, indent=2))
    (OUT_DIR / "r2_results.json").write_text(json.dumps(r2_result, indent=2))

    log(f"\n=== DONE in {(time.time()-t0)/60:.1f} min ===")
    log(f"Outputs in {OUT_DIR}")

    # Print summary
    log("\n--- R1 Summary (new Table I rPPG column, FINAL corpus 18-ID eval) ---")
    old = {
        "AniTalker": 0.908, "EchoMimic": 0.865, "EDTalk": 0.950,
        "FLOAT": 0.794, "IP_LAP": 0.690, "Real3DPortrait": 0.985,
        "SadTalker": 0.946,
    }
    for m in METHODS:
        new_val = r1_results[m]["mean_auc"]
        delta = new_val - old[m]
        log(f"  {m:<18} {new_val:.4f} ± {r1_results[m]['std_auc']:.4f}  "
            f"(prev {old[m]:.3f}, Δ={delta:+.3f})")

    log(f"\n--- R2 Summary (new Table I rPPG Overall, FINAL corpus) ---")
    log(f"  Combined AUC = {r2_result['mean_auc']:.4f} ± {r2_result['std_auc']:.4f}  "
        f"(prev Overall [old corpus not recorded separately])")


if __name__ == "__main__":
    main()
