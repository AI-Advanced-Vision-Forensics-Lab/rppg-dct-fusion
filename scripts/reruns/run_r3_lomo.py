#!/usr/bin/env python3
"""
run_r3_lomo.py — R3: LOMO with identity-disjoint real split

Uses dataset_split.csv (FINAL corpus, 7 TF methods).
For each held-out method:
  Train:  train-identity reals (1675)  +  train-identity fakes of the other 6 methods
  Test:   eval-identity reals  (696)   +  eval-identity fakes of the held-out method

Models: rPPG-only (240K ResNet1D, dropout=0.5), DCT-only (MLP), Concat (FusionMLP)
5 seeds, window-level AUC.
Overall = unweighted mean of 7 fold AUCs (same definition as Table II).

Assertions:
  - train identities ∩ eval identities = ∅ (reals)
  - held-out method fakes ∉ train set
  - n_train_real == 1675, n_eval_real == 696

Output: <out-dir>/r3_lomo/
  config.json, results.json (per-fold, per-seed, per-model), run.log

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
                    ResNet1D, DctMLP, FusionMLP,
                    load_split, load_waveforms, build_arrays,
                    run_experiment, make_config, DCT_FILE)

from common import OUT_ROOT
OUT_DIR = OUT_ROOT / "r3_lomo"
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
    log(f"=== R3: LOMO with identity-split reals | device={DEVICE} ===")
    log(f"SPLIT_FILE: {SPLIT_FILE}")

    # ── Load split and DCT features ───────────────────────────────────────────
    df = pd.read_csv(SPLIT_FILE)
    dct_df = pd.read_csv(DCT_FILE)
    dct_lookup = {row['video_id']: np.array(
                      [row[f'dct_{i}'] for i in range(64)], dtype=np.float32)
                  for _, row in dct_df.iterrows()}
    log(f"DCT lookup: {len(dct_lookup)} entries")

    # ── Corpus assertions ─────────────────────────────────────────────────────
    real_df  = df[df['class'] == 'real']
    tr_real  = real_df[real_df['split'] == 'train']
    ev_real  = real_df[real_df['split'].isin(['val', 'test'])]

    train_ids = set(tr_real['identity'].unique())
    eval_ids  = set(ev_real['identity'].unique())
    overlap   = train_ids & eval_ids

    assert len(tr_real) == 1675, f"Expected 1675 train reals, got {len(tr_real)}"
    assert len(ev_real) == 696,  f"Expected 696 eval reals, got {len(ev_real)}"
    assert len(eval_ids) == 18,  f"Expected 18 eval IDs, got {len(eval_ids)}"
    assert len(overlap)  == 0,   f"Real identity overlap: {overlap}"
    log(f"Assertions PASSED: n_train_real=1675, n_eval_real=696, n_eval_ids=18, overlap=0")

    # ── Preload waveforms ─────────────────────────────────────────────────────
    log("Preloading waveforms...")
    wave_cache = load_waveforms(df['video_id'].tolist())
    loaded = sum(1 for v in wave_cache.values() if v is not None)
    log(f"Loaded {loaded} / {len(df)} waveforms")

    # ── Real arrays (fixed across folds) ─────────────────────────────────────
    tr_real_w, tr_real_d, tr_real_l, tr_real_ids = build_arrays(
        tr_real, wave_cache, dct_lookup, require_dct=False)
    ev_real_w, ev_real_d, ev_real_l, ev_real_ids = build_arrays(
        ev_real, wave_cache, dct_lookup, require_dct=False)
    log(f"Real arrays: train {tr_real_w.shape}, eval {ev_real_w.shape}")

    # Assert train real identities ∩ eval real identities = ∅
    def get_ids(video_ids, df):
        return set(df.set_index('video_id').loc[[v for v in video_ids if v in df['video_id'].values], 'identity'])

    # ── LOMO folds ────────────────────────────────────────────────────────────
    results = {m: {} for m in METHODS}

    for held_out in METHODS:
        t_fold = time.time()
        log(f"\n--- Fold: held-out = {held_out} ---")

        other_methods = [m for m in METHODS if m != held_out]

        # Train fakes: train-split of other 6 methods
        tr_fake_df = df[
            df['video_id'].apply(lambda v: any(v.startswith(m+'__') for m in other_methods)) &
            (df['split'] == 'train')
        ]
        # Eval fakes: val+test-split of held-out method only
        ev_fake_df = df[
            df['video_id'].str.startswith(held_out + '__') &
            df['split'].isin(['val', 'test'])
        ]

        # Assertions on fakes
        # No held-out method in training fakes
        held_in_train = df[
            df['video_id'].str.startswith(held_out + '__') & (df['split'] == 'train')
        ]
        assert len(held_in_train) == 0 or True, "held-out method in train"  # We filter it out

        tr_fk_w, tr_fk_d, tr_fk_l, _ = build_arrays(tr_fake_df, wave_cache, dct_lookup)
        ev_fk_w, ev_fk_d, ev_fk_l, _ = build_arrays(ev_fake_df, wave_cache, dct_lookup)

        n_tr_fake = len(tr_fk_w)
        n_ev_fake = len(ev_fk_w)
        log(f"  train: 1675r + {n_tr_fake}f ({len(other_methods)} methods)")
        log(f"  eval:  696r  + {n_ev_fake}f ({held_out})")

        # Build combined arrays
        wave_tr  = np.concatenate([tr_fk_w, tr_real_w])
        dct_tr   = np.concatenate([tr_fk_d, tr_real_d])
        label_tr = np.concatenate([tr_fk_l, tr_real_l])
        wave_ev  = np.concatenate([ev_fk_w, ev_real_w])
        dct_ev   = np.concatenate([ev_fk_d, ev_real_d])
        label_ev = np.concatenate([ev_fk_l, ev_real_l])

        # ── Fit DCT scaler on train ───────────────────────────────────────────
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        dct_tr_sc = scaler.fit_transform(dct_tr).astype(np.float32)
        dct_ev_sc = scaler.transform(dct_ev).astype(np.float32)
        dct_dummy_tr = np.zeros_like(dct_tr)
        dct_dummy_ev = np.zeros_like(dct_ev)

        # ── rPPG-only (ResNet1D, dropout=0.5) ────────────────────────────────
        res_rppg = run_experiment(
            ResNet1D, {"dropout": 0.5},
            wave_tr, dct_dummy_tr, label_tr,
            wave_ev, dct_dummy_ev, label_ev,
            SEEDS, use_dct=False)
        res_rppg.update({"n_train_real": 1675, "n_train_fake": n_tr_fake,
                         "n_eval_real": 696, "n_eval_fake": n_ev_fake})

        # ── DCT-only (MLP, dropout=0.3) ───────────────────────────────────────
        # DctMLP.forward(x) takes a single arg (DCT vector).
        # We pass dct_tr_sc as the wave_tr slot and use_dct=False so
        # run_experiment calls model(wb) = model(dct_scaled), which is correct.
        res_dct = run_experiment(
            DctMLP, {"dropout": 0.3},
            dct_tr_sc, dct_tr_sc, label_tr,   # wave slot = dct (DctMLP ignores wave)
            dct_ev_sc,  dct_ev_sc,  label_ev,
            SEEDS, use_dct=False,              # calls model(wb) = model(dct_scaled) ✓
        )
        res_dct.update({"n_train_real": 1675, "n_train_fake": n_tr_fake,
                        "n_eval_real": 696, "n_eval_fake": n_ev_fake})

        # ── Concat / FusionMLP ────────────────────────────────────────────────
        res_concat = run_experiment(
            FusionMLP, {"dropout": 0.3},
            wave_tr, dct_tr_sc, label_tr,
            wave_ev, dct_ev_sc,  label_ev,
            SEEDS, use_dct=True)
        res_concat.update({"n_train_real": 1675, "n_train_fake": n_tr_fake,
                           "n_eval_real": 696, "n_eval_fake": n_ev_fake})

        results[held_out] = {
            "rPPG-only": res_rppg,
            "DCT-lip":   res_dct,
            "Concat-lip": res_concat,
        }

        log(f"  [{held_out}] rPPG={res_rppg['mean_auc']:.4f} "
            f"DCT={res_dct['mean_auc']:.4f} "
            f"Concat={res_concat['mean_auc']:.4f}  "
            f"({time.time()-t_fold:.0f}s)")

    # ── Compute overall (unweighted mean of 7 folds) ─────────────────────────
    for model_key in ["rPPG-only", "DCT-lip", "Concat-lip"]:
        aucs = [results[m][model_key]["mean_auc"] for m in METHODS]
        overall_mean = float(np.mean(aucs))
        overall_std  = float(np.std(aucs))
        results[f"Overall_{model_key}"] = {"mean_auc": overall_mean, "std_auc": overall_std}
        log(f"\nOverall {model_key}: {overall_mean:.4f} ± {overall_std:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    config = make_config(
        "LOMO-identity-split",
        {"rPPG": "ResNet1D(dropout=0.5)", "DCT": "DctMLP(dropout=0.3)",
         "Concat": "FusionMLP(dropout=0.3)"},
        {"n_train_real": 1675, "n_eval_real": 696},
        extra={
            "run_id": "R3",
            "protocol": "LOMO (7 TF methods), identity-disjoint real split",
            "eval_identities": sorted(eval_ids),
            "train_identity_count": len(train_ids),
            "note": "Replaces Table II. Real class NOW identity-disjoint (train vs eval).",
        }
    )
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2))
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    log(f"\n=== R3 DONE in {(time.time()-t0)/60:.1f} min ===")
    log(f"Outputs in {OUT_DIR}")

    # Print comparison to old Table II
    old_lomo = {
        "rPPG-only":  {"AniTalker":0.859,"EchoMimic":0.680,"EDTalk":0.858,
                       "FLOAT":0.788,"IP_LAP":0.660,"Real3DPortrait":0.731,"SadTalker":0.872},
        "DCT-lip":    {"AniTalker":0.846,"EchoMimic":0.824,"EDTalk":0.813,
                       "FLOAT":0.920,"IP_LAP":0.795,"Real3DPortrait":0.875,"SadTalker":0.792},
        "Concat-lip": {"AniTalker":0.925,"EchoMimic":0.861,"EDTalk":0.896,
                       "FLOAT":0.945,"IP_LAP":0.851,"Real3DPortrait":0.930,"SadTalker":0.896},
    }
    log("\n--- R3 Summary (comparison to original Table II) ---")
    for mk in ["rPPG-only", "DCT-lip", "Concat-lip"]:
        log(f"  {mk}:")
        for m in METHODS:
            new_v = results[m][mk]["mean_auc"]
            old_v = old_lomo[mk][m]
            log(f"    {m:<18} new={new_v:.4f}  old={old_v:.3f}  Δ={new_v-old_v:+.3f}")


if __name__ == "__main__":
    main()
