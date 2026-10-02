#!/usr/bin/env python3
"""
run_fusion_experiments.py — Concatenation fusion: rPPG waveform + DCT lip features.

Smoke run: does DCT add signal over rPPG alone? Especially on IP-LAP?

Protocol (matches Paper 1 per-method isolated experiment):
  - Per-method training: real_train + method_fakes_train
  - Eval: real_val+test + method_fakes_val+test (18 identities)
  - 5 seeds, mean ± std AUC
  - Overall row: all methods combined (mirrors eval18 baseline)
  - AdamW, lr=1e-3, wd=5e-4, 30 epochs, batch=64, weighted BCE

Model: FusionMLP
  rPPG encoder:  Linear(160→64) → BN → ReLU → Dropout(0.3)
  DCT encoder:   Linear(64→64)  → BN → ReLU → Dropout(0.3)
  Fusion:        cat([h_rppg, h_dct]) → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→1)

Results → data/results/{M_DD_YYYY}/fusion_experiments.json
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS  = [42, 7, 123, 999, 2024]
DCT_COLS = [f"dct_{i}" for i in range(64)]

METHODS = [
    "AniTalker", "EchoMimic", "EDTalk", "FLOAT",
    "IP_LAP", "Real3DPortrait", "SadTalker",
]

# Paper 1 per-method 1D ResNet AUC (5-seed mean, per-method isolated training)
RPPG_ONLY = {
    "AniTalker":      (0.9078, 0.0034),
    "EchoMimic":      (0.8653, 0.0028),
    "EDTalk":         (0.9496, 0.0020),
    "FLOAT":          (0.7935, 0.0050),
    "IP_LAP":         (0.6899, 0.0100),
    "Real3DPortrait": (0.9848, 0.0010),
    "SadTalker":      (0.9463, 0.0023),
    "Overall":        (0.8064, 0.0028),
}


# ── Model ─────────────────────────────────────────────────────────────────────

class FusionMLP(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.rppg_enc = nn.Sequential(
            nn.Linear(160, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
        )
        self.dct_enc = nn.Sequential(
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, w, d):
        h = torch.cat([self.rppg_enc(w), self.dct_enc(d)], dim=1)
        return self.fusion(h).squeeze(-1)


# ── Data loading ──────────────────────────────────────────────────────────────

def zscore_wave(w):
    return (w - w.mean()) / (w.std() + 1e-6)


def load_arrays(df, wave_root, dct_lookup):
    """
    df        : rows from merged dataset (has video_id, class)
    wave_root : path to waveforms/{real,fake}/
    dct_lookup: dict video_id → np.array(64,)
    Returns (waves, dcts, labels) as np.float32 arrays.
    Silently skips missing waveform files.
    """
    waves, dcts, labels = [], [], []
    for _, row in df.iterrows():
        vid = row["video_id"]
        if vid not in dct_lookup:
            continue
        wpath = wave_root / row["class"] / f"{vid}.npy"
        if not wpath.exists():
            continue
        w = np.load(wpath).astype(np.float32)
        waves.append(zscore_wave(w))
        dcts.append(dct_lookup[vid])
        labels.append(float(row["class"] == "fake"))
    if not waves:
        return None, None, None
    return np.stack(waves), np.stack(dcts), np.array(labels, dtype=np.float32)


# ── Training / eval ───────────────────────────────────────────────────────────

def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    i = np.argmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def run_one_seed(X_wave_tr, X_dct_tr, y_tr,
                 X_wave_ev, X_dct_ev, y_ev,
                 seed, lr=1e-3, wd=5e-4, epochs=30, batch=64, dropout=0.3):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Fit DCT scaler on train only (re-fit each seed for independence)
    scaler = StandardScaler()
    Xd_tr = scaler.fit_transform(X_dct_tr).astype(np.float32)
    Xd_ev = scaler.transform(X_dct_ev).astype(np.float32)

    pw   = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)]).to(DEVICE)
    mdl  = FusionMLP(dropout=dropout).to(DEVICE)
    opt  = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=wd)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    Xw_t = torch.from_numpy(X_wave_tr).to(DEVICE)
    Xd_t = torch.from_numpy(Xd_tr).to(DEVICE)
    yt   = torch.from_numpy(y_tr).to(DEVICE)
    Xw_e = torch.from_numpy(X_wave_ev)
    Xd_e = torch.from_numpy(Xd_ev)

    for _ in range(epochs):
        mdl.train()
        perm = torch.randperm(len(Xw_t), device=DEVICE)
        for s in range(0, len(Xw_t), batch):
            ix = perm[s:s + batch]
            opt.zero_grad()
            crit(mdl(Xw_t[ix], Xd_t[ix]), yt[ix]).backward()
            opt.step()

    mdl.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, len(Xw_e), 512):
            p = torch.sigmoid(
                mdl(Xw_e[s:s+512].to(DEVICE), Xd_e[s:s+512].to(DEVICE))
            ).cpu().numpy()
            probs.extend(p.tolist())

    probs = np.array(probs)
    auc = float(roc_auc_score(y_ev, probs))
    eer = compute_eer(y_ev, probs)
    return auc, eer


def run_experiment(label, train_df, eval_df, wave_root, dct_lookup, seeds, args):
    """Train + eval for one method (or 'Overall'). Returns result dict."""
    X_wave_tr, X_dct_tr, y_tr = load_arrays(train_df, wave_root, dct_lookup)
    X_wave_ev, X_dct_ev, y_ev = load_arrays(eval_df,  wave_root, dct_lookup)

    if X_wave_tr is None or X_wave_ev is None:
        print(f"  [{label}] SKIP — no data loaded")
        return None

    n_real_tr = int((y_tr == 0).sum())
    n_fake_tr = int((y_tr == 1).sum())
    n_real_ev = int((y_ev == 0).sum())
    n_fake_ev = int((y_ev == 1).sum())
    print(f"  [{label}] train: {n_real_tr} real + {n_fake_tr} fake  |  "
          f"eval: {n_real_ev} real + {n_fake_ev} fake  |  "
          f"pos_weight={n_real_tr/max(n_fake_tr,1):.3f}")

    aucs, eers = [], []
    t0 = time.time()
    for seed in seeds:
        auc, eer = run_one_seed(
            X_wave_tr, X_dct_tr, y_tr,
            X_wave_ev, X_dct_ev, y_ev,
            seed=seed, lr=args.lr, wd=args.wd,
            epochs=args.epochs, batch=args.batch,
        )
        aucs.append(auc)
        eers.append(eer)
        print(f"    seed={seed}  AUC={auc:.4f}  EER={eer*100:.1f}%")

    mean_auc = float(np.mean(aucs))
    std_auc  = float(np.std(aucs))
    elapsed  = time.time() - t0
    print(f"  → {mean_auc:.4f} ± {std_auc:.4f}  [{elapsed:.0f}s]")

    return {
        "mean_auc":     mean_auc,
        "std_auc":      std_auc,
        "mean_eer":     float(np.mean(eers)),
        "seed_aucs":    {str(s): float(a) for s, a in zip(seeds, aucs)},
        "n_train_real": n_real_tr,
        "n_train_fake": n_fake_tr,
        "n_eval_real":  n_real_ev,
        "n_eval_fake":  n_fake_ev,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--wd",     type=float, default=5e-4)
    p.add_argument("--epochs", type=int,   default=30)
    p.add_argument("--batch",  type=int,   default=64)
    p.add_argument("--method", default=None,
                   help="Run single method only (or 'Overall'). Useful for debugging.")
    p.add_argument("--dct-file", default=None,
                   help="DCT CSV to use (default: dct_features.csv)")
    p.add_argument("--out-file", default=None,
                   help="Output JSON filename (default: fusion_experiments.json)")
    return p.parse_args()


def main():
    args      = parse_args()
    data_root = Path(args.data_root)
    wave_root = data_root / "waveforms"

    today       = datetime.now().strftime("%-m_%d_%Y")
    results_dir = data_root / "results" / today
    results_dir.mkdir(parents=True, exist_ok=True)
    dct_file = Path(args.dct_file) if args.dct_file else data_root / "dct_features.csv"
    out_name = args.out_file or "fusion_experiments.json"
    out_path = results_dir / out_name

    print(f"Device:  {DEVICE}")
    print(f"Results: {out_path}")
    print(f"DCT file: {dct_file.name}")
    print(f"lr={args.lr}  wd={args.wd}  epochs={args.epochs}  batch={args.batch}")

    # ── Load and merge data ────────────────────────────────────────────────────
    split_df = pd.read_csv(data_root / "dataset_split.csv")
    dct_df   = pd.read_csv(dct_file)

    # Build fast lookup: video_id → dct vector
    dct_lookup = {
        row["video_id"]: row[DCT_COLS].values.astype(np.float32)
        for _, row in dct_df.iterrows()
    }
    print(f"\nLoaded {len(dct_lookup)} DCT vectors, {len(split_df)} split rows")

    # Derive method from video_id (fakes: "Method__stem", reals: no "__")
    split_df = split_df.copy()
    split_df["method"] = split_df["video_id"].apply(
        lambda x: x.split("__")[0] if "__" in x else "real"
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    results = {}
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        done = list(results.keys())
        print(f"Resuming — already done: {done}")

    # Determine which experiments to run
    targets = METHODS + ["Overall"]
    if args.method:
        targets = [args.method]

    # ── Per-method experiments ─────────────────────────────────────────────────
    for label in targets:
        if label in results:
            print(f"\n[SKIP] {label} — already done")
            continue

        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: {label}")
        print(f"{'='*60}")

        if label == "Overall":
            train_df = split_df[split_df["split"] == "train"]
            eval_df  = split_df[split_df["split"].isin(["val", "test"])]
        else:
            real_rows  = split_df[split_df["class"] == "real"]
            fake_rows  = split_df[(split_df["class"] == "fake") &
                                  (split_df["method"] == label)]
            train_df = pd.concat([
                real_rows[real_rows["split"] == "train"],
                fake_rows[fake_rows["split"] == "train"],
            ])
            eval_df  = pd.concat([
                real_rows[real_rows["split"].isin(["val", "test"])],
                fake_rows[fake_rows["split"].isin(["val", "test"])],
            ])

        res = run_experiment(label, train_df, eval_df, wave_root, dct_lookup,
                             SEEDS, args)
        if res is None:
            continue

        results[label] = res
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    # ── Comparison table ───────────────────────────────────────────────────────
    if len(results) < 2:
        print("\nNot enough results yet for comparison table.")
        return

    print(f"\n{'='*78}")
    print(f"  FUSION vs rPPG-ONLY  (FusionMLP, 5 seeds, per-method training)")
    print(f"{'='*78}")
    print(f"  {'Method':<16} {'rPPG-only AUC':>16} {'Fusion AUC':>14} {'Delta':>8}")
    print("  " + "-"*58)

    deltas = {}
    row_order = METHODS + ["Overall"]
    for label in row_order:
        if label not in results:
            continue
        rppg_mean, rppg_std = RPPG_ONLY[label]
        fus_mean             = results[label]["mean_auc"]
        fus_std              = results[label]["std_auc"]
        delta                = fus_mean - rppg_mean
        deltas[label]        = delta
        flag = "  ← KEY" if label == "IP_LAP" else ""
        print(f"  {label:<16} "
              f"{rppg_mean:.3f} ± {rppg_std:.3f}   "
              f"{fus_mean:.3f} ± {fus_std:.3f}   "
              f"{delta:+.3f}{flag}")

    # ── Hypothesis test ───────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"  HYPOTHESIS TEST")
    print(f"{'='*78}")
    if "IP_LAP" in deltas and len(deltas) > 2:
        other_methods = [m for m in METHODS if m != "IP_LAP" and m in deltas]
        ip_lap_delta  = deltas["IP_LAP"]
        avg_other     = np.mean([deltas[m] for m in other_methods])
        print(f"  IP-LAP  fusion delta:          {ip_lap_delta:+.4f}")
        print(f"  Avg other methods fusion delta: {avg_other:+.4f}")
        print(f"  Difference (IP-LAP − avg):     {ip_lap_delta - avg_other:+.4f}")
        if ip_lap_delta > avg_other:
            print("  HYPOTHESIS: SUPPORTED — DCT adds disproportionate signal on IP-LAP")
        else:
            print("  HYPOTHESIS: NOT SUPPORTED — IP-LAP improvement ≤ average")
    else:
        print("  (Need all methods + IP_LAP to test hypothesis)")

    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
