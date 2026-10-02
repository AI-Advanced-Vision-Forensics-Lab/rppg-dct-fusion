#!/usr/bin/env python3
"""
run_crossattn_experiments.py — Cross-attention fusion: rPPG waveform + DCT lip features.

rPPG tokens attend to DCT tokens, letting the model selectively draw on
lip texture only where it's informative (hypothesis: recovers SadTalker
regression while maintaining IP-LAP improvement).

Protocol: identical to run_fusion_experiments.py
  - Per-method isolated training (real + that method's fakes)
  - 5 seeds, 30 epochs, AdamW lr=1e-3 wd=5e-4, batch=64, weighted BCE
  - 18-identity eval (val+test), mean ± std AUC
  - StandardScaler on DCT fit on train only, per seed

Model: CrossAttnFusion
  rPPG branch:   Conv1d(1, 64, k=8, s=8) → (B, 20, 64) tokens
  DCT branch:    Linear(64, 512) → reshape → (B, 8, 64) tokens
  Cross-attn:    MHA(dim=64, heads=4) — rPPG queries, DCT keys/values
  Pool + classify: mean-pool → Linear(64→32) → ReLU → Dropout(0.3) → Linear(32→1)

Results → data/results/{M_DD_YYYY}/crossattn_experiments.json
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

CONCAT = {
    "AniTalker":      (0.9780, 0.0009),
    "EchoMimic":      (0.9928, 0.0004),
    "EDTalk":         (0.9905, 0.0005),
    "FLOAT":          (0.9871, 0.0003),
    "IP_LAP":         (0.8541, 0.0045),
    "Real3DPortrait": (0.9982, 0.0003),
    "SadTalker":      (0.9130, 0.0020),
    "Overall":        (0.8907, 0.0028),
}


# ── Model ─────────────────────────────────────────────────────────────────────

class CrossAttnFusion(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        # rPPG: (B,1,160) → Conv1d → (B,64,20) → transpose → (B,20,64)
        self.rppg_patch = nn.Conv1d(1, 64, kernel_size=8, stride=8)

        # DCT: (B,64) → Linear(64,512) → reshape → (B,8,64)
        self.dct_proj = nn.Linear(64, 512)

        # Cross-attention: rPPG queries attend to DCT keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=64, num_heads=4, batch_first=True, dropout=dropout
        )

        # Classifier on mean-pooled attended rPPG tokens
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, w, d):
        # w: (B, 160)  d: (B, 64)
        rppg_tokens = self.rppg_patch(w.unsqueeze(1)).transpose(1, 2)  # (B,20,64)
        dct_tokens  = self.dct_proj(d).reshape(d.size(0), 8, 64)       # (B, 8,64)

        # rPPG queries, DCT keys+values
        attn_out, _ = self.cross_attn(rppg_tokens, dct_tokens, dct_tokens)  # (B,20,64)

        pooled = attn_out.mean(dim=1)  # (B, 64)
        return self.classifier(pooled).squeeze(-1)


# ── Data helpers ──────────────────────────────────────────────────────────────

def zscore_wave(w):
    return (w - w.mean()) / (w.std() + 1e-6)


def load_arrays(df, wave_root, dct_lookup):
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

    scaler = StandardScaler()
    Xd_tr  = scaler.fit_transform(X_dct_tr).astype(np.float32)
    Xd_ev  = scaler.transform(X_dct_ev).astype(np.float32)

    pw   = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)]).to(DEVICE)
    mdl  = CrossAttnFusion(dropout=dropout).to(DEVICE)
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
    return float(roc_auc_score(y_ev, probs)), compute_eer(y_ev, probs)


def run_experiment(label, train_df, eval_df, wave_root, dct_lookup, seeds, args):
    X_wave_tr, X_dct_tr, y_tr = load_arrays(train_df, wave_root, dct_lookup)
    X_wave_ev, X_dct_ev, y_ev = load_arrays(eval_df,  wave_root, dct_lookup)

    if X_wave_tr is None or X_wave_ev is None:
        print(f"  [{label}] SKIP — no data loaded")
        return None

    n_real_tr = int((y_tr == 0).sum())
    n_fake_tr = int((y_tr == 1).sum())
    print(f"  [{label}] train: {n_real_tr} real + {n_fake_tr} fake  |  "
          f"eval: {int((y_ev==0).sum())} real + {int((y_ev==1).sum())} fake  |  "
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
    print(f"  → {mean_auc:.4f} ± {std_auc:.4f}  [{time.time()-t0:.0f}s]")

    return {
        "mean_auc":     mean_auc,
        "std_auc":      std_auc,
        "mean_eer":     float(np.mean(eers)),
        "seed_aucs":    {str(s): float(a) for s, a in zip(seeds, aucs)},
        "n_train_real": n_real_tr,
        "n_train_fake": n_fake_tr,
        "n_eval_real":  int((y_ev == 0).sum()),
        "n_eval_fake":  int((y_ev == 1).sum()),
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
                   help="Single method or 'Overall' for debugging.")
    p.add_argument("--dct-file", default=None,
                   help="DCT CSV to use (default: dct_features.csv)")
    p.add_argument("--out-file", default=None,
                   help="Output JSON filename (default: crossattn_experiments.json)")
    return p.parse_args()


def main():
    args      = parse_args()
    data_root = Path(args.data_root)
    wave_root = data_root / "waveforms"

    today       = datetime.now().strftime("%-m_%d_%Y")
    results_dir = data_root / "results" / today
    results_dir.mkdir(parents=True, exist_ok=True)
    dct_file = Path(args.dct_file) if args.dct_file else data_root / "dct_features.csv"
    out_name = args.out_file or "crossattn_experiments.json"
    out_path = results_dir / out_name

    print(f"Device:  {DEVICE}")
    print(f"Results: {out_path}")
    print(f"DCT file: {dct_file.name}")
    print(f"lr={args.lr}  wd={args.wd}  epochs={args.epochs}  batch={args.batch}")

    split_df = pd.read_csv(data_root / "dataset_split.csv")
    dct_df   = pd.read_csv(dct_file)

    split_df = split_df.copy()
    split_df["method"] = split_df["video_id"].apply(
        lambda x: x.split("__")[0] if "__" in x else "real"
    )
    dct_lookup = {
        row["video_id"]: row[DCT_COLS].values.astype(np.float32)
        for _, row in dct_df.iterrows()
    }
    print(f"\nLoaded {len(dct_lookup)} DCT vectors, {len(split_df)} split rows")

    results = {}
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        print(f"Resuming — already done: {list(results.keys())}")

    targets = METHODS + ["Overall"]
    if args.method:
        targets = [args.method]

    for label in targets:
        if label in results:
            print(f"\n[SKIP] {label}")
            continue

        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: {label}")
        print(f"{'='*60}")

        if label == "Overall":
            train_df = split_df[split_df["split"] == "train"]
            eval_df  = split_df[split_df["split"].isin(["val", "test"])]
        else:
            real_rows = split_df[split_df["class"] == "real"]
            fake_rows = split_df[(split_df["class"] == "fake") &
                                 (split_df["method"] == label)]
            train_df = pd.concat([
                real_rows[real_rows["split"] == "train"],
                fake_rows[fake_rows["split"] == "train"],
            ])
            eval_df = pd.concat([
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
    if not results:
        return

    print(f"\n{'='*82}")
    print(f"  rPPG-only | Concat fusion | CrossAttn fusion  (5 seeds, per-method training)")
    print(f"{'='*82}")
    print(f"  {'Method':<16} {'rPPG-only':>12} {'Concat':>12} {'CrossAttn':>12} {'Best':>14}")
    print("  " + "-"*66)

    row_order = [
        "Real3DPortrait", "EDTalk", "FLOAT", "SadTalker",
        "EchoMimic", "AniTalker", "IP_LAP", "Overall",
    ]
    ca_results = {}
    for label in row_order:
        if label not in results:
            continue

        rppg_m, rppg_s = RPPG_ONLY[label]
        cat_m,  cat_s  = CONCAT[label]
        ca_m            = results[label]["mean_auc"]
        ca_s            = results[label]["std_auc"]
        ca_results[label] = ca_m

        candidates = {"rPPG-only": rppg_m, "Concat": cat_m, "CrossAttn": ca_m}
        best = max(candidates, key=candidates.get)

        flags = []
        if label == "SadTalker": flags.append("← recover?")
        if label == "IP_LAP":    flags.append("← improve?")
        flag_str = "  " + " ".join(flags) if flags else ""

        print(f"  {label:<16} "
              f"{rppg_m:.3f}±{rppg_s:.3f}  "
              f"{cat_m:.3f}±{cat_s:.3f}  "
              f"{ca_m:.3f}±{ca_s:.3f}  "
              f"{best:>12}{flag_str}")

    # ── Hypothesis checks ──────────────────────────────────────────────────────
    print(f"\n{'='*82}")
    print(f"  HYPOTHESIS CHECKS")
    print(f"{'='*82}")

    checks = [
        (
            "SadTalker recovery (CrossAttn >= 0.940)",
            ca_results.get("SadTalker", 0) >= 0.940,
            f"CrossAttn={ca_results.get('SadTalker', 0):.4f}  target=0.940  "
            f"rPPG-only={RPPG_ONLY['SadTalker'][0]:.4f}  concat={CONCAT['SadTalker'][0]:.4f}",
        ),
        (
            "IP_LAP beyond concat (CrossAttn > 0.854)",
            ca_results.get("IP_LAP", 0) > 0.8541,
            f"CrossAttn={ca_results.get('IP_LAP', 0):.4f}  concat=0.854",
        ),
        (
            "Overall >= concat (CrossAttn >= 0.891)",
            ca_results.get("Overall", 0) >= 0.8907,
            f"CrossAttn={ca_results.get('Overall', 0):.4f}  concat=0.891",
        ),
    ]

    for desc, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"\n  [{status}] {desc}")
        print(f"         {detail}")

    n_pass = sum(p for _, p, _ in checks)
    print(f"\n  {n_pass}/{len(checks)} hypotheses supported")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
