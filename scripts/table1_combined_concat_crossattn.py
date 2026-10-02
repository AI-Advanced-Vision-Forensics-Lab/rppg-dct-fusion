#!/usr/bin/env python3
"""
table1_combined_concat_crossattn.py — Train on all 7 methods combined, eval per method.

Contrast with per-method isolated training to see if a unified model generalises
better or worse than specialised models.

Architectures:
  1. rPPG-only 1D ResNet — hardcoded from the companion paper's eval18.json (earlier corpus version) (not re-run)
  2. ConcatFusionMLP     — trained fresh
  3. CrossAttnFusion     — trained fresh

Training: all real train + all fake train (all 7 methods, 41 identities)
Eval:     per method — that method's val+test fakes + all real val+test (18 identities)
          Overall   — all fakes val+test + all real val+test
5 seeds, mean ± std AUC.

Results → data/results/{M_DD_YYYY}/full_training_comparison.json
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

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS    = [42, 7, 123, 999, 2024]
DCT_COLS = [f"dct_{i}" for i in range(64)]

METHODS = [
    "AniTalker", "EchoMimic", "EDTalk", "FLOAT",
    "IP_LAP", "Real3DPortrait", "SadTalker",
]

# ── Hardcoded baselines ────────────────────────────────────────────────────────

# Companion paper (earlier corpus version) — 1D ResNet, per-method isolated training, 5 seeds
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

# Concat FusionMLP, per-method isolated training, 5 seeds
CONCAT_ISOLATED = {
    "AniTalker":      (0.9780, 0.0009),
    "EchoMimic":      (0.9928, 0.0004),
    "EDTalk":         (0.9905, 0.0005),
    "FLOAT":          (0.9871, 0.0003),
    "IP_LAP":         (0.8541, 0.0045),
    "Real3DPortrait": (0.9982, 0.0003),
    "SadTalker":      (0.9130, 0.0020),
    "Overall":        (0.8907, 0.0028),
}

# CrossAttn, per-method isolated training, 5 seeds
CROSSATTN_ISOLATED = {
    "AniTalker":      (0.9450, 0.0042),
    "EchoMimic":      (0.9841, 0.0026),
    "EDTalk":         (0.9559, 0.0016),
    "FLOAT":          (0.9791, 0.0029),
    "IP_LAP":         (0.7673, 0.0089),
    "Real3DPortrait": (0.9873, 0.0021),
    "SadTalker":      (0.7060, 0.0098),
    "Overall":        (0.8102, 0.0040),
}


# ── Models ────────────────────────────────────────────────────────────────────

class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.conv1    = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1      = nn.BatchNorm1d(out_ch)
        self.conv2    = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2      = nn.BatchNorm1d(out_ch)
        self.drop     = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                          nn.BatchNorm1d(out_ch))
            if stride != 1 or in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        out = self.drop(torch.relu(self.bn1(self.conv1(x))))
        return torch.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class ResNet1D(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.stage1 = nn.Sequential(BasicBlock1D(32,  32,  dropout=dropout),
                                    BasicBlock1D(32,  32,  dropout=dropout))
        self.stage2 = nn.Sequential(BasicBlock1D(32,  64,  stride=2, dropout=dropout),
                                    BasicBlock1D(64,  64,  dropout=dropout))
        self.stage3 = nn.Sequential(BasicBlock1D(64,  128, stride=2, dropout=dropout),
                                    BasicBlock1D(128, 128, dropout=dropout))
        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.fc     = nn.Linear(128, 1)

    def forward(self, w, _d=None):
        if w.dim() == 2: w = w.unsqueeze(1)
        return self.fc(self.gap(self.stage3(self.stage2(self.stage1(self.stem(w))))).squeeze(-1)).squeeze(-1)


class ConcatFusion(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.rppg_enc = nn.Sequential(
            nn.Linear(160, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout))
        self.dct_enc  = nn.Sequential(
            nn.Linear(64, 64),  nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout))
        self.fusion   = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, w, d):
        return self.fusion(torch.cat([self.rppg_enc(w), self.dct_enc(d)], dim=1)).squeeze(-1)


class CrossAttnFusion(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.rppg_patch = nn.Conv1d(1, 64, kernel_size=8, stride=8)
        self.dct_proj   = nn.Linear(64, 512)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=64, num_heads=4, batch_first=True, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, w, d):
        q = self.rppg_patch(w.unsqueeze(1)).transpose(1, 2)
        k = self.dct_proj(d).reshape(d.size(0), 8, 64)
        out, _ = self.cross_attn(q, k, k)
        return self.classifier(out.mean(dim=1)).squeeze(-1)


ARCHS = {
    "Concat":    (ConcatFusion,    {"dropout": 0.3}),
    "CrossAttn": (CrossAttnFusion, {"dropout": 0.3}),
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def zscore_wave(w):
    return (w - w.mean()) / (w.std() + 1e-6)


def load_arrays(df, wave_root, dct_lookup, needs_dct=True):
    waves, dcts, labels = [], [], []
    for _, row in df.iterrows():
        vid   = row["video_id"]
        wpath = wave_root / row["class"] / f"{vid}.npy"
        if not wpath.exists():
            continue
        if needs_dct and vid not in dct_lookup:
            continue
        w = np.load(wpath).astype(np.float32)
        waves.append(zscore_wave(w))
        if needs_dct:
            dcts.append(dct_lookup[vid])
        labels.append(float(row["class"] == "fake"))
    if not waves:
        return None, None, None
    X_w = np.stack(waves)
    X_d = np.stack(dcts) if needs_dct else np.zeros((len(waves), 64), dtype=np.float32)
    return X_w, X_d, np.array(labels, dtype=np.float32)


def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    i   = np.argmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(arch_cls, arch_kwargs, X_wave_tr, X_dct_tr, y_tr,
                seed, lr=1e-3, wd=5e-4, epochs=30, batch=64):
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler()
    Xd_tr  = scaler.fit_transform(X_dct_tr).astype(np.float32)

    pw   = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)]).to(DEVICE)
    mdl  = arch_cls(**arch_kwargs).to(DEVICE)
    opt  = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=wd)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    Xw_t = torch.from_numpy(X_wave_tr).to(DEVICE)
    Xd_t = torch.from_numpy(Xd_tr).to(DEVICE)
    yt   = torch.from_numpy(y_tr).to(DEVICE)

    for _ in range(epochs):
        mdl.train()
        perm = torch.randperm(len(Xw_t), device=DEVICE)
        for s in range(0, len(Xw_t), batch):
            ix = perm[s:s + batch]
            opt.zero_grad()
            crit(mdl(Xw_t[ix], Xd_t[ix]), yt[ix]).backward()
            opt.step()

    return mdl, scaler


def eval_model(mdl, scaler, X_wave_ev, X_dct_ev, y_ev):
    Xd_ev = scaler.transform(X_dct_ev).astype(np.float32)
    Xw_e  = torch.from_numpy(X_wave_ev)
    Xd_e  = torch.from_numpy(Xd_ev)
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


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--wd",     type=float, default=5e-4)
    p.add_argument("--epochs", type=int,   default=30)
    p.add_argument("--batch",  type=int,   default=64)
    p.add_argument("--arch",     default=None, help="'Concat' or 'CrossAttn' only.")
    p.add_argument("--dct-file", default=None,
                   help="DCT CSV to use (default: dct_features.csv)")
    p.add_argument("--out-file", default=None,
                   help="Output JSON filename (default: full_training_comparison.json)")
    return p.parse_args()


def main():
    args      = parse_args()
    data_root = Path(args.data_root)
    wave_root = data_root / "waveforms"

    today       = datetime.now().strftime("%-m_%d_%Y")
    results_dir = data_root / "results" / today
    results_dir.mkdir(parents=True, exist_ok=True)
    dct_file = Path(args.dct_file) if args.dct_file else data_root / "dct_features.csv"
    out_name = args.out_file or "full_training_comparison.json"
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
    print(f"Loaded {len(dct_lookup)} DCT vectors, {len(split_df)} split rows")

    # ── Build fixed train / per-method eval splits ─────────────────────────────
    train_df = split_df[split_df["split"] == "train"]
    real_eval = split_df[(split_df["class"] == "real") &
                         (split_df["split"].isin(["val", "test"]))]

    # Per-method eval: method_fakes_eval + real_eval
    eval_sets = {}
    for method in METHODS:
        method_eval = split_df[(split_df["class"] == "fake") &
                               (split_df["method"] == method) &
                               (split_df["split"].isin(["val", "test"]))]
        eval_sets[method] = pd.concat([real_eval, method_eval])
    eval_sets["Overall"] = split_df[split_df["split"].isin(["val", "test"])]

    # ── Resume ────────────────────────────────────────────────────────────────
    results = {}
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        print(f"Resuming — done: {list(results.keys())}")

    arch_targets = list(ARCHS.keys()) if args.arch is None else [args.arch]

    for arch_name in arch_targets:
        if arch_name in results:
            print(f"\n[SKIP] {arch_name} — already done")
            continue

        arch_cls, arch_kwargs = ARCHS[arch_name]
        needs_dct = True  # both fusion archs need DCT

        print(f"\n{'='*64}")
        print(f"  ARCHITECTURE: {arch_name}")
        print(f"{'='*64}")

        # Load full training arrays once (shared across seeds)
        print("  Loading training data (all methods combined)...")
        X_wave_tr, X_dct_tr, y_tr = load_arrays(train_df, wave_root, dct_lookup, needs_dct)
        if X_wave_tr is None:
            print("  SKIP — no training data")
            continue

        n_real_tr = int((y_tr == 0).sum())
        n_fake_tr = int((y_tr == 1).sum())
        print(f"  Train: {n_real_tr} real + {n_fake_tr} fake  "
              f"(ratio {n_fake_tr/max(n_real_tr,1):.2f}:1)  "
              f"pos_weight={n_real_tr/max(n_fake_tr,1):.3f}")

        # Pre-load all eval arrays (reuse across seeds)
        print("  Loading eval data per method...")
        eval_arrays = {}
        for label, edf in eval_sets.items():
            Xw, Xd, y = load_arrays(edf, wave_root, dct_lookup, needs_dct)
            if Xw is not None:
                eval_arrays[label] = (Xw, Xd, y)
                r, f = int((y==0).sum()), int((y==1).sum())
                print(f"    {label:<16}: {r} real + {f} fake")

        # Per-seed training + eval
        seed_aucs = {label: [] for label in eval_arrays}
        t_arch = time.time()

        for seed in SEEDS:
            t0 = time.time()
            print(f"\n  [seed={seed}] training...")
            mdl, scaler = train_model(
                arch_cls, arch_kwargs, X_wave_tr, X_dct_tr, y_tr,
                seed=seed, lr=args.lr, wd=args.wd,
                epochs=args.epochs, batch=args.batch,
            )
            print(f"  [seed={seed}] trained in {time.time()-t0:.0f}s — evaluating...")

            for label, (Xw, Xd, y) in eval_arrays.items():
                auc, eer = eval_model(mdl, scaler, Xw, Xd, y)
                seed_aucs[label].append(auc)
                print(f"    {label:<16} AUC={auc:.4f}  EER={eer*100:.1f}%")

        # Aggregate
        arch_results = {}
        for label, aucs in seed_aucs.items():
            arch_results[label] = {
                "mean_auc":  float(np.mean(aucs)),
                "std_auc":   float(np.std(aucs)),
                "seed_aucs": {str(s): float(a) for s, a in zip(SEEDS, aucs)},
            }

        results[arch_name] = arch_results
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  {arch_name} done in {time.time()-t_arch:.0f}s")

    # ── Comparison table ───────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  FULL TRAINING COMPARISON  (train=all methods, eval=per method, 5 seeds)")
    print(f"{'='*90}")
    print(f"  {'Method':<16} {'rPPG-only':>14} {'Concat(full)':>14} "
          f"{'CrossAttn(full)':>17} {'Best':>12}")
    print("  " + "-"*76)

    row_order = [
        "Real3DPortrait", "EDTalk", "FLOAT", "SadTalker",
        "EchoMimic", "AniTalker", "IP_LAP", "Overall",
    ]
    for label in row_order:
        rppg_m, rppg_s = RPPG_ONLY[label]

        cat_m = results["Concat"][label]["mean_auc"]    if "Concat"    in results and label in results["Concat"]    else float("nan")
        cat_s = results["Concat"][label]["std_auc"]     if "Concat"    in results and label in results["Concat"]    else float("nan")
        ca_m  = results["CrossAttn"][label]["mean_auc"] if "CrossAttn" in results and label in results["CrossAttn"] else float("nan")
        ca_s  = results["CrossAttn"][label]["std_auc"]  if "CrossAttn" in results and label in results["CrossAttn"] else float("nan")

        candidates = {"rPPG-only": rppg_m}
        if not np.isnan(cat_m): candidates["Concat(full)"]    = cat_m
        if not np.isnan(ca_m):  candidates["CrossAttn(full)"] = ca_m
        best = max(candidates, key=candidates.get)

        cat_str = f"{cat_m:.3f}±{cat_s:.3f}" if not np.isnan(cat_m) else "pending"
        ca_str  = f"{ca_m:.3f}±{ca_s:.3f}"   if not np.isnan(ca_m)  else "pending"
        print(f"  {label:<16} {rppg_m:.3f}±{rppg_s:.3f}   "
              f"{cat_str:>14}   {ca_str:>17}   {best:>12}")

    # ── Delta vs isolated ─────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  DELTA: full training vs per-method isolated")
    print(f"  (positive = full training helps, negative = isolated was better)")
    print(f"{'='*90}")
    print(f"  {'Method':<16} {'Δ Concat':>12} {'Δ CrossAttn':>14}")
    print("  " + "-"*44)

    for label in row_order:
        cat_full = results["Concat"][label]["mean_auc"]    if "Concat"    in results and label in results["Concat"]    else None
        ca_full  = results["CrossAttn"][label]["mean_auc"] if "CrossAttn" in results and label in results["CrossAttn"] else None

        d_cat = f"{cat_full - CONCAT_ISOLATED[label][0]:+.3f}"    if cat_full is not None else "pending"
        d_ca  = f"{ca_full  - CROSSATTN_ISOLATED[label][0]:+.3f}" if ca_full  is not None else "pending"
        print(f"  {label:<16} {d_cat:>12} {d_ca:>14}")

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
