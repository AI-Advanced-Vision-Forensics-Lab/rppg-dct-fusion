#!/usr/bin/env python3
"""
table1_gated_fusion.py — Learned scalar gate: α·h_rppg + (1−α)·h_dct.

Gate hypothesis: the model should learn high α (rely on rPPG) for SadTalker
where DCT adds noise, and low/mid α (rely more on DCT) for IP_LAP where DCT
adds the most signal.

Protocol: identical to table1_concat_fusion.py
  - Per-method isolated training (real + that method's fakes)
  - 5 seeds, 30 epochs, AdamW lr=1e-3 wd=5e-4, batch=64, weighted BCE
  - 18-identity eval (val+test), mean ± std AUC
  - StandardScaler on DCT fit on train only, per seed
  - Records mean α across eval set per method per seed

Results → data/results/{M_DD_YYYY}/gated_fusion_experiments.json
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

class GatedFusion(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.rppg_enc = nn.Sequential(
            nn.Linear(160, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
        )
        self.dct_enc = nn.Sequential(
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
        )
        # Gate: takes both encoded representations, outputs scalar α per sample
        self.gate = nn.Sequential(
            nn.Linear(128, 1), nn.Sigmoid(),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, w, d, return_alpha=False):
        h_rppg = self.rppg_enc(w)          # (B, 64)
        h_dct  = self.dct_enc(d)           # (B, 64)
        alpha  = self.gate(torch.cat([h_rppg, h_dct], dim=1))  # (B, 1)
        h      = alpha * h_rppg + (1 - alpha) * h_dct          # (B, 64)
        out    = self.classifier(h).squeeze(-1)
        if return_alpha:
            return out, alpha.squeeze(-1)
        return out


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
    i   = np.argmin(np.abs(fnr - fpr))
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
    mdl  = GatedFusion(dropout=dropout).to(DEVICE)
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

    # Eval: collect probs and alpha values
    mdl.eval()
    probs, alphas = [], []
    with torch.no_grad():
        for s in range(0, len(Xw_e), 512):
            logits, alpha = mdl(
                Xw_e[s:s+512].to(DEVICE),
                Xd_e[s:s+512].to(DEVICE),
                return_alpha=True,
            )
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            alphas.extend(alpha.cpu().numpy().tolist())

    probs  = np.array(probs)
    alphas = np.array(alphas)
    auc    = float(roc_auc_score(y_ev, probs))
    eer    = compute_eer(y_ev, probs)
    mean_alpha = float(alphas.mean())

    return auc, eer, mean_alpha


def run_experiment(label, train_df, eval_df, wave_root, dct_lookup, seeds, args):
    X_wave_tr, X_dct_tr, y_tr = load_arrays(train_df, wave_root, dct_lookup)
    X_wave_ev, X_dct_ev, y_ev = load_arrays(eval_df,  wave_root, dct_lookup)

    if X_wave_tr is None or X_wave_ev is None:
        print(f"  [{label}] SKIP — no data")
        return None

    n_real_tr = int((y_tr == 0).sum())
    n_fake_tr = int((y_tr == 1).sum())
    print(f"  [{label}] train: {n_real_tr} real + {n_fake_tr} fake  |  "
          f"eval: {int((y_ev==0).sum())} real + {int((y_ev==1).sum())} fake  |  "
          f"pos_weight={n_real_tr/max(n_fake_tr,1):.3f}")

    aucs, eers, mean_alphas = [], [], []
    t0 = time.time()
    for seed in seeds:
        auc, eer, mean_alpha = run_one_seed(
            X_wave_tr, X_dct_tr, y_tr,
            X_wave_ev, X_dct_ev, y_ev,
            seed=seed, lr=args.lr, wd=args.wd,
            epochs=args.epochs, batch=args.batch,
        )
        aucs.append(auc)
        eers.append(eer)
        mean_alphas.append(mean_alpha)
        print(f"    seed={seed}  AUC={auc:.4f}  EER={eer*100:.1f}%  α={mean_alpha:.4f}")

    mean_auc   = float(np.mean(aucs))
    std_auc    = float(np.std(aucs))
    mean_alpha_agg = float(np.mean(mean_alphas))
    print(f"  → AUC={mean_auc:.4f}±{std_auc:.4f}  α={mean_alpha_agg:.4f}  "
          f"[{time.time()-t0:.0f}s]")

    return {
        "mean_auc":     mean_auc,
        "std_auc":      std_auc,
        "mean_eer":     float(np.mean(eers)),
        "mean_alpha":   mean_alpha_agg,
        "seed_aucs":    {str(s): float(a) for s, a in zip(seeds, aucs)},
        "seed_alphas":  {str(s): float(a) for s, a in zip(seeds, mean_alphas)},
        "n_train_real": n_real_tr,
        "n_train_fake": n_fake_tr,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--wd",     type=float, default=5e-4)
    p.add_argument("--epochs", type=int,   default=30)
    p.add_argument("--batch",  type=int,   default=64)
    p.add_argument("--method", default=None)
    return p.parse_args()


def main():
    args      = parse_args()
    data_root = Path(args.data_root)
    wave_root = data_root / "waveforms"

    today       = datetime.now().strftime("%-m_%d_%Y")
    results_dir = data_root / "results" / today
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "gated_fusion_experiments.json"

    print(f"Device:  {DEVICE}")
    print(f"Results: {out_path}")
    print(f"lr={args.lr}  wd={args.wd}  epochs={args.epochs}  batch={args.batch}")

    split_df = pd.read_csv(data_root / "dataset_split.csv")
    dct_df   = pd.read_csv(data_root / "dct_features.csv")

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
        print(f"Resuming — done: {list(results.keys())}")

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

    print(f"\n{'='*88}")
    print(f"  rPPG-only | Concat | GatedFusion  (5 seeds, per-method training)")
    print(f"{'='*88}")
    print(f"  {'Method':<16} {'rPPG-only':>12} {'Concat':>12} {'Gated':>10} "
          f"{'Mean α':>8} {'Best':>14}")
    print("  " + "-"*72)

    row_order = [
        "Real3DPortrait", "EDTalk", "FLOAT", "SadTalker",
        "EchoMimic", "AniTalker", "IP_LAP", "Overall",
    ]
    alpha_by_method = {}
    for label in row_order:
        if label not in results:
            continue

        rppg_m, rppg_s  = RPPG_ONLY[label]
        cat_m,  cat_s   = CONCAT[label]
        gated_m          = results[label]["mean_auc"]
        gated_s          = results[label]["std_auc"]
        alpha            = results[label]["mean_alpha"]
        alpha_by_method[label] = alpha

        candidates = {"rPPG-only": rppg_m, "Concat": cat_m, "Gated": gated_m}
        best = max(candidates, key=candidates.get)

        flags = []
        if label == "SadTalker": flags.append("← α→1?")
        if label == "IP_LAP":    flags.append("← α≈0.5?")
        flag_str = "  " + " ".join(flags) if flags else ""

        print(f"  {label:<16} "
              f"{rppg_m:.3f}±{rppg_s:.3f}  "
              f"{cat_m:.3f}±{cat_s:.3f}  "
              f"{gated_m:.3f}±{gated_s:.3f}  "
              f"{alpha:.4f}  "
              f"{best:>14}{flag_str}")

    # ── Hypothesis checks ─────────────────────────────────────────────────────
    print(f"\n{'='*88}")
    print(f"  HYPOTHESIS CHECKS")
    print(f"{'='*88}")

    sad_auc   = results.get("SadTalker", {}).get("mean_auc", 0)
    ip_auc    = results.get("IP_LAP",    {}).get("mean_auc", 0)
    sad_alpha = alpha_by_method.get("SadTalker", 0.5)
    ip_alpha  = alpha_by_method.get("IP_LAP",    0.5)

    checks = [
        (
            "SadTalker recovery   (Gated >= 0.940)",
            sad_auc >= 0.940,
            f"Gated={sad_auc:.4f}  target=0.940  concat={CONCAT['SadTalker'][0]:.4f}  rPPG={RPPG_ONLY['SadTalker'][0]:.4f}",
        ),
        (
            "IP_LAP maintained    (Gated >= 0.850)",
            ip_auc >= 0.850,
            f"Gated={ip_auc:.4f}  target=0.850  concat={CONCAT['IP_LAP'][0]:.4f}",
        ),
        (
            "Gate learned suppression (α_SadTalker > α_IP_LAP)",
            sad_alpha > ip_alpha,
            f"α_SadTalker={sad_alpha:.4f}  α_IP_LAP={ip_alpha:.4f}  Δ={sad_alpha-ip_alpha:+.4f}",
        ),
    ]

    for desc, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"\n  [{status}] {desc}")
        print(f"         {detail}")

    n_pass = sum(p for _, p, _ in checks)
    print(f"\n  {n_pass}/{len(checks)} hypotheses supported")

    # Alpha ranking (most rPPG-reliant → most DCT-reliant)
    if len(alpha_by_method) > 1:
        print(f"\n  Alpha ranking (high α = relies on rPPG, low α = relies on DCT):")
        for label, alpha in sorted(alpha_by_method.items(), key=lambda x: -x[1]):
            bar = "█" * int(alpha * 20)
            print(f"    {label:<16} {alpha:.4f}  {bar}")

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
