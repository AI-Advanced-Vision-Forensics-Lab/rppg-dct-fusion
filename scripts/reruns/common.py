"""
common.py — Shared utilities, model definitions, and data loaders
for this paper's reruns.  All models here are exact copies from the
original scripts (no modifications to architectures or HPs).

Original sources:
  ResNet1D     → src/run_per_method_isolated_rebaseline.py / per_method_full_experiments.py
  LinearRPPG   → table1_concat_fusion.py (FusionMLP.rppg_enc, no fusion head)
  DctMLP       → table1_dct_only.py / table1_concat_fusion.py
  FusionMLP    → table1_concat_fusion.py
  ResNetFusion → new (R5) — ResNet trunk + DCT branch, same fusion head as FusionMLP
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

# ── Paths ─────────────────────────────────────────────────────────────────────
import argparse as _argparse
_ap = _argparse.ArgumentParser(description="rerun for this paper (see README)")
_ap.add_argument("--data-root", default="data",
                 help="data folder laid out as described in the README (default: ./data)")
_ap.add_argument("--out-dir", default=None,
                 help="parent folder for run outputs (default: <data-root>/results/reruns)")
_args = _ap.parse_args()
DATA_ROOT   = Path(_args.data_root)
OUT_ROOT    = Path(_args.out_dir) if _args.out_dir else DATA_ROOT / "results" / "reruns"
SPLIT_FILE  = DATA_ROOT / "dataset_split.csv"
WAVE_ROOT   = DATA_ROOT / "waveforms" / "CelebDF"  # TalkingFace / Celeb-real subdirs
WAVE_FAKE   = DATA_ROOT / "waveforms" / "CelebDF" / "TalkingFace"  # per-method fake waveforms
DCT_FILE    = DATA_ROOT / "dct_features.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS  = [42, 7, 123, 999, 2024]
METHODS = ["AniTalker", "EchoMimic", "EDTalk", "FLOAT",
           "IP_LAP", "Real3DPortrait", "SadTalker"]

# ── Model architectures ───────────────────────────────────────────────────────

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
        out = self.drop(F.relu(self.bn1(self.conv1(x))))
        return F.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class ResNet1D(nn.Module):
    """240,161-param 1D ResNet.  dropout=0.5 for rPPG-only baseline."""
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

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        return self.fc(self.gap(
            self.stage3(self.stage2(self.stage1(self.stem(x))))
        ).squeeze(-1)).squeeze(-1)

    @staticmethod
    def param_count():
        return sum(p.numel() for p in ResNet1D().parameters())


class LinearRPPG(nn.Module):
    """10,432-param linear rPPG encoder (FusionMLP rppg_enc branch, standalone).
    Matches table1_concat_fusion.py FusionMLP.rppg_enc exactly."""
    def __init__(self, dropout=0.3):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(160, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout))
        self.fc  = nn.Linear(64, 1)

    def forward(self, x):
        return self.fc(self.enc(x)).squeeze(-1)

    @staticmethod
    def param_count():
        return sum(p.numel() for p in LinearRPPG().parameters())


class DctMLP(nn.Module):
    """4,353-param DCT-only MLP.
    Matches table1_dct_only.py exactly."""
    def __init__(self, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FusionMLP(nn.Module):
    """23,041-param concat-fusion MLP.
    Matches table1_concat_fusion.py FusionMLP exactly."""
    def __init__(self, dropout=0.3):
        super().__init__()
        self.rppg_enc = nn.Sequential(
            nn.Linear(160, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout))
        self.dct_enc  = nn.Sequential(
            nn.Linear(64,  64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout))
        self.fusion   = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout))
        self.fc       = nn.Linear(64, 1)

    def forward(self, wave, dct):
        h_r = self.rppg_enc(wave)
        h_d = self.dct_enc(dct)
        return self.fc(self.fusion(torch.cat([h_r, h_d], dim=1))).squeeze(-1)


class ResNetFusion(nn.Module):
    """~263K-param fusion: ResNet1D trunk (output dim 128) + DCT branch.
    R5 variant — replaces FusionMLP.rppg_enc with a full ResNet1D trunk
    (dropping ResNet1D.fc, using the 128-dim GAP output instead)."""
    def __init__(self, dropout=0.3):
        super().__init__()
        # ResNet trunk (same as ResNet1D but without .fc, dropout=0.3 to match fusion)
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
        # DCT branch (same as FusionMLP.dct_enc)
        self.dct_enc = nn.Sequential(
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout))
        # Fusion head: 128 (ResNet) + 64 (DCT) = 192
        self.fusion  = nn.Sequential(
            nn.Linear(192, 64), nn.ReLU(), nn.Dropout(dropout))
        self.fc      = nn.Linear(64, 1)

    def forward(self, wave, dct):
        if wave.dim() == 2: wave = wave.unsqueeze(1)
        h_r = self.gap(self.stage3(self.stage2(self.stage1(self.stem(wave))))).squeeze(-1)
        h_d = self.dct_enc(dct)
        return self.fc(self.fusion(torch.cat([h_r, h_d], dim=1))).squeeze(-1)

    @staticmethod
    def param_count():
        return sum(p.numel() for p in ResNetFusion().parameters())


# ── Helpers ───────────────────────────────────────────────────────────────────

def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-6)


def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    i = np.argmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2


def load_split(method=None):
    """Load dataset_split.csv; optionally filter to one method's fakes + all reals.
    Returns DataFrame with columns: video_id, class, identity, split.
    """
    df = pd.read_csv(SPLIT_FILE)
    if method is not None:
        fake_mask = df['video_id'].str.startswith(method + '__')
        df = df[(df['class'] == 'real') | fake_mask].reset_index(drop=True)
    return df


def load_waveforms(video_ids, real_ids_set=None):
    """Load waveforms for a list of video_ids.  Returns dict {video_id: np.array(160,)}."""
    waves = {}
    for vid in video_ids:
        if '__' in vid:
            # fake: AniTalker__id0_0000_test_...
            method, stem = vid.split('__', 1)
            p = WAVE_FAKE / method / (stem + '.npy')
        else:
            # real: id0_0000_w3
            p = WAVE_ROOT / 'Celeb-real' / (vid + '.npy')
        if p.exists():
            w = np.load(p).astype(np.float32)
            if len(w) >= 160:
                waves[vid] = zscore(w[:160])
            else:
                waves[vid] = None
        else:
            waves[vid] = None
    return waves


def build_arrays(df, wave_cache, dct_lookup, require_dct=False):
    """Build (waves, dcts, labels) numpy arrays from a DataFrame subset.
    If require_dct=False, waves-only rows are filled with dct=zeros.
    Returns waves(N,160), dcts(N,64), labels(N,), video_ids list.
    """
    waves_list, dcts_list, labels_list, ids_list = [], [], [], []
    for _, row in df.iterrows():
        vid = row['video_id']
        w = wave_cache.get(vid)
        if w is None:
            continue
        d = dct_lookup.get(vid, None)
        if require_dct and d is None:
            continue
        if d is None:
            d = np.zeros(64, dtype=np.float32)
        waves_list.append(w)
        dcts_list.append(d)
        labels_list.append(float(row['label']))
        ids_list.append(vid)
    return (np.stack(waves_list).astype(np.float32),
            np.stack(dcts_list).astype(np.float32),
            np.array(labels_list, dtype=np.float32),
            ids_list)


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_seed(model, wave_tr, dct_tr, label_tr, seed,
                   lr=1e-3, wd=5e-4, epochs=30, batch=64, use_dct=False):
    """Train model for one seed.  Returns trained model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_fake = int(label_tr.sum())
    n_real = int((label_tr == 0).sum())
    pos_w  = torch.tensor([n_real / max(n_fake, 1)], device=DEVICE)
    crit   = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    W = torch.from_numpy(wave_tr)
    D = torch.from_numpy(dct_tr)
    L = torch.from_numpy(label_tr)
    loader = DataLoader(TensorDataset(W, D, L), batch_size=batch,
                        shuffle=True, drop_last=False)

    model.train()
    for ep in range(epochs):
        for wb, db, lb in loader:
            wb, db, lb = wb.to(DEVICE), db.to(DEVICE), lb.to(DEVICE)
            opt.zero_grad()
            if use_dct:
                out = model(wb, db)
            else:
                out = model(wb)
            crit(out, lb).backward()
            opt.step()
    return model


def eval_model(model, wave_ev, dct_ev, label_ev, use_dct=False):
    """Evaluate model; returns window-level AUC and EER."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(wave_ev), 256):
            wb = torch.from_numpy(wave_ev[i:i+256]).to(DEVICE)
            db = torch.from_numpy(dct_ev[i:i+256]).to(DEVICE)
            if use_dct:
                logits = model(wb, db)
            else:
                logits = model(wb)
            preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    preds = np.array(preds)
    auc = roc_auc_score(label_ev, preds)
    eer = compute_eer(label_ev.astype(int), preds)
    return auc, eer


def run_experiment(model_factory, model_kwargs, wave_tr, dct_tr, label_tr,
                   wave_ev, dct_ev, label_ev, seeds, use_dct=False,
                   lr=1e-3, wd=5e-4, epochs=30, batch=64):
    """Run one experiment across all seeds.
    Returns dict with mean_auc, std_auc, mean_eer, seed_aucs.
    """
    seed_aucs, seed_eers = [], []
    for seed in seeds:
        model = model_factory(**model_kwargs).to(DEVICE)
        model = train_one_seed(model, wave_tr, dct_tr, label_tr, seed,
                               lr=lr, wd=wd, epochs=epochs, batch=batch,
                               use_dct=use_dct)
        auc, eer = eval_model(model, wave_ev, dct_ev, label_ev, use_dct=use_dct)
        seed_aucs.append(float(auc))
        seed_eers.append(float(eer))
    aucs = np.array(seed_aucs)
    return {
        "mean_auc": float(aucs.mean()),
        "std_auc":  float(aucs.std()),
        "mean_eer": float(np.mean(seed_eers)),
        "seed_aucs": {str(s): float(a) for s, a in zip(seeds, seed_aucs)},
    }


# ── Config serialization ──────────────────────────────────────────────────────

def make_config(model_name, model_kwargs, corpus_info, extra=None):
    """Build config dict to save alongside results."""
    cfg = {
        "model_name":    model_name,
        "model_kwargs":  model_kwargs,
        "corpus":        "FINAL (dataset_split.csv)",
        "split_file":    str(SPLIT_FILE),
        "eval_set":      "val+test (18 identities)",
        "n_eval_real":   corpus_info.get("n_eval_real"),
        "n_train_real":  corpus_info.get("n_train_real"),
        "seeds":         SEEDS,
        "lr":            1e-3,
        "wd":            5e-4,
        "epochs":        30,
        "batch":         64,
        "weighted_bce":  True,
        "window_level_auc": True,
        "git_commit":    "NO_GIT_REPO",
        "device":        str(DEVICE),
    }
    if extra:
        cfg.update(extra)
    return cfg
