#!/usr/bin/env python3
"""
make_fig2_grid.py — Figure 2 for Paper 2.

3×2 grid. Rows: Real, IP-LAP, SadTalker (all identity id48).
Col 1: z-scored rPPG waveform (160 samples). Col 2: lip-region DCT
log-magnitude map (8×8, dct2d[:8,:8] directly — not zigzag).

Clip selection: closest to per-method median DCT L2-norm.
  Real:      id48_0002_w0                              (dist=38.57)
  IP-LAP:    IP_LAP__id48_0005_test_id01460_t0i-doPmN9c  (dist=15.71)
  SadTalker: SadTalker__id48_0001_test_id01509_asrGhE2enzQ (dist=12.01)

ICASSP constraints: pdf.fonttype=42, ps.fonttype=42, grayscale only,
bbox_inches='tight', pad_inches=0.02, no figure title.
"""

import sys
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["font.family"]  = "DejaVu Sans"
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.fft import dctn

# ── Paths ─────────────────────────────────────────────────────────────────────
import argparse as _argparse
_ap = _argparse.ArgumentParser(description="Figure 2: example rPPG waveforms and lip-DCT maps")
_ap.add_argument("--data-root", default="data",
                 help="data folder laid out as described in the README (default: ./data)")
_ap.add_argument("--out-dir", default="figures", help="folder for the figure (default: ./figures)")
_args = _ap.parse_args()
DATA     = Path(_args.data_root)
WV_REAL  = DATA / "waveforms/CelebDF/Celeb-real"
WV_FAKE  = DATA / "waveforms/CelebDF/TalkingFace"
FR_REAL  = DATA / "frames/celeb/Celeb-real"
FR_FAKE  = DATA / "frames/celeb/TalkingFace"
LANDMARK = DATA / "face_landmarker.task"
FIG_DIR  = Path(_args.out_dir)
FIG_DIR.mkdir(exist_ok=True)

# ── DCT parameters (identical to extract_dct_features.py) ────────────────────
LIP_INDICES = [61, 291, 0, 17]
CROP_SIZE   = 64
DCT_BLOCK   = 8
PAD         = 20

# ── Clip registry ─────────────────────────────────────────────────────────────
ROWS = [
    {
        "label":    "Real",
        "waveform": WV_REAL / "id48_0002_w0.npy",
        "fr_dir":   FR_REAL,
        "fr_prefix": "id48_0002",
    },
    {
        "label":    "IP-LAP",
        "waveform": WV_FAKE / "IP_LAP/id48_0005_test_id01460_t0i-doPmN9c.npy",
        "fr_dir":   FR_FAKE / "IP_LAP",
        "fr_prefix": "id48_0005_test_id01460_t0i-doPmN9c",
    },
    {
        "label":    "SadTalker",
        "waveform": WV_FAKE / "SadTalker/id48_0001_test_id01509_asrGhE2enzQ.npy",
        "fr_dir":   FR_FAKE / "SadTalker",
        "fr_prefix": "id48_0001_test_id01509_asrGhE2enzQ",
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_frame(fr_dir, prefix, preference=5):
    candidates = [preference] + [i for i in range(10) if i != preference]
    for idx in candidates:
        p = fr_dir / f"{prefix}_{idx:02d}.jpg"
        if p.exists():
            bgr = cv2.imread(str(p))
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), idx, p
    raise FileNotFoundError(f"No readable frame: {fr_dir}/{prefix}_*.jpg")


def compute_dct_block(frame_rgb, landmarker):
    h, w = frame_rgb.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_img)
    if not result.face_landmarks:
        return None, "no face detected"
    lm = result.face_landmarks[0]
    xs = [lm[i].x * w for i in LIP_INDICES]
    ys = [lm[i].y * h for i in LIP_INDICES]
    x1 = max(0,     int(min(xs)) - PAD)
    y1 = max(0,     int(min(ys)) - PAD)
    x2 = min(w - 1, int(max(xs)) + PAD)
    y2 = min(h - 1, int(max(ys)) + PAD)
    if x2 <= x1 or y2 <= y1:
        return None, "degenerate bbox"
    gray  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    patch = cv2.resize(gray[y1:y2, x1:x2], (CROP_SIZE, CROP_SIZE),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    dct2d = dctn(patch, norm="ortho")
    block = dct2d[:DCT_BLOCK, :DCT_BLOCK].copy()
    return block, f"bbox=({x1},{y1},{x2},{y2})"


# ── Verify required files ─────────────────────────────────────────────────────
print("=== FILE VERIFICATION ===")
for row in ROWS:
    ok = row["waveform"].exists()
    print(f"  {'OK' if ok else 'MISSING'} waveform: {row['waveform']}")
    if not ok:
        print("STOP: missing waveform.")
        sys.exit(1)
if not LANDMARK.exists():
    print(f"STOP: missing {LANDMARK}")
    sys.exit(1)
print(f"  OK landmark model: {LANDMARK}")
print()

# ── MediaPipe ─────────────────────────────────────────────────────────────────
landmarker = mp_vision.FaceLandmarker.create_from_options(
    mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(LANDMARK)),
        num_faces=1,
    )
)

# ── Load data ─────────────────────────────────────────────────────────────────
print("=== LOADING DATA ===")
waveforms  = []
dct_blocks = []
frame_info = []

for row in ROWS:
    label = row["label"]

    # Waveform
    w = np.load(row["waveform"]).astype(np.float32)
    assert len(w) == 160, f"{label}: expected 160 samples, got {len(w)}"
    waveforms.append(w)
    print(f"  [{label}] waveform shape={w.shape}  raw min={w.min():.3f} max={w.max():.3f}")

    # Frame + DCT
    frame, fidx, fpath = load_frame(row["fr_dir"], row["fr_prefix"])
    block, status = compute_dct_block(frame, landmarker)
    if block is None:
        print(f"  [{label}] detection failed on frame {fidx:02d}: {status} — trying others")
        for alt in range(10):
            if alt == fidx:
                continue
            frame, fidx, fpath = load_frame(row["fr_dir"], row["fr_prefix"], preference=alt)
            block, status = compute_dct_block(frame, landmarker)
            if block is not None:
                print(f"  [{label}] succeeded on fallback frame {fidx:02d}")
                break
        if block is None:
            print(f"STOP: detection failed on all frames for {label}.")
            sys.exit(1)

    dct_blocks.append(block)
    frame_info.append((fpath, fidx, status))
    print(f"  [{label}] frame={fpath.name}  DCT {status}")

print()

# ── Compute display arrays ────────────────────────────────────────────────────
# Z-scored waveforms
wv_z = [(w - w.mean()) / (w.std() + 1e-6) for w in waveforms]

# Log-magnitude DCT blocks
dct_log = [np.log1p(np.abs(b)) for b in dct_blocks]

# Shared limits
wv_ymin = min(z.min() for z in wv_z)
wv_ymax = max(z.max() for z in wv_z)
dct_vmin = min(d.min() for d in dct_log)
dct_vmax = max(d.max() for d in dct_log)

print("=== SHARED LIMITS ===")
print(f"  Col 1 y-limits:  [{wv_ymin:.4f}, {wv_ymax:.4f}]")
print(f"  Col 2 vmin/vmax: [{dct_vmin:.4f}, {dct_vmax:.4f}]")
print()

print("=== RAW 8×8 LOG-MAGNITUDE ARRAYS ===")
np.set_printoptions(precision=3, suppress=True, linewidth=100)
for row, d in zip(ROWS, dct_log):
    print(f"  {row['label']}:")
    print(d)
    print()

# ── Figure layout ─────────────────────────────────────────────────────────────
# 3 rows × 3 cols: [waveform | DCT | colorbar]
# Width ratios chosen so DCT panel is roughly square given the figure height.
# With FIG_W=3.4 and these ratios the DCT column is ~1.0in wide; we set
# FIG_H so each row is ~1.1in tall → panels are approximately square.
FONTSIZE   = 8    # axis labels, column headers — matches fig3_v2_inverted
FONTSIZE_S = 7    # row labels, tick labels, colorbar — matches fig3_v2_inverted
FIG_W      = 3.4
ROW_H      = 1.10  # target row height in inches
N_ROWS     = 3
HSPACE_IN  = 0.10  # inter-row gap in inches
FIG_H      = N_ROWS * ROW_H + (N_ROWS - 1) * HSPACE_IN + 0.22 + 0.18  # top + bottom margin

fig = plt.figure(figsize=(FIG_W, FIG_H))

gs = gridspec.GridSpec(
    N_ROWS, 3,
    figure=fig,
    width_ratios=[1.9, 1.0, 0.07],
    hspace=HSPACE_IN / ROW_H,
    wspace=0.08,
)

axes_wave = [fig.add_subplot(gs[r, 0]) for r in range(N_ROWS)]
axes_dct  = [fig.add_subplot(gs[r, 1]) for r in range(N_ROWS)]
ax_cbar   = fig.add_subplot(gs[1, 2])   # one panel tall (middle row)

ROW_LABELS = [row["label"] for row in ROWS]
COL_HEADERS = ["rPPG", "Lip DCT"]

# ── Column 1: waveforms ───────────────────────────────────────────────────────
for ri, (ax, wz, row) in enumerate(zip(axes_wave, wv_z, ROWS)):
    ax.plot(wz, color="black", linewidth=0.8)
    ax.set_xlim(0, 159)
    ax.set_ylim(wv_ymin - 0.05, wv_ymax + 0.05)
    ax.tick_params(length=2, labelsize=FONTSIZE - 1)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    # Row label as rotated y-axis label
    ax.set_ylabel(ROW_LABELS[ri], fontsize=FONTSIZE_S, rotation=90, labelpad=4, va="center")
    # x tick labels only on bottom row
    if ri < N_ROWS - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("frame", fontsize=FONTSIZE - 1)
    # Column header on top row only
    if ri == 0:
        ax.set_title(COL_HEADERS[0], fontsize=FONTSIZE, pad=3)

# ── Column 2: DCT maps ────────────────────────────────────────────────────────
ims = []
for ri, (ax, dl) in enumerate(zip(axes_dct, dct_log)):
    im = ax.imshow(dl, cmap="gray", vmin=dct_vmin, vmax=dct_vmax,
                   aspect="equal", interpolation="nearest")
    ims.append(im)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if ri == 0:
        ax.set_title(COL_HEADERS[1], fontsize=FONTSIZE, pad=3)

# ── Shared colorbar ───────────────────────────────────────────────────────────
cbar = fig.colorbar(ims[-1], cax=ax_cbar)
cbar.ax.tick_params(labelsize=FONTSIZE_S, length=2)
cbar.set_label("log|DCT|", fontsize=FONTSIZE_S, labelpad=2)

# ── Save ──────────────────────────────────────────────────────────────────────
print("=== SAVING ===")
for ext, dpi in [("pdf", None), ("png", 300)]:
    out = FIG_DIR / f"fig2_grid.{ext}"
    kw  = dict(bbox_inches="tight", pad_inches=0.02)
    if dpi:
        kw["dpi"] = dpi
    fig.savefig(out, **kw)
    size = out.stat().st_size
    print(f"  {out}  {size:,} bytes")

plt.close(fig)
print()
print(f"Figure size set to: {FIG_W:.2f} × {FIG_H:.2f} in  (w × h)")
