#!/usr/bin/env python3
"""
extract_dct_features.py — Lip-region DCT features aligned with stride-60 rPPG windows.

For each entry in dataset_split.csv:
  - Locate source video and seek to the middle frame of that window
  - Run MediaPipe Face Mesh to detect lip landmarks (61, 291, 0, 17)
  - Crop tight bounding box + 20px padding, resize to 64×64 grayscale
  - Apply 2D DCT (scipy.fft.dctn, norm='ortho')
  - Keep top-left 8×8 block in zigzag order → 64 values
  - Save one row per video_id to data/dct_features.csv

Resumable: already-processed video_ids are skipped.

Usage:
    python3 scripts/02_extract_lip_dct.py --data-root data
"""

import argparse
import csv
import re
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import pandas as pd
from scipy.fft import dctn

# Lip landmark indices in MediaPipe Face Mesh (468 points)
LIP_INDICES = [61, 291, 0, 17]   # left-corner, right-corner, top-lip, bottom-lip
STRIDE      = 60
N_FRAMES    = 160
PATCH_SIZE  = 64
DCT_BLOCK   = 8
PAD         = 20


# ── Zigzag scan order for 8×8 DCT block ──────────────────────────────────────
def _zigzag_indices(n=8):
    result = []
    for d in range(2 * n - 1):
        if d % 2 == 0:
            r, c = min(d, n - 1), max(0, d - n + 1)
            while r >= 0 and c < n:
                result.append((r, c)); r -= 1; c += 1
        else:
            r, c = max(0, d - n + 1), min(d, n - 1)
            while r < n and c >= 0:
                result.append((r, c)); r += 1; c -= 1
    return result

ZIGZAG = _zigzag_indices(DCT_BLOCK)   # 64 (row, col) pairs


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    return p.parse_args()


def window_index(video_id):
    """Extract window index k from real video_id '{stem}_w{k}'. Returns None for fakes."""
    m = re.search(r"_w(\d+)$", video_id)
    return int(m.group(1)) if m else None


def source_video_path(video_id, video_root):
    """Return (Path, start_frame) for a given video_id."""
    if "__" in video_id:
        # Fake: AniTalker__id0_0000_test_...
        method, stem = video_id.split("__", 1)
        for ext in (".mp4", ".avi", ".mov", ".mkv"):
            p = video_root / "fake" / method / (stem + ext)
            if p.exists():
                return p, None      # middle frame = total_frames // 2
        return None, None
    else:
        # Real: id0_0000_w3  →  stem = id0_0000, window = 3
        k    = window_index(video_id)
        stem = re.sub(r"_w\d+$", "", video_id)
        for ext in (".mp4", ".avi", ".mov", ".mkv"):
            p = video_root / "real" / (stem + ext)
            if p.exists():
                start = (k or 0) * STRIDE
                return p, start
        return None, None


def load_middle_frame(video_path, start_frame):
    """Load the middle frame of a window. start_frame=None → middle of whole video."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame is None:
        target = total // 2
    else:
        target = start_frame + N_FRAMES // 2
        target = min(target, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def lip_bbox(frame_rgb, face_landmarker):
    """Run Face Landmarker and return (x1,y1,x2,y2) bounding box of lip landmarks + padding."""
    h, w = frame_rgb.shape[:2]
    mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result  = face_landmarker.detect(mp_img)
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    xs = [lm[i].x * w for i in LIP_INDICES]
    ys = [lm[i].y * h for i in LIP_INDICES]
    x1 = max(0,     int(min(xs)) - PAD)
    y1 = max(0,     int(min(ys)) - PAD)
    x2 = min(w - 1, int(max(xs)) + PAD)
    y2 = min(h - 1, int(max(ys)) + PAD)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def dct_vector(frame_rgb, bbox):
    """Crop lip region, apply 2D DCT, return top-left 8×8 in zigzag order."""
    x1, y1, x2, y2 = bbox
    gray  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    patch = cv2.resize(gray[y1:y2, x1:x2], (PATCH_SIZE, PATCH_SIZE),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    dct2d = dctn(patch, norm="ortho")
    block = dct2d[:DCT_BLOCK, :DCT_BLOCK]
    return np.array([block[r, c] for r, c in ZIGZAG], dtype=np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args      = parse_args()
    data_root = Path(args.data_root)
    video_root = data_root / "videos"
    split_df  = pd.read_csv(data_root / "dataset_split.csv")
    out_path       = data_root / "dct_features.csv"
    fail_log_path  = data_root / "dct_extraction_failures.log"
    landmark_model = data_root / "face_landmarker.task"

    # Resume: load already-done video_ids
    done = set()
    if out_path.exists():
        existing = pd.read_csv(out_path)
        done = set(existing["video_id"].tolist())
        print(f"Resuming — {len(done)} already done")

    # MediaPipe Face Landmarker (Tasks API, v0.10+)
    face_mesh = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(landmark_model)),
            num_faces=1,
        )
    )

    fieldnames = ["video_id", "class", "split", "identity"] + [f"dct_{i}" for i in range(64)]

    n_ok, n_fail, n_skip = 0, 0, 0
    t0    = time.time()
    total = len(split_df)

    with open(out_path, "a", newline="") as f, \
         open(fail_log_path, "a") as fail_log:

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not done:
            writer.writeheader()

        for i, row in split_df.iterrows():
            vid_id = row["video_id"]

            if vid_id in done:
                n_skip += 1
                continue

            vpath, start = source_video_path(vid_id, video_root)
            if vpath is None:
                fail_log.write(f"{vid_id}\tno_video_file\n")
                n_fail += 1
                continue

            frame = load_middle_frame(vpath, start)
            if frame is None:
                fail_log.write(f"{vid_id}\tframe_read_failed\t{vpath}\n")
                n_fail += 1
                continue

            bbox = lip_bbox(frame, face_mesh)
            if bbox is None:
                fail_log.write(f"{vid_id}\tno_face_detected\t{vpath}\n")
                n_fail += 1
                continue

            vec = dct_vector(frame, bbox)

            row_out = {
                "video_id": vid_id,
                "class":    row["class"],
                "split":    row["split"],
                "identity": row["identity"],
            }
            row_out.update({f"dct_{j}": float(vec[j]) for j in range(64)})
            writer.writerow(row_out)
            f.flush()
            done.add(vid_id)
            n_ok += 1

            processed = n_ok + n_fail
            if processed % 200 == 0 or processed == 1:
                elapsed = time.time() - t0
                rate    = processed / elapsed
                eta     = (total - len(done)) / max(rate, 1e-6)
                print(f"  [{len(done):>6}/{total}] ok={n_ok} fail={n_fail} "
                      f"| {rate:.1f}/s | ETA {eta/60:.0f} min")

    face_mesh.close()

    print(f"\n=== Extraction complete ===")
    print(f"  Attempted: {n_ok + n_fail}")
    print(f"  OK:        {n_ok}")
    print(f"  Failed:    {n_fail}  (logged to {fail_log_path})")
    print(f"  Skipped (resume): {n_skip}")
    print(f"  Saved → {out_path}")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")

    # ── Validation ────────────────────────────────────────────────────────────
    print(f"\n=== Validation ===")
    df_out = pd.read_csv(out_path)
    dct_cols = [f"dct_{i}" for i in range(64)]
    X = df_out[dct_cols].values

    print(f"  Rows: {len(df_out)}  |  Columns: {len(df_out.columns)}")
    print(f"  Class balance: real={( df_out['class']=='real').sum()}  fake={(df_out['class']=='fake').sum()}")

    print(f"\n  First 5 DCT coefficient stats (across all rows):")
    print(f"  {'Coeff':<10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("  " + "─"*60)
    for i in range(5):
        col = X[:, i]
        print(f"  dct_{i:<6} {col.mean():>12.4f} {col.std():>12.4f} "
              f"{col.min():>12.4f} {col.max():>12.4f}")

    print(f"\n  Full matrix range: min={X.min():.4f}  max={X.max():.4f}")

    all_zero = (X == 0).all(axis=1).sum()
    print(f"  All-zero rows: {all_zero}  "
          f"({'✓ none — good' if all_zero == 0 else '⚠ check failures'})")

    near_const = (X.std(axis=1) < 1e-3).sum()
    print(f"  Near-constant rows (std < 0.001): {near_const}")

    print(f"\n  dct_0 (DC component) by class:")
    for cls in ["real", "fake"]:
        dc = df_out[df_out["class"] == cls]["dct_0"].values
        print(f"    {cls:<6}  mean={dc.mean():>10.4f}  std={dc.std():>8.4f}")


if __name__ == "__main__":
    main()
