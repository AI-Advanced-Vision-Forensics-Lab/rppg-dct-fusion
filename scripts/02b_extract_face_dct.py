#!/usr/bin/env python3
"""
extract_dct_fullface.py — Full-face DCT features aligned with stride-60 rPPG windows.

Identical pipeline to extract_dct_features.py, except the bounding box covers
ALL 468 face landmarks instead of only lip indices. Tests whether synthesis
artifacts are detectable outside the mouth region.

Output: data/dct_features_fullface.csv  (same schema as dct_features.csv)
"""

import argparse
import csv
import random
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

STRIDE     = 60
N_FRAMES   = 160
PATCH_SIZE = 64
DCT_BLOCK  = 8
PAD        = 20


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

ZIGZAG = _zigzag_indices(DCT_BLOCK)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def source_video_path(video_id, video_root):
    if "__" in video_id:
        method, stem = video_id.split("__", 1)
        for ext in (".mp4", ".avi", ".mov", ".mkv"):
            p = video_root / "fake" / method / (stem + ext)
            if p.exists():
                return p, None
        return None, None
    m = re.search(r"_w(\d+)$", video_id)
    k = int(m.group(1)) if m else 0
    stem = re.sub(r"_w\d+$", "", video_id)
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        p = video_root / "real" / (stem + ext)
        if p.exists():
            return p, k * STRIDE
    return None, None


def load_middle_frame(video_path, start_frame):
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = total // 2 if start_frame is None else min(start_frame + N_FRAMES // 2, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, bgr = cap.read()
    cap.release()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None


def face_bbox(frame_rgb, face_landmarker):
    """Bounding box from ALL 468 face landmarks (not just lips)."""
    h, w  = frame_rgb.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = face_landmarker.detect(mp_img)
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    xs = [l.x * w for l in lm]
    ys = [l.y * h for l in lm]
    x1 = max(0,     int(min(xs)) - PAD)
    y1 = max(0,     int(min(ys)) - PAD)
    x2 = min(w - 1, int(max(xs)) + PAD)
    y2 = min(h - 1, int(max(ys)) + PAD)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def dct_vector(frame_rgb, bbox):
    x1, y1, x2, y2 = bbox
    gray  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    patch = cv2.resize(gray[y1:y2, x1:x2], (PATCH_SIZE, PATCH_SIZE),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    dct2d = dctn(patch, norm="ortho")
    block = dct2d[:DCT_BLOCK, :DCT_BLOCK]
    return np.array([block[r, c] for r, c in ZIGZAG], dtype=np.float32)


def save_spotcheck(video_id, video_root, face_landmarker, out_dir, label):
    vpath, start = source_video_path(video_id, video_root)
    if vpath is None:
        print(f"  [{label}] video not found")
        return
    frame = load_middle_frame(vpath, start)
    if frame is None:
        print(f"  [{label}] frame read failed")
        return
    bbox  = face_bbox(frame, face_landmarker)
    fbgr  = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(fbgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        crop = cv2.resize(gray[y1:y2, x1:x2], (64, 64))
        cv2.imwrite(str(out_dir / f"{label}_crop64.png"), crop)
        print(f"  [{label}] bbox=({x1},{y1})→({x2},{y2})  w={x2-x1} h={y2-y1}")
    else:
        print(f"  [{label}] NO FACE DETECTED")
    cv2.imwrite(str(out_dir / f"{label}_full.png"), fbgr)


def main():
    args      = parse_args()
    data_root = Path(args.data_root)
    video_root = data_root / "videos"
    split_df   = pd.read_csv(data_root / "dataset_split.csv")
    out_path       = data_root / "dct_features_fullface.csv"
    fail_log_path  = data_root / "dct_fullface_extraction_failures.log"
    landmark_model = data_root / "face_landmarker.task"

    done = set()
    if out_path.exists():
        existing = pd.read_csv(out_path)
        done = set(existing["video_id"].tolist())
        print(f"Resuming — {len(done)} already done")

    face_landmarker = mp_vision.FaceLandmarker.create_from_options(
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

        for _, row in split_df.iterrows():
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

            bbox = face_bbox(frame, face_landmarker)
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

    face_landmarker.close()

    print(f"\n=== Extraction complete ===")
    print(f"  Attempted: {n_ok + n_fail}")
    print(f"  OK:        {n_ok}")
    print(f"  Failed:    {n_fail}  (logged to {fail_log_path})")
    print(f"  Skipped (resume): {n_skip}")
    print(f"  Saved → {out_path}")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")

    # ── Validation ────────────────────────────────────────────────────────────
    print(f"\n=== Validation ===")
    df_out   = pd.read_csv(out_path)
    dct_cols = [f"dct_{i}" for i in range(64)]
    X        = df_out[dct_cols].values

    print(f"  Rows: {len(df_out)}  |  Columns: {len(df_out.columns)}")
    print(f"  Class balance: real={(df_out['class']=='real').sum()}  "
          f"fake={(df_out['class']=='fake').sum()}")

    all_zero   = (X == 0).all(axis=1).sum()
    near_const = (X.std(axis=1) < 1e-3).sum()
    print(f"  All-zero rows:              {all_zero}  "
          f"({'✓' if all_zero == 0 else '⚠'})")
    print(f"  Near-constant rows (std<1e-3): {near_const}")

    print(f"\n  First 5 DCT coefficient stats:")
    print(f"  {'Coeff':<10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("  " + "─"*60)
    for i in range(5):
        col = X[:, i]
        print(f"  dct_{i:<6} {col.mean():>12.4f} {col.std():>12.4f} "
              f"{col.min():>12.4f} {col.max():>12.4f}")

    print(f"\n  Full matrix range: min={X.min():.4f}  max={X.max():.4f}")

    print(f"\n  dct_0 (DC / mean brightness) by class:")
    for cls in ["real", "fake"]:
        dc = df_out[df_out["class"] == cls]["dct_0"].values
        print(f"    {cls:<6}  mean={dc.mean():>10.4f}  std={dc.std():>8.4f}")

    df_out["method"] = df_out["video_id"].apply(
        lambda x: x.split("__")[0] if "__" in x else "real"
    )
    print(f"\n  dct_1 (first AC) by method:")
    for m, grp in df_out.groupby("method")["dct_1"]:
        print(f"    {m:<18}  mean={grp.mean():>9.4f}  std={grp.std():>8.4f}")

    # ── Spot-check: 1 real + 1 SadTalker + 1 IP_LAP ──────────────────────────
    print(f"\n=== Spot-check (full-face bbox) ===")
    spotcheck_dir = data_root / "dct_spotcheck" / "fullface"
    spotcheck_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    face_landmarker2 = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(landmark_model)),
            num_faces=1,
        )
    )

    def pick(method):
        if method == "real":
            ids = df_out[df_out["method"] == "real"]["video_id"].tolist()
        else:
            ids = df_out[df_out["method"] == method]["video_id"].tolist()
        return random.choice(ids) if ids else None

    for label, method in [("real", "real"), ("sadtalker", "SadTalker"), ("ip_lap", "IP_LAP")]:
        vid = pick(method)
        if vid:
            print(f"  {label}: {vid}")
            save_spotcheck(vid, video_root, face_landmarker2, spotcheck_dir, label)

    face_landmarker2.close()
    print(f"  Crops saved → {spotcheck_dir}/")

    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()
