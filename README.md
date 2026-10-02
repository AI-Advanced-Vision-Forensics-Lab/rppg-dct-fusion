# Complementary rPPG-Derived and Lip-Region Frequency Cues for Talking-Face Deepfake Detection

**Othmane Harraq, Tamer Aldwairi** — Temple University

Code and result files for the paper. It studies two lightweight, visual-only cues on the seven talking-face (TF) methods of Celeb-DF++, using an identity-disjoint split:

- rPPG-derived waveforms extracted by RhythmFormer
- lip-region DCT coefficients

It compares a Gated, a Concat and a Cross-attention fusion of the two cues, both in-domain and under leave-one-generator-out (LOGO) evaluation.

| Combined model (all 7 methods) | AUC |
|---|---|
| rPPG only (1D ResNet) | 0.824 |
| Lip DCT only (MLP) | 0.827 |
| Concat fusion | **0.891** |
| LOGO mean: rPPG / DCT / Concat | 0.752 / 0.729 / 0.798 |

The rPPG extraction pipeline is the one from the companion repository [rPPG-TalkingFace-Detector](https://github.com/AI-Advanced-Vision-Forensics-Lab/rPPG-TalkingFace-Detector) (PulseGuard), applied here to a 17,497-video subset of the fakes.

---

## Repository layout

```
scripts/            extraction, split, one script per table / figure
scripts/reruns/     the runs behind the rPPG, rPPG-linear and Concat-ResNet columns of Table 1 and all of Table 2
results/6_29_2026/  lip-DCT, Gated, Concat, Cross-attention runs (Table 1)
results/6_30_2026/  full-face DCT runs (Table 3)
results/reruns/     outputs of scripts/reruns/ (config + results JSON per run)
splits/             identity_split.csv (59 identities) and corpus_17497.csv
figures/            fig3_per_method_auc.{pdf,png}
```

## Setup

Python 3.10 and an NVIDIA GPU are required.

```bash
pip install -r requirements.txt          # see the note in requirements.txt about the CUDA build of torch
git clone https://github.com/zizheng-guo/RhythmFormer.git
git -C RhythmFormer checkout 7c9cea2
```

- **RhythmFormer checkpoint.** Place `UBFC_cross_RhythmFormer.pth` from the RhythmFormer authors in `RhythmFormer/PreTrainedModels/`. We do not redistribute it.
- **MediaPipe models.** Put `face_landmarker.task` (MediaPipe FaceLandmarker) in `--data-root`, for the DCT features. The BlazeFace short-range model is downloaded automatically by the waveform extractor.
- **Arguments.** Every script takes `--data-root` (default `./data`). The `scripts/reruns/` scripts also take `--out-dir` (default `<data-root>/results/reruns`). The `table1_*.py` scripts write to `<data-root>/results/<M_DD_YYYY>/`.

## Dataset

We use the TalkingFace subset of **Celeb-DF++** (Li et al., 2025). Request access from the authors at <https://github.com/OUC-VAS/Celeb-DF-PP>. The dataset is released for non-commercial academic research under the authors' terms of use.

**This repository contains no videos, frames, face crops, extracted waveforms, DCT features or model weights.**

The corpus is the 590 real videos plus **2,500 randomly sampled videos per TF method**. Three EchoMimic videos yielded no DCT features, giving 17,497 fakes. `splits/corpus_17497.csv` lists every sample used (video id, class, method, identity, split). Copy or symlink the listed videos into this layout:

```
data/
├── videos/real/<stem>.mp4                       # Celeb-real
├── videos/fake/<method>/<stem>.mp4              # the 2,500 (2,497) sampled videos per method
├── face_landmarker.task
└── waveforms/                                   # written by 00_extract_waveforms.py
    ├── real/<stem>_w<k>.npy
    └── fake/<method>__<stem>.npy
```

The `scripts/reruns/` scripts read the same waveforms with the companion repository's file layout. Create symlinks once after extraction:

```bash
cd data/waveforms
mkdir -p CelebDF/TalkingFace && ln -s ../real CelebDF/Celeb-real
for m in AniTalker EchoMimic EDTalk FLOAT IP_LAP Real3DPortrait SadTalker; do
  mkdir -p CelebDF/TalkingFace/$m
  for f in fake/${m}__*.npy; do ln -s ../../../$f CelebDF/TalkingFace/$m/${f#fake/${m}__}; done
done
```

In our runs the two layouts held byte-identical files.

## Identity split

`splits/identity_split.csv` assigns each of the 59 identities to `train` (41), `val` (9) or `test` (9). Results are reported on the 18 validation + test identities combined: 696 real windows and 5,553 fake videos. `01_build_split.py` writes `data/dataset_split.csv`, which every training script reads.

## Reproducing the paper

All runs use 5 seeds {42, 7, 123, 999, 2024}, AdamW (lr 1e-3, wd 5e-4), batch 64 and 30 epochs. The shipped JSON is the file each number was read from.

| Paper item | Command | Shipped JSON |
|---|---|---|
| rPPG waveforms (Sec. 3.1) | `python scripts/00_extract_waveforms.py --repo-dir RhythmFormer` | — |
| Split | `python scripts/01_build_split.py` | — |
| Lip DCT features (Sec. 3.2) | `python scripts/02_extract_lip_dct.py` | — |
| Table 1, rPPG (1D ResNet) + Combined | `python scripts/reruns/run_r1_r2.py` | `results/reruns/r1_r2_rppg_resnet/r1_results.json`, `r2_results.json` |
| Table 1, rPPG-linear + Combined | `python scripts/reruns/run_r4_linear_rppg.py` | `results/reruns/r4_linear_rppg/r4_iso_results.json`, `r4_comb_results.json` |
| Table 1, DCT + Combined | `python scripts/table1_dct_only.py` | `results/6_29_2026/dct_only_experiments.json` |
| Table 1, Gated + Combined, gate weights α | `python scripts/table1_gated_fusion.py` | `results/6_29_2026/gated_fusion_experiments.json` (`mean_alpha`) |
| Table 1, Concat (Con.) + Combined | `python scripts/table1_concat_fusion.py` | `results/6_29_2026/fusion_experiments.json` |
| Table 1, Cross-attention (CA) + Combined | `python scripts/table1_crossattn_fusion.py` | `results/6_29_2026/crossattn_experiments.json` |
| Combined Concat / CA cross-check | `python scripts/table1_combined_concat_crossattn.py` | `results/6_29_2026/full_training_comparison.json` |
| Sec. 4.1, Concat with ResNet encoder | `python scripts/reruns/run_r5_resnet_fusion.py` | `results/reruns/r5_resnet_fusion/r5_iso_results.json` |
| Table 2, LOGO (rPPG, DCT, Concat) | `python scripts/reruns/run_r3_lomo.py` | `results/reruns/r3_lomo/results.json` |
| Table 3, full-face DCT features | `python scripts/02b_extract_face_dct.py` | — |
| Table 3, DCT-only face | `python scripts/table1_dct_only.py --dct-file data/dct_features_fullface.csv --out-file dct_fullface_only.json` | `results/6_30_2026/dct_fullface_only.json` |
| Table 3, Concat face | `python scripts/table1_concat_fusion.py --dct-file data/dct_features_fullface.csv --out-file fusion_fullface_experiments.json` | `results/6_30_2026/fusion_fullface_experiments.json` |
| Fig. 2 | `python scripts/fig2_examples.py` | — (see notes) |
| Fig. 3 | `python scripts/fig3_per_method_auc.py` | reads the four JSONs listed in its header |

In the result files, `Overall` (or the `r2` / `comb` files) is the **Combined** row: one model trained on all seven methods and evaluated on the pooled eval set. Per-method entries are models trained on one method's fakes plus all training reals.

### Notes

- **Oracle average (Sec. 4.2, 0.818).** This is the unweighted mean over held-out methods of max(rPPG, DCT) in `results/reruns/r3_lomo/results.json`, which gives 0.8176. It is computed from that file and not stored separately.
- **Fig. 3** is drawn only from shipped results. It reads `r1_results.json` (rPPG), `dct_only_experiments.json` (DCT), `fusion_experiments.json` (Concat) and `r3_lomo/results.json` (LOGO), and needs no data or GPU.
- **Fig. 2** needs pre-extracted face frames under `data/frames/celeb/{Celeb-real,TalkingFace/<method>}/`: 10 uniformly sampled, 299×299 BlazeFace crops per video. The frame-extraction step is not part of this repository. The DCT maps are recomputed from those frames, and the clips used are listed in the script header.
- **Fig. 1** (pipeline diagram) is not included because it contains a frame from the dataset.
- **Path edits only.** Relative to the code that produced the results, the scripts were changed only to make paths configurable and to rename files. `fig3_per_method_auc.py` is new and only plots. The training and extraction scripts were not re-run end to end after these edits. They were checked statically: they compile, `--help` works, and referenced paths exist.

## Citation

```bibtex
@misc{harraq2026complementary,
  title  = {Complementary rPPG-Derived and Lip-Region Frequency Cues for Talking-Face Deepfake Detection},
  author = {Harraq, Othmane and Aldwairi, Tamer},
  year   = {2026},
  note   = {Under review}
}

@article{harraq2026pulseguard,
  title   = {PulseGuard: rPPG-Derived Temporal Signals as a Forensic Modality for Talking-Face Deepfake Detection},
  author  = {Harraq, Othmane and Aldwairi, Tamer},
  journal = {arXiv preprint arXiv:2607.21776},
  year    = {2026}
}
```

## Acknowledgements

[RhythmFormer](https://github.com/zizheng-guo/RhythmFormer), [MediaPipe](https://github.com/google-ai-edge/mediapipe) and [Celeb-DF++](https://github.com/OUC-VAS/Celeb-DF-PP).
