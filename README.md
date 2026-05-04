# Acoustic Anomaly Detection — MIMII Fan Dataset

Unsupervised acoustic anomaly detection for industrial machines using a hybrid Transformer + TCN autoencoder trained on the [MIMII dataset](https://zenodo.org/record/3384388).

---

## Overview

The model is trained exclusively on **normal machine audio** and detects anomalies at inference time by flagging high reconstruction error — the same evaluation protocol used in DCASE and MIMII baselines. No anomaly labels are required during training.

The core idea: a model trained only on normal sounds should reconstruct normal audio well and reconstruct anomalous audio poorly. The reconstruction error (MSE) becomes the anomaly score.

---

## Architecture

The pipeline has three stages:

**Frontend — Log-Mel Spectrogram**
Raw 16kHz mono audio is converted to 128-band log-Mel spectrograms. Sliding-window segmentation produces fixed-length frames as model input.

**Encoder — Transformer + TCN**
- A **Transformer encoder** captures short- and mid-term spectral dependencies via multi-head self-attention, handling non-adjacent frame relationships and periodic patterns.
- A **Temporal Convolutional Network (TCN)** with dilated residual blocks extends the receptive field exponentially, capturing slowly evolving or quasi-stationary machine behaviour across hundreds of frames.
- The TCN output is bottlenecked to an 8-channel latent sequence, enforcing compression and discouraging memorisation of training data.

**Decoder — Reconstruction**
A temporal MLP expands the latent representation, followed by 1D convolutional refinement layers and a residual shortcut, reconstructing the full 128-band mel spectrogram.

**Anomaly Scoring**
Per-segment MSE is aggregated to file-level scores using 95th-percentile pooling, consistent with official DCASE evaluation protocol. ROC-AUC, PR-AUC, and F1@threshold are computed at file level.

---

## Dataset & Splits

The **MIMII fan subset** contains recordings of four fan units (ID_00, ID_02, ID_04, ID_06) across multiple SNR conditions, with both normal and anomalous labelled recordings.

| Split      | Unit IDs       | Labels used          |
|------------|----------------|----------------------|
| Training   | ID_00, ID_06   | Normal only          |
| Validation | ID_04          | Normal + anomalous   |
| Test       | ID_02          | Normal + anomalous   |

Splits are separated by **unit ID**, not by file, to enforce domain shift generalisation to unseen machines. Normalisation statistics are computed from normal training data only to prevent data leakage.

---

## Training

- **Loss:** Mean Squared Error (reconstruction)
- **Optimiser:** AdamW
- **Scheduler:** Cosine annealing
- **Epochs:** 30
- Training and validation loss converge smoothly from ~1.9 to ~0.77 with no instability, confirming stable learning without overfitting.

---

## Results

| Metric        | Validation |
|---------------|------------|
| AUC (ROC)     | 0.62       |
| AUPR          | 0.31       |
| F1 Score      | 0.50       |

Performance is consistent with reconstruction-based MIMII baselines under the cross-unit evaluation protocol. The AUC drop on the held-out test unit reflects the well-documented cross-unit domain shift problem in MIMII/DCASE literature, driven primarily by the small number of available fan units (4 total) and the non-IID nature of the dataset.

---

## Known Limitations

- **Cross-unit domain shift** is the primary failure mode. With only 4 fan units total, the model cannot learn robust domain-invariant features. This is a known limitation across DCASE/MIMII baselines.
- **BatchNorm in TCN blocks** is sensitive to domain shift — replacing with LayerNorm or GroupNorm would likely improve cross-unit stability.
- **Latent bottleneck width (8 channels)** enforces compression but limits the model's ability to represent legitimate normal variability. 32 channels caused overfitting; 4 was too narrow.
- **Training/evaluation windowing mismatch** — sliding window during training vs. full clips during evaluation introduces inconsistency in temporal statistics.
- **Dataset size** — the fan subset is small for training generalisable deep learning models. More units across varied acoustic environments would substantially improve performance.

A supervised variant (binary cross-entropy on normal vs. anomalous labels) was tested first but failed due to class imbalance and label noise, motivating the switch to the unsupervised reconstruction approach.

---

## Full Report

A complete project report covering architecture decisions, training methodology, quantitative analysis, and extended limitations discussion is available in repository.

---

## Dependencies

```
torch
torchaudio
numpy
matplotlib
jupyter
```

---

## Usage

```bash
# See report
```


---

## References

- Purohit et al. (2019) — MIMII Dataset
- Koizumi et al. (2021) — DCASE baseline ASD
- Vaswani et al. (2017) — Attention Is All You Need
- Bai, Kolter & Koltun (2018) — An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
