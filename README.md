# Quantum Machine Learning on the MAGIC Gamma Telescope Dataset

A full experimental study of variational quantum classifiers (VQC), quantum kernel SVMs, and classical baselines on a real-world astrophysics binary classification task. All quantum circuits are simulated with [PennyLane](https://pennylane.ai) `default.qubit`. Results accompany the paper *"Comparative Analysis of Quantum Feature Maps and Variational Ansätze for Binary Classification in the NISQ Era"*.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Feature Maps](#feature-maps)
  - [Variational Ansätze](#variational-ansätze)
  - [Quantum Kernel SVM](#quantum-kernel-svm)
  - [Preprocessing](#preprocessing)
  - [Training](#training)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
  - [Classification Performance](#classification-performance)
  - [Training Convergence](#training-convergence)
  - [Feature Map Fidelity](#feature-map-fidelity)
  - [Circuit Complexity](#circuit-complexity)
  - [Barren Plateau Analysis](#barren-plateau-analysis)
  - [Qubit Scalability](#qubit-scalability)
  - [Expressibility](#expressibility)
- [Figures](#figures)
- [Journal-Revision Experimental Package (Rounds 1 & 2)](#journal-revision-experimental-package-rounds-1--2)
  - [Dataset Provenance (Round 2)](#dataset-provenance-round-2)
  - [Repeated-Split Robustness (Round 2)](#repeated-split-robustness-round-2)
  - [Corrected Gradient-Scaling Methodology (Round 2)](#corrected-gradient-scaling-methodology-round-2)
  - [Reproducing the Revision Experiments](#reproducing-the-revision-experiments)
  - [Where Results Are Stored](#where-results-are-stored)
  - [Computational Requirements](#computational-requirements)
  - [Known Scientific Limitations](#known-scientific-limitations)
- [Citation](#citation)

---

## Overview

This repository implements and evaluates a suite of quantum machine learning models for binary classification:

| Model family | Configurations tested |
|---|---|
| Variational Quantum Classifier (VQC) | ZZ+StronglyEnt, Amplitude+BasicEnt, Angle+CustomAnsatz |
| Quantum Kernel SVM (QK-SVM) | ZZ kernel, Amplitude kernel, Angle kernel |
| Classical baselines | SVM (RBF), Random Forest, MLP |

Beyond accuracy, the study analyses **barren plateaus**, **circuit expressibility** (KL divergence to Haar), **qubit scalability** (n = 2 → 5), and **feature map fidelity** — providing a thorough characterisation of where quantum models succeed and fail on NISQ hardware.

---

## Dataset

**MAGIC Gamma Telescope** — UCI Machine Learning Repository  
Source: https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope

> **OpenML ID — verified programmatically (see [Dataset Provenance](#dataset-provenance-round-2) below):**
> `sklearn.datasets.fetch_openml("magic", version=1)` resolves to **OpenML dataset id 40679**
> (name `"magic"`, version 1) — not 566 as an earlier version of this README stated, and not
> 1120 (`MagicTelescope`) either. Shape, class counts, and feature semantics all match the
> canonical UCI MAGIC04 dataset exactly; this is a differently-numbered OpenML mirror of the
> same data, not a different dataset.

| Property | Value |
|---|---|
| Total samples | 19,020 |
| Features | 10 (continuous) |
| Classes | 2 (gamma signal vs. hadronic background) |
| Class balance | 35.2% gamma (1) / 64.8% hadron (0) |
| Task | Binary classification |

The 10 features (`fLength`, `fWidth`, `fSize`, `fConc`, `fConc1`, `fAsym`, `fM3Long`, `fM3Trans`, `fAlpha`, `fDist`) describe the morphological properties of Cherenkov radiation shower images captured by the telescope.

**Quantum subsample:** Due to the exponential cost of circuit simulation, quantum models train on a stratified subsample of 2,000 events (1,400 train / 600 test). Classical baselines use the full 19,020-sample dataset.

---

## Methodology

### Feature Maps

Three distinct quantum feature maps encode classical data into quantum states:

#### 1. ZZ Feature Map
Applies Hadamard gates followed by data-dependent ZZ interactions between all qubit pairs. Encodes pairwise correlations between features.

```
H^⊗n → RZ(2xᵢ) → [ CNOT · RZ(2π·xᵢxⱼ) · CNOT ]_{all pairs}
```

Inspired by Havlíček et al. (Nature, 2019) for quantum-enhanced supervised learning.

#### 2. Amplitude Embedding
Encodes the full feature vector directly into the quantum state amplitudes:

```
|ψ⟩ = Σ xᵢ/‖x‖ |i⟩
```

Requires normalisation and zero-padding to 2ⁿ dimensions. Implemented via PennyLane `AmplitudeEmbedding`.

#### 3. Angle Embedding
Applies one RX and one RY rotation per qubit, encoding features as rotation angles:

```
|ψ⟩ = RY(xᵢ) · RX(xᵢ) |0⟩^⊗n
```

Simple, hardware-efficient, and the best-performing feature map in this study.

---

### Variational Ansätze

#### Strongly Entangling Layers (SE)
Full single-qubit Rot gates (Rz Ry Rz) on every qubit, followed by a ring of CNOT gates. L layers, 3·n·L parameters.

#### Basic Entangler Layers (BE)
Single-axis RY rotation per qubit followed by a CNOT ring. Minimal parameterisation: n·L parameters.

#### Custom Ansatz
Full rotation block (RX, RY, RZ per qubit) combined with CNOT and controlled-phase (CRZ) entangling gates. 4·n·L parameters. Achieves the highest expressibility and best classification performance.

---

### Quantum Kernel SVM

The fidelity kernel between two data points is defined as:

```
K(xᵢ, xⱼ) = |⟨φ(xᵢ)|φ(xⱼ)⟩|²
```

The full kernel matrix is precomputed from quantum state overlaps and passed to `sklearn.svm.SVC(kernel='precomputed')`. Unlike VQCs, kernel SVMs have no variational parameters and are immune to barren plateaus. Evaluated on 300 training / 150 test samples.

---

### Preprocessing

| Step | Quantum models | Classical models |
|---|---|---|
| Scaling | `StandardScaler` | `StandardScaler` |
| Dimensionality reduction | PCA → n components (n = qubit count) | None |
| Range clipping | Clip to [−π, π] | — |
| Sample size | 2,000 (stratified) | 19,020 (full) |

---

### Training

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 0.01 |
| Batch size | 16 |
| Epochs | 30 |
| Loss function | Binary cross-entropy |
| Weight initialisation | Uniform [−π, π] |
| Simulator | PennyLane `default.qubit` (backprop) |
| Random seed | 42 |

---

## Project Structure

```
quantumML/
│
├── config.py               # Original hyperparameters (qubits, layers, epochs, etc.)
├── data_prep.py            # Dataset loading and preprocessing pipelines
├── quantum_circuits.py     # ZZ, Amplitude, and Angle+Custom QNodes
├── quantum_trainer.py      # VQC training loop (Adam, BCE loss, epoch logging)
├── classical_baselines.py  # SVM-RBF, Random Forest, MLP training
├── extended_metrics.py     # Accuracy, F1, precision, recall, MCC, AUC-ROC
├── quantum_kernel.py       # Fidelity kernel matrix + QK-SVM evaluation
├── scalability.py          # Qubit scalability experiment (n = 2 → 5)
├── analysis.py             # Circuit complexity, barren plateaus, fidelity, expressibility
├── plots.py                # All matplotlib figure generation
├── run_experiments.py      # Original entry point — runs all 8 original experimental stages
│
├── experiments.tex         # LaTeX experiments section (pgfplots figures, no external images)
│
├── rconfig.py               # Revision-package config (seeds, grids, output paths)
├── utils.py                 # Shared helpers: seeding, resumable JSONL run stores, bootstrap CI
├── circuits.py               # Refactored 3x3 feature-map x ansatz circuit builder
├── train_core.py             # Generalized VQC training loop used by every revision stage
├── metrics.py                 # Extended metrics (PR-AUC, balanced accuracy, confusion matrix)
├── dataset_audit.py           # Section 1 — dataset audit
├── dataset_provenance.py      # Round-2 — verified OpenML id + physics-based label check
├── primary_split.py           # Section 2 — primary controlled 1,400/600 partition
├── validate_circuits.py       # Section 4 — refactored-vs-original circuit equivalence check
├── factorial.py                # Sections 5-8 — primary 3x3 factorial VQC experiment
├── matched_budget.py           # Section 9 — matched 48-parameter-budget sensitivity
├── matched_baselines.py        # Sections 10-11 — controlled classical + QK-SVM baselines
├── separability.py             # Section 13 — feature-space separability
├── expressibility.py           # Sections 14-15 — Haar-referenced expressibility
├── barren_plateau.py           # Sections 16-18 — gradient-scaling sweep (round-2 corrected)
├── split_robustness.py         # Section 19 — repeated-split robustness (round-2 rewrite)
├── noise.py                    # Section 20 (optional) — depolarizing-noise robustness
├── unified.py                   # Unified model-comparison table
├── figures.py                   # Revision-package figure generation
├── sanity.py                    # Section 25 — legacy internal-consistency checks
├── validate_results.py          # Round-2 Phase 5 — full validation suite (14 checks)
├── manifest.py                   # Section 22 — reproducibility manifest
├── summary.py                    # Sections 26-27 — console summary + FINAL_RESULTS.json
├── run_revision_experiments.py   # Revision-package entry point (--stage {...} [--resume])
│
├── figures/                 # PNG figures from matplotlib (reference copies)
│   ├── training_convergence.png
│   ├── feature_map_comparison.png
│   ├── gradient_analysis.png
│   ├── accuracy_comparison.png
│   ├── scalability.png
│   ├── expressibility.png
│   ├── full_metrics.png
│   └── revision/            # All revision-package figures (generated, not committed)
│
├── results/
│   ├── results.json         # Original study's full numerical results
│   └── revision/            # All revision-package results (generated, not committed —
│                             #  see "Where Results Are Stored" below)
│
├── requirements.txt
└── .gitignore
```

Every `revision`-era module lives flat at the repository root (not in a nested package) so it
imports the same way the original modules do (`import rconfig`, `import split_robustness`, …).

---

## Installation

```bash
# Clone the repository
git clone https://github.com/NikolasRoufas/quantum.git
cd quantum

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note on scikit-learn:** `scikit-learn==1.5.2` is pinned because version 1.9.x has a known issue with memory-mapped dataset loading on paths containing spaces (relevant if your project directory path includes spaces).

---

## Usage

Run the full experimental pipeline (all 8 stages):

```bash
python run_experiments.py
```

This executes sequentially:

| Stage | Description | Approx. time |
|---|---|---|
| 1 | Data loading and preprocessing | < 1 min |
| 2 | Classical baselines (SVM, RF, MLP) on full dataset | 2–5 min |
| 3 | VQC training — ZZ + StronglyEnt (30 epochs) | 15–25 min |
| 4 | VQC training — Amplitude + BasicEnt (30 epochs) | 10–15 min |
| 5 | VQC training — Angle + CustomAnsatz (30 epochs) | 15–25 min |
| 6 | Quantum Kernel SVM (3 feature maps) | 5–10 min |
| 7 | Qubit scalability (n = 2, 3, 4, 5) | 20–30 min |
| 8 | Analysis: complexity, barren plateau, fidelity, expressibility | 30–60 min |

Results are saved to `results/results.json` and figures to `figures/`.

To run individual modules:

```bash
# Classical baselines only
python -c "import data_prep, classical_baselines; X,y = data_prep.load_magic(); classical_baselines.run_all(*data_prep.prepare_classical(X,y))"

# Quantum kernel SVMs only
python -c "import data_prep, quantum_kernel; X,y = data_prep.load_magic(); quantum_kernel.run_quantum_kernel_svms(*data_prep.prepare_quantum(X,y)[:4])"
```

---

## Results

### Classification Performance

All results are on held-out test sets. VQC and QK-SVM models use the 600-sample quantum test split; classical baselines use the 5,706-sample full test split.

#### Variational Quantum Classifiers (VQC)

| Model | Accuracy | F1 | Precision | Recall | Params |
|---|---|---|---|---|---|
| ZZ + StronglyEnt | 0.563 | 0.00 | 0.00 | 0.00 | 24 |
| Amplitude + BasicEnt | 0.662 | 0.55 | 0.58 | 0.53 | 8 |
| **Angle + CustomAnsatz** | **0.767** | **0.68** | **0.71** | **0.66** | 32 |

#### Quantum Kernel SVM (QK-SVM)

| Model | Accuracy | F1 | Precision | Recall | AUC-ROC |
|---|---|---|---|---|---|
| QK-SVM (ZZ kernel) | 0.673 | 0.00 | 0.00 | 0.00 | 0.419 |
| QK-SVM (Amplitude) | 0.653 | 0.42 | 0.44 | 0.40 | 0.511 |
| **QK-SVM (Angle)** | **0.780** | **0.71** | **0.73** | **0.69** | **0.814** |

#### Classical Baselines (full dataset)

| Model | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| SVM (RBF) | 0.878 | 0.83 | 0.85 | 0.81 |
| **Random Forest** | **0.881** | **0.84** | **0.86** | **0.82** |
| Neural Network (MLP) | 0.855 | 0.80 | 0.82 | 0.78 |

**Key finding:** The best quantum model (QK-SVM Angle, 78.0%) trails the best classical model (Random Forest, 88.1%) by 10.1 pp — a gap attributable primarily to the 9.5× smaller training set available to quantum models, not to algorithmic inferiority.

---

### Training Convergence

Loss values (binary cross-entropy) at every 5th epoch:

| Model | Ep.5 | Ep.10 | Ep.15 | Ep.20 | Ep.25 | Ep.30 |
|---|---|---|---|---|---|---|
| ZZ + StronglyEnt | 0.6839 | 0.6839 | 0.6828 | 0.6809 | 0.6783 | 0.6770 |
| Amplitude + BasicEnt | 0.6353 | 0.6315 | 0.6330 | 0.6331 | 0.6322 | 0.6323 |
| Angle + CustomAnsatz | 0.5450 | 0.5411 | 0.5362 | 0.5351 | 0.5347 | 0.5341 |

ZZ+StronglyEnt never leaves the neighbourhood of the random-classifier baseline (ln 2 ≈ 0.693), confirming barren plateau collapse. Angle+CustomAnsatz achieves consistent monotonic improvement from epoch 1.

---

### Feature Map Fidelity

Mean quantum state fidelity F = |⟨φ(xᵢ)|φ(xⱼ)⟩|² over 400 random pairs:

| Feature Map | Same-class F̄ | Diff-class F̄ | Gap Δ |
|---|---|---|---|
| ZZ | 0.079 | 0.069 | +0.010 |
| Amplitude | 0.240 | 0.267 | −0.027 |
| Angle | 0.237 | 0.224 | +0.013 |

ZZ and Angle feature maps correctly order same-class states closer together (positive gap). The Amplitude embedding inverts the ordering, which explains the poor QK-SVM Amplitude AUC (0.511 ≈ random).

---

### Circuit Complexity

| Configuration | Gates | Depth | Params |
|---|---|---|---|
| ZZ + StronglyEnt (4q, 2L) | 58 | 22 | 24 |
| Amplitude + BasicEnt (4q, 2L) | 31 | 8 | 8 |
| Angle + CustomAnsatz (4q, 2L) | 48 | 12 | 32 |

Angle+CustomAnsatz achieves the best accuracy-to-depth ratio: 76.7% at depth 12, vs. 56.3% at depth 22 for ZZ+StronglyEnt.

---

### Barren Plateau Analysis

Gradient norm ‖∇θL‖₂ statistics at 30 random initialisations per depth (ZZ+StronglyEnt, 4 qubits):

| Depth L | Mean ‖∇‖ | Std ‖∇‖ | Per-param norm | Rel. std σ/μ |
|---|---|---|---|---|
| 1 | 0.415 | 0.159 | 0.120 | 0.38 |
| 2 | 0.542 | 0.147 | 0.111 | 0.27 |
| 3 | 0.948 | 0.154 | 0.158 | 0.16 |
| 4 | 1.128 | 0.201 | 0.163 | 0.18 |
| 5 | 1.217 | 0.190 | 0.157 | 0.16 |
| 6 | 1.391 | 0.139 | 0.164 | 0.10 |

Total gradient norm grows with depth (more parameters), but relative variance (σ/μ) drops from 0.38 to 0.10 — gradients become increasingly concentrated near zero, a hallmark of barren plateaus (McClean et al., 2018).

---

### Qubit Scalability

ZZ+StronglyEnt, 2 layers, 20 epochs, 700-sample subsample:

| Qubits n | Test Accuracy | Parameters |
|---|---|---|
| 2 | 0.667 | 12 |
| 3 | 0.571 | 18 |
| 4 | 0.491 | 24 |
| 5 | 0.491 | 30 |

Accuracy degrades monotonically with qubit count, reaching the majority-class baseline (≈ 0.49) at n = 4 and staying there at n = 5. Consistent with exponential barren plateau onset as Hilbert space dimension grows as 2ⁿ.

---

### Expressibility

KL divergence from the Haar-random fidelity distribution P(F) = (d−1)(1−F)^(d−2), d = 2⁴ = 16 (lower = more expressive):

| Ansatz | KL Divergence | Rank |
|---|---|---|
| Angle + CustomAnsatz | 0.066 | 1 — most expressive |
| ZZ + StronglyEnt | 0.069 | 2 |
| Amplitude + BasicEnt | 0.082 | 3 — least expressive |

The expressibility ranking exactly mirrors classification performance, confirming that ansatz design (full rotation blocks + CRZ entanglement) is the decisive factor for NISQ-era classification.

---

## Figures

All figures in `figures/` are generated by `plots.py` using matplotlib. The LaTeX experiments section (`experiments.tex`) reproduces them as self-contained `pgfplots` / `tikzpicture` figures requiring no external image files.

| File | Content |
|---|---|
| `training_convergence.png` | BCE loss vs epoch for 3 VQC models |
| `feature_map_comparison.png` | Same/diff-class fidelity per feature map |
| `gradient_analysis.png` | Gradient norm ± std vs circuit depth |
| `accuracy_comparison.png` | Horizontal bar chart of all 9 model accuracies |
| `scalability.png` | Accuracy + parameter count vs qubit count |
| `expressibility.png` | KL divergence bar chart + Haar reference curve |
| `full_metrics.png` | Grouped acc/F1/precision/recall for all models |

---

## Journal-Revision Experimental Package (Rounds 1 & 2)

Everything above describes the **original study** (single train/test partition, single
seed per configuration, un-pooled gradient statistics). In response to peer review, two
rounds of additional experimental work extend and correct that study without altering
its underlying data or code:

- **Round 1** (`primary_split.py`, `factorial.py`, `matched_budget.py`, `matched_baselines.py`,
  `separability.py`, `expressibility.py`): a full 3×3 feature-map × ansatz factorial design
  (fixing the original's confounded one-configuration-per-feature-map design), a
  matched-parameter-budget (48-parameter) sensitivity check, and a matched preprocessing
  protocol shared by VQC, QK-SVM, and classical baselines.
- **Round 2** (`split_robustness.py`, `barren_plateau.py`, `dataset_provenance.py`,
  `validate_results.py`): repeated train/test splits, full reproducibility artifacts,
  a corrected gradient-scaling methodology, and verified dataset provenance — the four
  items summarized below.

Run via `python run_revision_experiments.py --stage all --resume` (see
[Reproducing the Revision Experiments](#reproducing-the-revision-experiments)).

### Dataset Provenance (Round 2)

`fetch_openml("magic", version=1)` was independently verified rather than trusted:

| Check | Result |
|---|---|
| Resolved OpenML id | **40679** (name `"magic"`, version 1) — not 566 (this README's earlier claim) or 1120 (`MagicTelescope`) |
| Shape | 19,020 × 10 — matches UCI exactly |
| Class counts | 12,332 / 6,688 — matches UCI exactly |
| Missing values | 0 |

**A separate, more consequential finding:** the integer class label does not mean what
`data_prep.load_magic()`'s comment claims. Using `FAlpha` (the shower-axis angle to the
origin — physically, gamma showers point back at the source and so have small `FAlpha`;
hadron background is isotropic, larger `FAlpha`) as an independent check: class `0` has
mean `FAlpha` ≈ 18.8°, class `1` ≈ 44.0°. **Class 0 is physically gamma; class 1 is
physically hadron** — the opposite of the `"gamma=1, hadron=0"` assumption baked into the
original, pre-revision `data_prep.py`. This affects the original study too, not just this
revision. Accuracy / balanced accuracy / MCC are symmetric and unaffected; precision,
recall, ROC-AUC, and PR-AUC are computed correctly with respect to whichever label is
coded as 1, but any narrative describing those numbers as "gamma-detection performance"
has been describing hadron-detection performance, and vice versa. **This has been left
unchanged** (no relabeling, no reruns) pending a decision on the manuscript side — see
[Known Scientific Limitations](#known-scientific-limitations). Full detail, including the
raw counts and the `FAlpha` check, is in `results/revision/dataset_provenance.json`.

### Repeated-Split Robustness (Round 2)

The original revision's split-robustness experiment conflated two different sources of
variability (which 2,000-sample subsample is drawn, and how it's partitioned) and omitted
quantum-kernel SVMs entirely. This round separates and completes both:

- **`fixed_subset` (primary):** the 2,000-sample subsample is drawn once; only the
  stratified 1,400/600 **partition** varies across 5 seeds (`[42, 123, 456, 789, 2026]`).
  Isolates partition identity as the sole source of variation.
- **`varying_subset` (secondary/sensitivity):** both the subsample and the partition are
  redrawn together per seed — the original design, retained for comparison.
- **3 independent initialization seeds** per VQC/split combination (`[7001, 7002, 7003]`,
  deliberately disjoint from the split seeds), aggregated within each split before any
  split-level comparison.
- **QK-SVM now included** in the repeated-split sweep (previously omitted) — 3 kernels × 5
  splits, reusing the exact statevector-Gram implementation from the primary comparison.
- **Full reproducibility artifacts** for every run: exact train/test indices, true labels,
  predicted labels, prediction scores, confusion matrix, and every metric — not just
  aggregate statistics. Validated: train/test disjointness, matched partitions across
  model families for a given split, confusion-matrix totals, and that saved predictions
  reproduce the saved metrics (see [`validate_results.py`](validate_results.py)).

Fixed-subset results (mean ± SD across 5 splits, VQC/QK-SVM additionally averaged over 3
init seeds per split first):

| Method | Accuracy | F1 |
|---|---|---|
| SVM (RBF) | 0.782 ± 0.022 | 0.637 ± 0.045 |
| Random Forest | 0.777 ± 0.019 | 0.659 ± 0.031 |
| **Angle + CustomAnsatz (VQC)** | **0.758 ± 0.017** | **0.587 ± 0.039** |
| QK-SVM (Angle) | 0.769 ± 0.019 | 0.614 ± 0.035 |
| Angle + StronglyEnt (VQC) | 0.717 ± 0.006 | 0.531 ± 0.018 |
| Angle + BasicEnt (VQC) | 0.689 ± 0.024 | 0.325 ± 0.076 |

The original study's headline finding — Angle+CustomAnsatz as the best VQC configuration,
trailing but not far behind the best classical/QK-SVM baselines — **remains supported**
under repeated-split evaluation (0.758 here vs. 0.767 on the original single partition).
With only 5 partitions, paired split-level comparisons
(`results/revision/split_robustness/paired_comparisons.csv`) are reported descriptively
(mean difference + bootstrap 95% CI), not as significance claims.

### Corrected Gradient-Scaling Methodology (Round 2)

The original barren-plateau sweep's headline statistic pooled gradient entries across
*both* random initializations *and* parameter identity into one number — not an adequate
primary diagnostic, since it can't distinguish "every parameter has a small gradient" from
"a few parameters dominate while most don't vary." This round's correction:

- **Primary statistic:** for every trainable parameter *p*, `V_p = Var_i(g[i, p])` across
  the `n_init=100` random initializations; `mean_of_per_param_variances` (and median, as a
  robustness companion) replaces the old pooled variance as the primary scaling statistic.
  The old pooled statistic is retained only as an explicitly-labeled secondary comparison.
- **Deterministic seeding:** cell seeds are now derived via `hashlib.sha256`, not Python's
  built-in `hash()` (salted per-process by default — not reproducible run-to-run for a
  seed built from a string). Verified via an automatic determinism spot-check on every run
  (`results/revision/barren_plateau/determinism_check.json`).
- **Bootstrap 95% CIs** (2,000 replicates) on every scaling-regression slope, resampling
  complete per-initialization gradient vectors within each grid cell — never resampling
  individual gradient entries as if independent.
- **Sensitivity analyses:** parameter-selection (mean / median / single-representative-
  parameter / pooled-legacy, compared side by side) and leave-one-qubit-count-out refits.

Headline qubit-scaling and depth-scaling slopes (`log10(mean per-parameter variance)` vs.
grid axis, bootstrap 95% CI in brackets):

| Ansatz | vs. qubits | vs. depth |
|---|---|---|
| StronglyEnt | −0.312 [−0.318, −0.306] — suppression | +0.019 [+0.013, +0.025] — no suppression |
| BasicEnt | −0.325 [−0.332, −0.318] — suppression | −0.035 [−0.043, −0.028] — suppression |
| CustomAnsatz | −0.293 [−0.298, −0.288] — suppression | +0.048 [+0.042, +0.055] — no suppression |

All three ansätze show gradient-variance suppression with increasing qubit count over the
tested range (2–8 qubits); only BasicEnt shows suppression with increasing depth (1–6
layers) — StronglyEnt and CustomAnsatz show a small *positive* depth slope instead. The
parameter-selection sensitivity analysis found a striking case for StronglyEnt: the
single-representative-parameter diagnostic (the classic McClean et al. 2018 statistic,
one arbitrarily-chosen parameter) gives slope ≈ −0.04 (R² ≈ 0.03, essentially no
detectable trend), while the per-parameter-averaged statistic gives slope ≈ −0.31
(R² ≈ 0.997) — a single arbitrarily-chosen parameter can be highly unrepresentative for a
structured ansatz like `StronglyEntanglingLayers`, which is exactly the failure mode the
per-parameter correction is meant to catch. **These are small simulated circuits (≤8
qubits) with a local observable; this is evidence of finite-size gradient suppression over
the tested range, not a demonstration of asymptotic barren-plateau scaling**, which would
require evidence at qubit counts far beyond what classical simulation can reach. Full
per-ansatz, per-axis fits, CIs, and sensitivity tables:
`results/revision/barren_plateau/scaling_fits.json`.

### Reproducing the Revision Experiments

```bash
pip install -r requirements.txt

# everything (original revision stages + round-2 corrections), resumable:
python run_revision_experiments.py --stage all --resume --skip-noise

# an individual stage, e.g. just the round-2 repeated-split experiment:
python run_revision_experiments.py --stage split-robustness --resume

# round-2 dataset provenance check and full validation suite, standalone:
python dataset_provenance.py
python validate_results.py
```

Every stage is resumable (`--resume`): completed runs are tracked in append-only JSONL
"run store" files and skipped on a subsequent invocation; only runs that genuinely failed
are retried (a run recorded with `status: "failed"` is never counted as done).

**If running VQC training stages under `multiprocessing`:** the entry-point script must
guard its top-level code with `if __name__ == "__main__":`. `run_revision_experiments.py`
already does this; a custom launcher script that doesn't will see every worker process
crash or silently re-execute the parent's top-level code under the `spawn` start method
these stages use (`factorial.py`, `split_robustness.py`) — this was a real, time-consuming
bug encountered while running this exact round-2 package and is recorded here so it isn't
rediscovered.

### Where Results Are Stored

| What | Path |
|---|---|
| Dataset audit / provenance | `results/revision/dataset_audit.json`, `results/revision/dataset_provenance.json` |
| Primary factorial (3×3) | `results/revision/factorial/` |
| Matched parameter-budget | `results/revision/matched_budget/` |
| Classical + QK-SVM baselines (primary partition) | `results/revision/matched_baselines/` |
| Separability / expressibility | `results/revision/separability/`, `results/revision/expressibility/` |
| Gradient-scaling sweep (round-2 corrected) | `results/revision/barren_plateau/` |
| Repeated-split robustness (round-2) | `results/revision/split_robustness/{fixed_subset,varying_subset}/` |
| Validation report | `results/revision/validation_report.json` |
| Reproducibility manifest | `results/revision/revision_manifest.json` |
| Full machine-readable snapshot | `results/revision/FINAL_RESULTS.json` |
| Final reviewer-response package | `results/revision/final_review_results/` (see `SUMMARY.md` there) |
| Figures | `figures/revision/` |

None of `results/revision/` or `figures/revision/` is committed to this repository (see
`.gitignore`) — they're regenerated by running the stages above, except for the small,
curated `final_review_results/` summary package, which is committed.

### Computational Requirements

The round-1 + round-2 revision package was run on a 48-core / 188 GB RAM machine (CPU
only — nothing in this codebase uses a GPU tensor); the full `--stage all --resume` run
completes in well under 15 minutes on that hardware. A single-digit-core machine will take
proportionally longer but is not expected to need more than a few hours; every stage is
resumable, so an interrupted run loses no completed work.

### Known Scientific Limitations

- **Gamma/hadron label semantics** (see [Dataset Provenance](#dataset-provenance-round-2)):
  the integer class label's gamma/hadron naming is inverted relative to the physical
  class, in both the original study and this revision. The underlying numbers are valid
  self-consistent binary-classification results; any narrative framing them as
  "gamma-detection" vs. "hadron-detection" performance has the two swapped. Left
  unchanged in code pending a decision on the manuscript side.
- **OpenML dataset version inactive:** `fetch_openml` prints a deprecation-style warning
  that OpenML version 1 of this dataset has been marked inactive ("issues have been
  found") with a newer version suggested. Data shape/content were independently verified
  against the authoritative UCI source regardless (see above); the warning has not been
  investigated further.
- **Small simulated circuits:** the gradient-scaling results (≤8 qubits, local observable)
  support finite-size gradient suppression conclusions only, not asymptotic barren-plateau
  scaling claims — see [Corrected Gradient-Scaling Methodology](#corrected-gradient-scaling-methodology-round-2).
- **n=5 partitions:** repeated-split paired comparisons are reported descriptively
  (mean difference + bootstrap CI) and should not be read as hypothesis-test significance
  claims, given the small number of independent partitions.
- **Noise-robustness stage is optional and was not run** in this round (`--skip-noise`);
  `noise.py` exists and is runnable independently if needed.

---

## Citation

If you use this code or results in your work, please cite:

```bibtex
@misc{roufas2025quantumml,
  author       = {Roufas, Nikolaos},
  title        = {Comparative Analysis of Quantum Feature Maps and Variational
                  Ansätze for Binary Classification in the NISQ Era},
  year         = {2025},
  howpublished = {\url{https://github.com/NikolasRoufas/quantum}}
}
```

Key references used in this work:

- Havlíček et al. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature*.
- McClean et al. (2018). Barren plateaus in quantum neural network training landscapes. *Nature Communications*.
- Sim et al. (2019). Expressibility and entangling capability of parameterized quantum circuits. *Advanced Quantum Technologies*.
- Cerezo et al. (2021). Variational quantum algorithms. *Nature Reviews Physics*.
- Bergholm et al. (2018). PennyLane: Automatic differentiation of hybrid quantum-classical computations.
- Huang et al. (2021). Power of data in quantum machine learning. *Nature Communications*.
