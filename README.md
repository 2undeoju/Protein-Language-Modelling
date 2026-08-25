# xLSTM | Transformers(ESM2) | Protein Language Modelling Under Resource Constraints 

This repository contains the implementation of the xLSTM-based protein language model developed for the MSc thesis:

**"xLSTM for Bi-directional Protein Language Modelling Under Resource Constraints"**
Johannes Kepler University Linz, 2026

## Overview

This is my Master's thesis at Johannes Kepler University, Linz: **"From Protein Language
Modelling to mLSTM via Transformer"** — a hands-on comparison of two architectures for
learning representations of protein sequences, under a strict, fixed compute budget.

In plain terms: protein language models (pLMs) read amino acid sequences the way a
language model reads text, and learn to predict masked-out parts of the sequence. Once
trained, the model's internal representations turn out to be useful for predicting real
protein properties — stability, fitness, function — without ever being told the biology
explicitly. Today's best pLMs (like ESM2) are Transformers, and Transformers get slower
and more memory-hungry the longer the sequence gets. This thesis asks a practical
question: **can a newer, LSTM-style architecture called xLSTM match a Transformer's
protein-modelling quality, while training under identical, tightly-constrained
resources?**

**Short answer:** not quite, but closer than you'd expect. The Transformer baseline
(ESM2) came out ahead on both raw language-modelling quality and downstream prediction
tasks, but the gap was small on tasks that depend on short, local patterns in a
sequence, and only became large on a task that needs the model to relate distant parts
of a long sequence to each other — which is exactly where an LSTM-style, recurrent
model is expected to struggle relative to a Transformer's attention mechanism. Full
numbers are in [Results](#results) below.

## For non-technical readers — what was actually tested

Think of it like this: two students are given the exact same 24 hours, the same
textbook (the same protein dataset), and the same amount of "brain size" (matched
parameter count), and are told to learn to fill in blanked-out words in protein
sequences. Student A studies using **attention** — for every word, it looks back and
forth across the whole page at once. Student B studies using a **memory that updates
sequentially**, word by word, carrying forward a compressed summary as it reads.
Student A is the established technique (Transformer/ESM2); Student B is the newer,
untested-for-proteins technique (xLSTM). Both students then take the same exam
(validation perplexity — literally, "how surprised is the model by the correct next
piece of the sequence"), and then both are tested on three real-world tasks that
biologists actually care about:

- **GB1** — predicting how a mutated protein's binding strength changes
- **AAV** — predicting how well a mutated viral capsid protein still functions
- **Meltome** — predicting a protein's melting temperature (thermal stability)

Student A (Transformer) won the exam and all three real-world tests, but Student B
(xLSTM) was close on two of them and only fell noticeably behind on one (AAV) — the one
that most depends on relating far-apart positions in a long sequence to each other,
which is the Transformer's specific strength.

## Why this matters

- Protein language models are usually large Transformers, which scale quadratically
  with sequence length — expensive to train and to run at scale.
- xLSTM (Beck et al., 2024) is a recent recurrent architecture that processes sequences
  in linear time and constant memory per step, using a matrix-valued memory cell and
  input-dependent gating instead of attention.
- If an xLSTM-style model can match Transformer-level pLM quality, it opens the door to
  cheaper training and cheaper inference for protein modelling — relevant for anyone
  without a large GPU cluster.
- Rather than training both architectures to convergence (which would take far more
  compute than a Master's thesis affords), this project uses the **"cramming"**
  methodology (Geiping & Goldstein, 2023; adapted for proteins by Frey et al., 2024):
  both models get an identical, fixed, realistic training budget, and the question
  becomes "what's the best each architecture can do in that budget?" — a much more
  relevant question for practitioners with limited compute than "what's the absolute
  ceiling given infinite compute?"

## Method

**Dataset.** UniRef50 protein sequences, streamed on-the-fly from FASTA (no full
in-memory dataset), tokenised per-residue, batched by a token budget rather than a fixed
batch size so that GPU memory is used consistently regardless of sequence length
(`DataLoader.py`, `data_setup.py`). Training uses the standard masked language modelling
(MLM) objective with 25% masking and the classic 80/10/10 mask/random/keep split.

**Architectures compared, matched to be as fair as possible:**

| | ESM2 baseline (Transformer) | mLSTM (this work) |
|---|---|---|
| Type | Multi-head self-attention, Pre-LayerNorm, GELU | Matrix-memory LSTM (mLSTM) with input-dependent exponential gating |
| Base config | `facebook/esm2_t6_8M_UR50D` architecture (random init — **not** pretrained ESM2 weights) | Custom implementation (`mlstm_modelNew.py`), chunkwise-parallel recurrence for training-time speed |
| Directionality | Bidirectional (native to attention) | Bidirectional adaptation: sequence is run forward and reversed, then the two representations are concatenated |
| Default depth (baseline comparison) | 6 layers | 6 layers, hidden size 320, embed_dim 128, 8 heads, chunk size 64 |
| Params | ~8M (matched to the ESM2-8M reference config) | matched to be comparable at the same depth |

Both models are trained **completely from scratch with random initialisation** — the
ESM2 architecture is used only as a config/tokenizer template, not as a pretrained
checkpoint, so this is a fair from-scratch comparison rather than "pretrained
Transformer vs. untrained LSTM."

**Training budget (identical for both):** 10,000 optimiser updates, batch size 12 with
24 gradient-accumulation steps, max sequence length 512, AdamW (β = 0.99/0.98,
eps = 1e-12), gradient clipping at norm 0.5, linear warmup + decay learning-rate
schedule, mixed precision (FP16 forward/backward, FP32 parameter updates). Four
learning-rate/warmup combinations were swept for each architecture
(`lr ∈ {4e-4, 1e-3}` × `warmup ∈ {1000, 2000}` steps) and the best run per architecture
is reported below; exact configs are in `config.py` (`BASELINE_SWEEPS`).

**Depth vs. width ablation (mLSTM only).** Because a recurrent model's effective
capacity is shaped differently by depth than a Transformer's, a second experiment holds
the mLSTM parameter budget roughly fixed while varying depth: 1 layer (wide,
hidden=672), 6 layers (balanced, hidden=320), and 12 layers (deep, hidden=240) — each
swept across the same 4 learning-rate/warmup combinations (`config.py`,
`DEPTH_VARIANTS`). This isolates whether depth or width matters more for mLSTM under a
fixed compute budget.

**Evaluation.**

1. **Upstream**: validation perplexity (`PPL = exp(validation loss)`), computed after
   training on a held-out validation split (`evaluate.py`).
2. **Downstream**: three tasks from the FLIP benchmark — GB1 (protein-protein binding
   fitness), AAV (viral capsid function), Meltome (thermal stability). Representations
   are extracted **frozen** (no fine-tuning) from each trained model, and a simple
   linear regression head is trained on top to predict the target property, scored with
   Spearman rank correlation against the true labels (`ds_eval_flip.py`,
   `ds_data_flip.py`, `ds_config.py`). This tests whether the *learned representations*
   are useful for real biology, not just whether the model is good at the training
   objective itself.

## Results

All numbers below are pulled directly from this repository's
[`thesis_tables/`](https://github.com/2undeoju/Protein-Language-Modelling/tree/main/thesis_tables)
CSVs, which are generated by `generate_thesis_results.py` / `generate_checkpoint_table.py`
from the actual training/eval runs — not rounded or re-derived by hand.

### Upstream: validation perplexity (lower is better)

**Head-to-head baseline, matched 6-layer depth, best learning-rate/warmup setting per model:**

| Model | Step 1000 | Step 5000 | Step 10000 | Final PPL |
|---|---:|---:|---:|---:|
| ESM2 (Transformer), lr 1e-3, warmup 2000 | 13.70 | 12.96 | 12.74 | **12.76** |
| mLSTM, lr 1e-3, warmup 2000 | 13.69 | 13.44 | 13.30 | **13.27** |

At matched depth and an identical training budget, the Transformer baseline reaches
about **4% lower validation perplexity** than the mLSTM. The result is consistent, not
a fluke of one run: across all 4 learning-rate/warmup settings tried, ESM2's final PPL
ranged 12.76–12.87 and mLSTM's ranged 13.27–13.31 — the gap holds up across the whole
sweep. (A second, independently-run ESM2 confirmation sweep, labelled
`esm2_baseline_*` in `thesis_tables/checkpoint_comparison.csv`, lands in the same
12.76–12.89 range, supporting that this isn't noise.)

**Depth vs. width ablation — mLSTM only, best setting per depth:**

| mLSTM variant | Hidden size | Layers | Final PPL |
|---|---:|---:|---:|
| 1-layer (wide) | 672 | 1 | 13.44 |
| 6-layer (balanced) | 320 | 6 | 13.29 |
| 12-layer (deep) | 240 | 12 | **13.16** |

Under a fixed, matched compute/parameter budget, **going deeper (and narrower) helped
the mLSTM more than staying wide**: the 12-layer variant closes roughly a quarter of the
gap to ESM2's 12.76 (from a 0.51 PPL gap at 6 layers down to a 0.40 PPL gap at 12
layers), though it still doesn't fully close it.

### Downstream: FLIP benchmark (frozen representations, linear head, Spearman ρ — higher is better)

| Task | ESM2 (Transformer) | mLSTM | Δ (ESM2 − mLSTM) | mLSTM as % of ESM2 |
|---|---:|---:|---:|---:|
| GB1 (binding fitness) | 0.5707 | 0.5169 | +0.0538 | 90.6% |
| AAV (capsid function) | 0.5884 | 0.3932 | +0.1951 | 66.8% |
| Meltome (thermal stability) | 0.5958 | 0.5788 | +0.0169 | **97.2%** |
| **Average** | **0.5849** | **0.4963** | +0.0886 | 84.9% |

**Interpretation.** The pattern across tasks is not uniform, and that's the most
interesting result of the thesis: on **Meltome**, mLSTM keeps 97% of ESM2's
performance — thermal stability appears to depend mostly on local sequence composition
and short-range structural motifs, which a sequential/recurrent memory can capture
almost as well as attention. On **GB1**, the gap widens a little (91%) but the mLSTM
representation is still clearly useful. On **AAV**, the gap is large — mLSTM only
retains 67% of ESM2's score. AAV fitness depends heavily on long-range interactions
across a relatively long, information-dense sequence region, which plays to a
Transformer's core strength (direct, all-pairs attention) versus a recurrent model that
must compress everything it has seen so far into a fixed-size state as it scans through
the sequence. Put together: **under an identical, tightly-constrained training budget,
xLSTM is a competitive but not equal substitute for a Transformer on protein
representation learning — closest to competitive on short-range/local tasks, furthest
behind on long-range/global tasks.**

## Limitations & what I'd do next

- Both models were trained for 10,000 updates on a single GPU within a fixed wall-clock
  budget — this is a "cramming" regime, not a claim about the asymptotic ceiling of
  either architecture. A larger budget could change the gap in either direction.
- Only one dataset (UniRef50) and three downstream tasks (GB1/AAV/Meltome) were tested;
  broader FLIP/ProteinGym coverage would make the local-vs-global pattern more certain.
- The depth-vs-width ablation was only run for mLSTM; running the equivalent sweep for
  ESM2 would clarify how much of the ESM2 advantage is architecture vs. depth choice.
- Downstream heads were simple frozen-representation linear probes, not fine-tuned end
  to end — fine-tuning could change the relative ranking, especially on AAV.

## Code map

| File | What it does |
|---|---|
| `config.py` | Single source of truth for every experiment config: training hyperparameters, the ESM2/mLSTM architecture templates, the 4-way learning-rate/warmup sweep, and the 1/6/12-layer depth variants. |
| `mlstm_modelNew.py` | The mLSTM model itself — multi-head mLSTM attention block (Q/K/V + gate projections), matrix-valued memory update, chunkwise-parallel recurrence kernel, bidirectional (forward + reversed) wrapper. |
| `esm2_model.py` | The Transformer baseline — builds an ESM2-architecture model from a Hugging Face config with **random weights** (explicitly not loading the pretrained checkpoint), so it trains from scratch like the mLSTM. |
| `chunkwise/` | The chunkwise-parallel mLSTM recurrence kernel (`mlstm_chunkwise__native_custbw`) that makes training an LSTM-style model tractable at this sequence length/batch size. |
| `mlstm_utils.py` | Supporting ops for the mLSTM (e.g. normalisation) used by the model. |
| `DataLoader.py` | Streaming FASTA dataset + token-bucketed batching, and the MLM masking function (80/10/10 rule). |
| `data_setup.py` | Wraps `DataLoader.py` into ready-to-use train/val loaders for either tokenizer (ESM2 or mLSTM vocab). |
| `data_utils.py` | Shared vocabulary definitions (amino-acid ↔ id mappings, mask token id). |
| `train.py` | The unified training entrypoint — same `train_model(...)` function drives both architectures, so training is as close to apples-to-apples as possible. |
| `training_utils.py` | Shared batch/forward-pass helpers used during training and evaluation. |
| `resume_training.py`, `run_layers.py`, `keep_alive.py` | Resuming interrupted runs, driving the depth-exploration sweep, and keeping long unattended training jobs alive. |
| `evaluate.py` | Post-training upstream evaluation — loads a checkpoint, computes validation perplexity, updates the results file used for plotting/tables. |
| `ds_config.py`, `ds_data_flip.py` | FLIP benchmark task configs and data loading/splitting for GB1/AAV/Meltome. |
| `ds_eval_flip.py` | Downstream evaluation pipeline: extract frozen embeddings from a trained checkpoint, cache them, train a linear head, score with Spearman correlation. |
| `run_flip_evaluation.py` | Entrypoint that runs `ds_eval_flip.py` across all three FLIP tasks for both models. |
| `generate_checkpoint_table.py`, `generate_thesis_results.py`, `generate_pretrained_baseline.py` | Scripts that turn raw run logs/checkpoints into the CSV/TeX tables in `thesis_tables/` (the exact source of the numbers in this README). |
| `plot.py`, `plot_layers.py`, `plot_summary.py`, `plot_training_curves.py`, `comparison_plots.py`, `ds_plot_flip.py` | Plotting scripts for training curves, depth-comparison plots, and FLIP result figures. |
| `thesis_tables/` | The generated CSV/TeX/PNG result tables and figures — the ground truth for every number in this README. |

## Full thesis

The complete thesis document — literature review, full methodology, all result tables
and figures, discussion, and bibliography — covers this project in full. 

## Tech stack

`Python` · `PyTorch` · custom xLSTM/mLSTM implementation · Hugging Face `transformers`
(tokenizer/config only) · `Biopython` (FASTA parsing) · mixed-precision training ·
FLIP benchmark · Hugging Face Hub (checkpoint & dataset hosting)


## Guide

This guide explains how to reproduce the environment and run the code on:

* GPU server (recommended for training)
* CPU machine (Mac or Windows, recommended for evaluation and verification)

---

# Repository Structure

```
Codebase_root/
│
├── __main__.py                  # Main entry point
├── train.py                     # Training logic
├── config.py                    # GPU Configuration 
├── DataLoader.py
├── Data_utils.py
├── environment_server.yml      # Exact GPU server environment
├── requirements_server.txt     # Exact pip environment (GPU server)
├── environment_cpu.yml         # Portable CPU environment
├── requirements_cpu.txt        # Portable CPU requirements
├── run.sh                      # One-command run script (Mac/Linux/server)
├── run.bat                     # One-command run script (Windows)
.
.
.
└── README.md

```

---

---

# One-Command Execution (Recommended)

To simplify reproducibility, this repository includes automated run scripts for both Unix-based systems (Mac/Linux/GPU servers) and Windows. These scripts automatically create the required environment, install dependencies, and execute the program.


From the Codebase root directory, run:
##### Note: Do you want to re-run the baseline evaluation? (yes/no): type no

## Window
```
run.bat
```

## Mac / GPU Server

### Make it executable:
```bash
chmod +x run.sh
```

### Run 
```
./run.sh
```

# Option 1 — Run on GPU Server (Recommended)

This reproduces the exact environment used for thesis experiments.

## Step 1 — Create environment

```
conda env create -f environment_server.yml
```

## Step 2 — Activate environment

```
conda activate thesis
```

## Step 3 — Install pip dependencies

```
pip install -r requirements_server.txt
```

## Step 4 — Run the program

```
python __main__.py
```

---

# Option 2 — Run on CPU (Mac or Windows)

This is recommended for:

* verifying reproducibility
* running inference
* testing the code without GPU

## Step 1 — Create CPU environment

```
conda env create -f environment_cpu.yml
```

## Step 2 — Activate environment

```
conda activate thesis_cpu
```

## Step 3 — Install pip dependencies

```
pip install -r requirements_cpu.txt
```

## Step 4 — Run the program

```
python __main__.py
```

---

# Expected behavior

The script will:

* Load configuration
* Load protein dataset
* Initialize the model
* Run training or evaluation depending on configuration

---

Output locations:
* 📊 Upstream plots:    plots_depth_comparison/
* 📊 Downstream plots:  results/plots/
* 📄 Tables:            thesis_tables/
* 📄 Report:            reports/full_report.txt
* 💾 Server outputs:    thesis_outputs/
* 💾 Outputs:           thesis_outputs/

---
# Verify installation

Run this to confirm PyTorch is working:

```
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected:

GPU server:

```
CUDA available: True
```

CPU machine:

```
CUDA available: False
```

This is correct.

---

# Notes on portability

The CPU environment intentionally excludes GPU-specific CUDA packages to ensure compatibility with:

* macOS (Intel and Apple Silicon)
* Windows
* Linux CPU systems

The server environment includes GPU-specific dependencies and should only be used on compatible GPU systems.

---

# Reproducibility guarantee

The following files ensure full reproducibility:

```
environment_server.yml
requirements_server.txt
environment_cpu.yml
requirements_cpu.txt
```

These files allow the exact reconstruction of both GPU and CPU environments.

---

# Author

Akintunde Ojutiku
MSc Artificial Intelligence
Johannes Kepler University Linz
2026

---

# Contact

For questions regarding reproducibility or execution, please contact the author: aojutiku@live.com

## Data & Checkpoints
- Trained checkpoints: https://huggingface.co/2undeoju/xlstm-protein-checkpoints
- Training data: https://huggingface.co/datasets/2undeoju/xlstm-protein-training-data
