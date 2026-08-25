# xLSTM Protein Language Modelling Under Resource Constraints — Reproducible Environment and Execution Guide

This repository contains the implementation of the xLSTM-based protein language model developed for the MSc thesis:

**"xLSTM for Bi-directional Protein Modelling Under Resource Constraints"**
Johannes Kepler University Linz, 2026

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
