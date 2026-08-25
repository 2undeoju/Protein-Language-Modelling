"""
UNIFIED MAIN - Protein Language Model Thesis Workflow
======================================================
ONE COMMAND for everything:
    python __main__.py

WHAT IT DOES:
  1. Checks ALL upstream training files (.pt + .pkl in training_data/)
  2. Checks ALL downstream results (.pkl in results/)
  3. If files exist → SKIP training/evaluation, regenerate all plots/tables
  4. If files missing → Train/evaluate what is needed, save to correct folders
  5. Generates ALL plots, tables (upstream + downstream)
  6. Copies all outputs to LOCAL folder and SERVER folder

SUPERVISOR Note:
  - No retraining if files exist
  - One command gives ALL results
  - All plots/tables saved in organized output folders
  - Full report generated automatically

CONFIGURE PATHS AT THE TOP OF THIS FILE (see USER CONFIGURATION section).
"""

import glob
import json
import logging
import math
import pickle
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =============================================================================
# USER CONFIGURATION - Edit these paths before running
# =============================================================================

# Where your trained models and training history live
TRAINING_DATA_DIR = Path("training_data")

# Where downstream FLIP results are saved
RESULTS_DIR = Path("results")

# Where upstream plots are saved
UPSTREAM_PLOTS_DIR = Path("plots_depth_comparison")

# Where downstream plots are saved
DOWNSTREAM_PLOTS_DIR = Path("results/plots")

# Where thesis tables are saved
TABLES_DIR = Path("thesis_tables")

# Where experiment summaries and reports go
SUMMARIES_DIR = Path("experiment_summaries")
REPORTS_DIR = Path("reports")

# LOCAL folder to copy all outputs to (your Mac/laptop)
# Set this to your local path, e.g.: Path("/Users/tunde/Desktop/thesis_outputs")
#LOCAL_OUTPUT_DIR = Path("~/Desktop/thesis_outputs").expanduser()
LOCAL_OUTPUT_DIR = Path("thesis_outputs")
LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# SERVER output folder (already your working directory - outputs will be collected here)
SERVER_OUTPUT_DIR = Path("thesis_outputs")
SERVER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Best checkpoint config for downstream evaluation (from your plots: lr1e-3, warmup=1k was best)
BEST_ESM2_CHECKPOINT  = "training_data/esm2_baseline_lr1e-3_w1k_final.pt"
BEST_MLSTM_CHECKPOINT = "training_data/mlstm_6layer_lr1e-3_w1k_final.pt"

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: UPSTREAM FILE DETECTION
# =============================================================================

# All expected upstream experiments
UPSTREAM_EXPERIMENTS = [
    # ESM2 baselines (4 configs)
    "esm2_baseline_lr1e-3_w1k",
    "esm2_baseline_lr1e-3_w2k",
    "esm2_baseline_lr4e-4_w1k",
    "esm2_baseline_lr4e-4_w2k",
    # ESM2 standard naming (4 configs)
    "esm2_lr1e-3_w1000",
    "esm2_lr1e-3_w2000",
    "esm2_lr4e-4_w1000",
    "esm2_lr4e-4_w2000",
    # mLSTM 1-layer (4 configs)
    "mlstm_1layer_lr1e-3_w1k",
    "mlstm_1layer_lr1e-3_w2k",
    "mlstm_1layer_lr4e-4_w1k",
    "mlstm_1layer_lr4e-4_w2k",
    # mLSTM 6-layer (4 configs)
    "mlstm_6layer_lr1e-3_w1k",
    "mlstm_6layer_lr1e-3_w2k",
    "mlstm_6layer_lr4e-4_w1k",
    "mlstm_6layer_lr4e-4_w2k",
    # mLSTM 12-layer (4 configs)
    "mlstm_12layer_lr1e-3_w1k",
    "mlstm_12layer_lr1e-3_w2k",
    "mlstm_12layer_lr4e-4_w1k",
    "mlstm_12layer_lr4e-4_w2k",
    # mLSTM standard naming (4 configs)
    "mlstm_lr1e-3_w1000",
    "mlstm_lr1e-3_w2000",
    "mlstm_lr4e-4_w1000",
    "mlstm_lr4e-4_w2000",
]

# Downstream FLIP tasks
DOWNSTREAM_TASKS = ["gb1", "aav", "meltome"]


def check_upstream_file(experiment_name: str) -> Dict:
    """
    Check if upstream .pt and .pkl files exist for an experiment.
    ONLY looks in training_data/ folder (as updated in train.py and config.py).
    """
    # Both files must be in training_data/ - this is where train.py saves them
    final_pt  = TRAINING_DATA_DIR / f"{experiment_name}_final.pt"
    train_pkl = TRAINING_DATA_DIR / f"{experiment_name}_training_data.pkl"

    has_pt  = final_pt.exists()
    has_pkl = train_pkl.exists()

    return {
        "name":     experiment_name,
        "has_pt":   has_pt,
        "has_pkl":  has_pkl,
        "complete": has_pt and has_pkl,
        "pt_path":  str(final_pt),
        "pkl_path": str(train_pkl),
    }


def check_downstream_checkpoints() -> Dict:
    """
    Check that the BEST checkpoint .pt files exist in training_data/
    before downstream evaluation can run.
    These are set in ds_config.py CHECKPOINTS and loaded by ds_utils.py.
    """
    return {
        "esm2":  {
            "path":    BEST_ESM2_CHECKPOINT,
            "exists":  Path(BEST_ESM2_CHECKPOINT).exists(),
        },
        "mlstm": {
            "path":    BEST_MLSTM_CHECKPOINT,
            "exists":  Path(BEST_MLSTM_CHECKPOINT).exists(),
        },
    }


def check_downstream_results() -> Dict:
    """Check if downstream FLIP results .pkl exist for all tasks."""
    status = {}
    for task in DOWNSTREAM_TASKS:
        results_file = RESULTS_DIR / f"{task}_results.pkl"
        status[task] = {
            "complete":      results_file.exists(),
            "results_file":  str(results_file),
        }
    return status


def print_upstream_status(upstream_status: List[Dict]) -> Tuple[int, int]:
    """Print a clear table of upstream file status. Returns (complete, missing)."""
    complete = sum(1 for s in upstream_status if s["complete"])
    missing  = len(upstream_status) - complete

    print("\n" + "=" * 70)
    print("UPSTREAM STATUS (training_data/ folder)")
    print("=" * 70)
    print(f"{'Experiment':<40} {'  .pt':>6} {'  .pkl':>6} {'Status':>10}")
    print("-" * 70)

    for s in upstream_status:
        pt_mark  = "✅" if s["has_pt"]  else "❌"
        pkl_mark = "✅" if s["has_pkl"] else "❌"
        status   = "COMPLETE" if s["complete"] else "MISSING"
        print(f"{s['name']:<40} {pt_mark:>6} {pkl_mark:>6} {status:>10}")

    print("-" * 70)
    print(f"Complete: {complete}/{len(upstream_status)}")
    print("=" * 70)
    return complete, missing


def print_downstream_status(downstream_status: Dict) -> Tuple[int, int]:
    """Print downstream FLIP results status including checkpoint availability."""
    # Show which checkpoints ds_utils.py will load from training_data/
    ckpt_status = check_downstream_checkpoints()
    print("\n" + "=" * 70)
    print("DOWNSTREAM - CHECKPOINTS IN training_data/ (loaded by ds_utils.py)")
    print("=" * 70)
    for model, info in ckpt_status.items():
        mark = "✅" if info["exists"] else "❌"
        print(f"  {mark} {model.upper():8s}: {info['path']}")
    if not all(v["exists"] for v in ckpt_status.values()):
        print("  ⚠️  WARNING: Missing checkpoint(s) - downstream evaluation cannot run!")

    # Show per-task results status
    complete = sum(1 for v in downstream_status.values() if v["complete"])
    missing  = len(downstream_status) - complete

    print("\n" + "=" * 70)
    print("DOWNSTREAM STATUS - FLIP Results (results/ folder)")
    print("=" * 70)
    print(f"  {'Task':<12} {'results .pkl':>14} {'Status':>12}")
    print("-" * 45)
    for task, info in downstream_status.items():
        mark   = "✅" if info["complete"] else "❌"
        status = "COMPLETE" if info["complete"] else "MISSING"
        print(f"  {task.upper():<12} {mark:>14} {status:>12}")
    print("-" * 45)
    print(f"  Complete: {complete}/{len(downstream_status)}")
    print("=" * 70)
    return complete, missing


# =============================================================================
# SECTION 2: UPSTREAM TRAINING (only if needed)
# =============================================================================

def run_upstream_training() -> None:
    """Run upstream training for any missing experiments."""
    try:
        from config import (
            DEVICE, ESM2_CONFIG, MLSTM_CONFIG,
            get_all_layer_experiments, get_all_esm2_baselines,
        )
        from train import train_model
        from esm2_model import build_esm2_model
        from mlstm_modelNew import build_mlstm_model
    except ImportError as e:
        logger.error(f"❌ Cannot import training modules: {e}")
        return

    # Collect all experiments from config
    exps = []

    # ESM2 baselines
    for cfg in get_all_esm2_baselines():
        exps.append(("esm2", cfg))

    # mLSTM depth variants
    for _, __, cfg in get_all_layer_experiments():
        exps.append(("mlstm", cfg))

    total = len(exps)
    trained = 0

    for idx, (model_type, cfg) in enumerate(exps, 1):
        name = cfg["experiment_name"]
        status = check_upstream_file(name)

        if status["complete"]:
            logger.info(f"⏭️  [{idx}/{total}] Skipping {name} (already complete)")
            continue

        logger.info(f"\n🚀 [{idx}/{total}] Training: {name}")
        logger.info(f"   LR={cfg.get('lr')}, Warmup={cfg.get('warmup_steps')}")

        try:
            if model_type == "esm2":
                model = build_esm2_model(cfg).to(DEVICE)
            else:
                model = build_mlstm_model(cfg).to(DEVICE)

            # Explicitly set output_dir to training_data/ so _final.pt and
            # _training_data.pkl are saved there (as updated in train.py)
            cfg_with_dir = dict(cfg)
            cfg_with_dir["output_dir"] = str(TRAINING_DATA_DIR)

            train_model(model=model, model_type=model_type, device=DEVICE, config=cfg_with_dir)
            trained += 1
            logger.info(f"✅ Completed: {name}")
            logger.info(f"   Saved: {TRAINING_DATA_DIR}/{name}_final.pt")
            logger.info(f"   Saved: {TRAINING_DATA_DIR}/{name}_training_data.pkl")

        except Exception as e:
            logger.error(f"❌ Failed: {name} | {e}", exc_info=True)

    logger.info(f"\n✅ Upstream training done. Newly trained: {trained}")


# =============================================================================
# SECTION 3: DOWNSTREAM EVALUATION (only if needed)
# =============================================================================

def run_downstream_evaluation(tasks_to_run: List[str]) -> None:
    """
    Run FLIP downstream evaluation for missing tasks.

    Loads model checkpoints from training_data/ via ds_config.py CHECKPOINTS dict,
    exactly as updated in ds_utils.py load_model() function.
    ds_*.py files must be in the same directory as __main__.py (pyscript/).
    """
    if not tasks_to_run:
        return

    # ── CRITICAL: verify the best .pt checkpoints exist in training_data/ ──
    # ds_utils.py load_model() reads these paths from ds_config.py CHECKPOINTS
    ckpt_status = check_downstream_checkpoints()
    missing_ckpts = [m for m, v in ckpt_status.items() if not v["exists"]]
    if missing_ckpts:
        logger.error("❌ Cannot run downstream evaluation - missing checkpoints:")
        for m in missing_ckpts:
            logger.error(f"   {ckpt_status[m]['path']}")
        logger.error(f"   Make sure these _final.pt files are in {TRAINING_DATA_DIR}/")
        logger.error("   Then update ds_config.py CHECKPOINTS dict to match.")
        return

    # ── Import downstream modules (now in same pyscript/ directory) ──
    try:
        from ds_eval_flip import extract_or_load_embeddings, evaluate_model_on_task
        from ds_data_flip import load_splits, prepare_flip_task
        from ds_config import PATHS
    except ImportError as e:
        logger.error(f"❌ Cannot import downstream modules: {e}")
        logger.error("   Make sure ds_*.py files are in pyscript/ (same dir as __main__.py)")
        return

    # Use PATHS.results_dir from ds_config.py (consistent with ds_eval_flip.py)
    results_dir = Path(PATHS.results_dir) if hasattr(PATHS, "results_dir") else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks_to_run:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"EVALUATING: {task.upper()}")
        logger.info(f"  ESM2 checkpoint:  {BEST_ESM2_CHECKPOINT}")
        logger.info(f"  mLSTM checkpoint: {BEST_MLSTM_CHECKPOINT}")
        logger.info(f"{'=' * 60}")

        try:
            # Load or prepare FLIP data splits
            try:
                splits = load_splits(task)
            except FileNotFoundError:
                logger.info(f"Preparing {task} data splits...")
                splits = prepare_flip_task(task)

            # Extract embeddings from training_data/ checkpoints + evaluate
            results = {}
            for model_name in ["esm2", "mlstm"]:
                logger.info(f"\nExtracting {model_name.upper()} embeddings for {task}...")
                # ds_utils.py load_model() reads from training_data/ via ds_config.CHECKPOINTS
                embeddings = extract_or_load_embeddings(model_name, task, splits)
                logger.info(f"Evaluating {model_name.upper()} on {task}...")
                results[model_name] = evaluate_model_on_task(
                    model_name, task, splits, embeddings
                )

            # Save results to results/ (consistent with ds_config.PATHS.results_dir)
            results_file = results_dir / f"{task}_results.pkl"
            with open(results_file, "wb") as f:
                pickle.dump(results, f)
            logger.info(f"✅ Saved: {results_file}")

        except Exception as e:
            logger.error(f"❌ Failed to evaluate {task}: {e}", exc_info=True)


# =============================================================================
# SECTION 4: GENERATE ALL UPSTREAM PLOTS
# =============================================================================

def generate_upstream_plots() -> None:
    """Generate all upstream training curve plots."""
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING UPSTREAM PLOTS")
    logger.info("=" * 70)

    # 1. Pretrained baseline reference line
    try:
        from generate_pretrained_baseline import main as baseline_main
        logger.info("Step 1/4: Checking pretrained ESM2-8M baseline...")
        baseline_main()
        logger.info("✅ Baseline ready")
    except Exception as e:
        logger.warning(f"⚠️  Baseline step failed: {e}")

    # 2. ESM2 vs mLSTM comparison plots (per config)
    # NOTE: comparison_plots.py line 200 must call load_baseline() with NO arguments
    # so it uses the default path "training_data/esm2_8m_pretrained_baseline.pkl"
    # We patch this at runtime to be safe.
    try:
        import comparison_plots as _cp
        # Runtime patch: ensure load_baseline() is called without override argument
        # (fixes the bug where line 200 passed the filename, overriding the training_data/ default)
        _orig_main = _cp.main
        def _patched_main():
            # Replace any hardcoded call to load_baseline("esm2_8m_pretrained_baseline.pkl")
            # by monkey-patching load_baseline to ignore any positional arg
            _orig_load = _cp.load_baseline
            def _safe_load_baseline(path=None):
                # Always use the default (training_data/) path regardless of argument
                return _orig_load()
            _cp.load_baseline = _safe_load_baseline
            try:
                return _orig_main()
            finally:
                _cp.load_baseline = _orig_load
        logger.info("Step 2/4: Generating ESM2 vs mLSTM comparison plots...")
        _patched_main()
        logger.info("✅ Comparison plots generated")
    except Exception as e:
        logger.warning(f"⚠️  Comparison plots failed: {e}")

    # 3. Depth comparison plots (1/6/12 layer per config)
    try:
        from plot_layers import create_all_comparison_plots
        logger.info("Step 3/4: Generating depth comparison plots...")
        create_all_comparison_plots()
        logger.info("✅ Depth comparison plots generated")
    except Exception as e:
        logger.warning(f"⚠️  Depth comparison plots failed: {e}")

    # 4. Summary grid (2×2 all configs in one figure)
    try:
        from plot_summary import create_summary_grid
        logger.info("Step 4/4: Generating summary grid...")
        create_summary_grid()
        logger.info("✅ Summary grid generated")
    except Exception as e:
        logger.warning(f"⚠️  Summary grid failed: {e}")


# =============================================================================
# SECTION 5: GENERATE ALL DOWNSTREAM PLOTS AND TABLES
# =============================================================================

def generate_downstream_plots() -> None:
    """Generate all downstream FLIP plots and tables."""
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING DOWNSTREAM PLOTS & TABLES")
    logger.info("=" * 70)

    DOWNSTREAM_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Per-task scatter plots and bar charts
    try:
        from ds_plot_flip import plot_all_tasks_results
        logger.info("Step 1/3: Generating FLIP scatter + bar plots...")
        plot_all_tasks_results()
        logger.info("✅ FLIP task plots generated")
    except Exception as e:
        logger.warning(f"⚠️  FLIP task plots failed: {e}")

    # 2. Unified Spearman results table (PNG + PDF + LaTeX + CSV)
    _generate_results_table()

    # 3. Checkpoint comparison table
    _generate_checkpoint_table()


def _generate_results_table() -> None:
    """Generate the unified Spearman results table in all formats."""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        logger.info("Step 2/3: Generating unified results table...")

        # Load results
        task_data = {}
        for task in DOWNSTREAM_TASKS:
            results_file = RESULTS_DIR / f"{task}_results.pkl"
            if not results_file.exists():
                logger.warning(f"⚠️  Missing: {results_file}")
                continue
            with open(results_file, "rb") as f:
                task_data[task] = pickle.load(f)

        if not task_data:
            logger.warning("⚠️  No downstream results found, skipping table")
            return

        # Collect scores
        esm2_scores  = []
        mlstm_scores = []
        task_labels  = []

        for task in DOWNSTREAM_TASKS:
            if task not in task_data:
                continue
            esm2_s  = task_data[task].get("esm2",  {}).get("test_spearman", float("nan"))
            mlstm_s = task_data[task].get("mlstm", {}).get("test_spearman", float("nan"))
            esm2_scores.append(esm2_s)
            mlstm_scores.append(mlstm_s)
            task_labels.append(task.upper())

        if not task_labels:
            return

        esm2_avg  = float(np.nanmean(esm2_scores))
        mlstm_avg = float(np.nanmean(mlstm_scores))

        all_tasks   = task_labels + ["Average"]
        all_esm2    = esm2_scores + [esm2_avg]
        all_mlstm   = mlstm_scores + [mlstm_avg]
        all_delta   = [e - m for e, m in zip(all_esm2, all_mlstm)]
        all_pct     = [
            f"{(m / e * 100):.1f}%" if e > 0 else "N/A"
            for e, m in zip(all_esm2, all_mlstm)
        ]

        # ---- PNG / PDF figure ----
        header_color = "#1f4788"
        row_colors   = ["#f0f0f0", "white"] * 4
        avg_color    = "#e8f4f8"

        table_data = []
        for i, task in enumerate(all_tasks):
            table_data.append([
                task,
                f"{all_esm2[i]:.4f}",
                f"{all_mlstm[i]:.4f}",
                f"{all_delta[i]:+.4f}",
                all_pct[i],
            ])

        for fmt, figsize, fontsize in [("full", (14, 6), 14), ("compact", (10, 4), 12)]:
            fig, ax = plt.subplots(figsize=figsize)
            ax.axis("tight")
            ax.axis("off")

            tbl = ax.table(
                cellText=table_data,
                colLabels=[
                    "Task",
                    "ESM2 (Transformer)",
                    "mLSTM (Recurrent)",
                    "Δ (ESM2 − mLSTM)",
                    "mLSTM / ESM2",
                ],
                cellLoc="center",
                loc="center",
                colWidths=[0.15, 0.22, 0.22, 0.22, 0.19],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(fontsize)
            tbl.scale(1, 2.4)

            # Header
            for j in range(5):
                cell = tbl[(0, j)]
                cell.set_facecolor(header_color)
                cell.set_text_props(weight="bold", color="white", fontsize=fontsize + 1)

            # Rows
            for i in range(1, len(all_tasks) + 1):
                for j in range(5):
                    cell = tbl[(i, j)]
                    if i == len(all_tasks):
                        cell.set_facecolor(avg_color)
                        cell.set_text_props(weight="bold")
                    else:
                        cell.set_facecolor(row_colors[(i - 1) % 2])
                    if j == 3:
                        cell.set_text_props(color="#2e7d32", weight="bold")
                    if j == 0:
                        cell.set_text_props(weight="bold")

            plt.title(
                "Downstream Task Performance: Spearman Correlation\n"
                "(~8M Parameters, Cramming Constraints, UniRef50)",
                fontsize=fontsize + 2,
                weight="bold",
                pad=16,
            )
            fig.text(
                0.5, 0.02,
                "FLIP Benchmark | 24h Training on Single GPU | ESM2 vs mLSTM",
                ha="center",
                fontsize=fontsize - 2,
                style="italic",
                color="#666666",
            )
            plt.tight_layout(rect=[0, 0.06, 1, 0.95])

            for ext in ["png", "pdf"]:
                out = TABLES_DIR / f"downstream_results_{fmt}.{ext}"
                plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()

        # ---- CSV ----
        csv_path = TABLES_DIR / "downstream_results.csv"
        with open(csv_path, "w") as f:
            f.write("Task,ESM2_Spearman,mLSTM_Spearman,Delta,mLSTM_pct_of_ESM2\n")
            for i, task in enumerate(all_tasks):
                f.write(f"{task},{all_esm2[i]:.4f},{all_mlstm[i]:.4f},{all_delta[i]:+.4f},{all_pct[i]}\n")

        # ---- LaTeX ----
        tex_path = TABLES_DIR / "downstream_results.tex"
        with open(tex_path, "w") as f:
            f.write("% Downstream Results Table - Copy into thesis\n\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Downstream evaluation on FLIP benchmark (Spearman correlation). "
                    "Models trained from scratch under 24-hour cramming constraints (~8M parameters).}\n")
            f.write("\\label{tab:downstream_results}\n")
            f.write("\\begin{tabular}{l r r r r}\n")
            f.write("\\toprule\n")
            f.write("Task & ESM2 & mLSTM & $\\Delta$ & mLSTM/ESM2 \\\\\n")
            f.write("\\midrule\n")
            for i, task in enumerate(all_tasks):
                if task == "Average":
                    f.write("\\midrule\n")
                    f.write(f"\\textbf{{{task}}} & \\textbf{{{all_esm2[i]:.4f}}} "
                            f"& \\textbf{{{all_mlstm[i]:.4f}}} "
                            f"& \\textbf{{{all_delta[i]:+.4f}}} "
                            f"& \\textbf{{{all_pct[i]}}} \\\\\n")
                else:
                    f.write(f"{task} & {all_esm2[i]:.4f} & {all_mlstm[i]:.4f} "
                            f"& {all_delta[i]:+.4f} & {all_pct[i]} \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        logger.info(f"✅ Results tables saved to {TABLES_DIR}/")
        logger.info(f"   - downstream_results_full.png/pdf")
        logger.info(f"   - downstream_results_compact.png/pdf")
        logger.info(f"   - downstream_results.csv")
        logger.info(f"   - downstream_results.tex")

    except Exception as e:
        logger.warning(f"⚠️  Results table generation failed: {e}", exc_info=True)


def _generate_checkpoint_table() -> None:
    """Generate checkpoint comparison table across training steps."""
    try:
        import numpy as np
        logger.info("Step 3/3: Generating checkpoint comparison table...")

        checkpoints_to_show = [1000, 5000, 10000]

        rows = []
        for exp_name in UPSTREAM_EXPERIMENTS:
            pkl_path = TRAINING_DATA_DIR / f"{exp_name}_training_data.pkl"
            if not pkl_path.exists():
                continue
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            val_steps = data.get("val_steps", [])
            val_ppl   = data.get("val_perplexities", [])

            if not val_steps:
                val_losses = data.get("val_losses", [])
                val_ppl = [math.exp(l) for l in val_losses] if val_losses else []

            row = {"Model": exp_name}
            for ckpt in checkpoints_to_show:
                if val_steps and val_ppl:
                    steps_arr = list(val_steps)
                    diffs = [abs(s - ckpt) for s in steps_arr]
                    idx = diffs.index(min(diffs))
                    row[f"Step {ckpt}"] = (
                        f"{val_ppl[idx]:.4f}" if diffs[idx] <= 500 else "N/A"
                    )
                else:
                    row[f"Step {ckpt}"] = "N/A"

            final_ppl = data.get("final_val_perplexity")
            row["Final"] = f"{final_ppl:.4f}" if final_ppl else "N/A"
            rows.append(row)

        if not rows:
            logger.warning("⚠️  No upstream PKL files found for checkpoint table")
            return

        TABLES_DIR.mkdir(parents=True, exist_ok=True)

        # Text table
        txt_path = TABLES_DIR / "checkpoint_comparison.txt"
        with open(txt_path, "w") as f:
            f.write("=" * 85 + "\n")
            f.write("CHECKPOINT COMPARISON - VALIDATION PERPLEXITY\n")
            f.write("=" * 85 + "\n")
            f.write(f"{'Model':<45} {'Step 1000':>10} {'Step 5000':>10} {'Step 10000':>11} {'Final':>8}\n")
            f.write("-" * 85 + "\n")
            for row in rows:
                f.write(
                    f"{row['Model']:<45}"
                    f"{row.get('Step 1000','N/A'):>10}"
                    f"{row.get('Step 5000','N/A'):>10}"
                    f"{row.get('Step 10000','N/A'):>11}"
                    f"{row.get('Final','N/A'):>8}\n"
                )
            f.write("=" * 85 + "\n")

        # LaTeX table
        tex_path = TABLES_DIR / "checkpoint_comparison.tex"
        with open(tex_path, "w") as f:
            f.write("% Checkpoint Comparison Table\n\n")
            f.write("\\begin{table}[htbp]\n\\centering\n")
            f.write("\\caption{Validation perplexity at key training checkpoints}\n")
            f.write("\\label{tab:checkpoint_comparison}\n")
            f.write("\\begin{tabular}{l r r r r}\n\\toprule\n")
            f.write("Model & Step 1000 & Step 5000 & Step 10000 & Final \\\\\n\\midrule\n")
            for row in rows:
                f.write(
                    f"{row['Model']} & {row.get('Step 1000','--')} "
                    f"& {row.get('Step 5000','--')} "
                    f"& {row.get('Step 10000','--')} "
                    f"& {row.get('Final','--')} \\\\\n"
                )
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

        # CSV
        csv_path = TABLES_DIR / "checkpoint_comparison.csv"
        with open(csv_path, "w") as f:
            f.write("Model,Step 1000,Step 5000,Step 10000,Final\n")
            for row in rows:
                f.write(
                    f"{row['Model']},"
                    f"{row.get('Step 1000','N/A')},"
                    f"{row.get('Step 5000','N/A')},"
                    f"{row.get('Step 10000','N/A')},"
                    f"{row.get('Final','N/A')}\n"
                )

        logger.info(f"✅ Checkpoint tables saved to {TABLES_DIR}/")

    except Exception as e:
        logger.warning(f"⚠️  Checkpoint table failed: {e}", exc_info=True)


# =============================================================================
# SECTION 6: GENERATE FINAL REPORT
# =============================================================================

def generate_final_report(upstream_status: List[Dict], downstream_status: Dict) -> Path:
    """Generate comprehensive final report (text + JSON)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    ts        = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    txt_file  = REPORTS_DIR / f"full_report_{ts}.txt"
    json_file = REPORTS_DIR / f"full_report_{ts}.json"

    # Load downstream results for report
    downstream_results = {}
    for task in DOWNSTREAM_TASKS:
        rf = RESULTS_DIR / f"{task}_results.pkl"
        if rf.exists():
            with open(rf, "rb") as f:
                downstream_results[task] = pickle.load(f)

    # JSON output
    report_data = {
        "timestamp":          ts,
        "upstream_summary":   upstream_status,
        "downstream_summary": downstream_status,
        "downstream_results": {
            task: {
                model: {
                    "test_spearman": res.get("test_spearman"),
                    "test_mse":      res.get("test_mse"),
                }
                for model, res in results.items()
                if isinstance(res, dict)
            }
            for task, results in downstream_results.items()
        },
    }
    with open(json_file, "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    # Text report
    with open(txt_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PROTEIN LANGUAGE MODEL - COMPLETE THESIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # ---- Upstream summary ----
        f.write("=" * 80 + "\n")
        f.write("UPSTREAM TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Experiment':<45} {'  .pt':>5} {'  .pkl':>6} {'Status':>10}\n")
        f.write("-" * 75 + "\n")
        for s in upstream_status:
            pt  = "YES" if s["has_pt"]  else "NO"
            pkl = "YES" if s["has_pkl"] else "NO"
            st  = "COMPLETE" if s["complete"] else "MISSING"
            f.write(f"{s['name']:<45} {pt:>5} {pkl:>6} {st:>10}\n")

        total_up   = len(upstream_status)
        done_up    = sum(1 for s in upstream_status if s["complete"])
        f.write("-" * 75 + "\n")
        f.write(f"Complete: {done_up}/{total_up}\n\n")

        # ---- Best upstream models ----
        f.write("=" * 80 + "\n")
        f.write("BEST UPSTREAM MODELS (by final validation perplexity)\n")
        f.write("=" * 80 + "\n")

        perplexities = []
        for s in upstream_status:
            if not s["complete"]:
                continue
            pkl_path = Path(s["pkl_path"])
            if not pkl_path.exists():
                continue
            try:
                with open(pkl_path, "rb") as fp:
                    data = pickle.load(fp)
                ppl = data.get("final_val_perplexity")
                if ppl:
                    perplexities.append((s["name"], ppl))
            except Exception:
                pass

        if perplexities:
            perplexities.sort(key=lambda x: x[1])
            f.write("\nTop 5 models:\n")
            for i, (name, ppl) in enumerate(perplexities[:5], 1):
                f.write(f"  {i}. {name}: {ppl:.4f}\n")

            best_esm2  = [(n, p) for n, p in perplexities if "esm2"  in n]
            best_mlstm = [(n, p) for n, p in perplexities if "mlstm" in n]
            if best_esm2:
                f.write(f"\nBest ESM2:  {best_esm2[0][0]} ({best_esm2[0][1]:.4f})\n")
            if best_mlstm:
                f.write(f"Best mLSTM: {best_mlstm[0][0]} ({best_mlstm[0][1]:.4f})\n")

        # ---- Downstream results ----
        f.write("\n" + "=" * 80 + "\n")
        f.write("DOWNSTREAM EVALUATION RESULTS (FLIP Benchmark)\n")
        f.write("=" * 80 + "\n")

        if downstream_results:
            f.write(f"\n{'Task':<12} {'ESM2':>12} {'mLSTM':>12} {'Δ':>10} {'mLSTM%':>10}\n")
            f.write("-" * 60 + "\n")

            esm2_all  = []
            mlstm_all = []

            for task in DOWNSTREAM_TASKS:
                if task not in downstream_results:
                    f.write(f"{task.upper():<12} {'N/A':>12} {'N/A':>12}\n")
                    continue
                res  = downstream_results[task]
                esm2_s  = res.get("esm2",  {}).get("test_spearman", float("nan"))
                mlstm_s = res.get("mlstm", {}).get("test_spearman", float("nan"))
                delta   = esm2_s - mlstm_s
                pct     = f"{mlstm_s / esm2_s * 100:.1f}%" if esm2_s > 0 else "N/A"
                esm2_all.append(esm2_s)
                mlstm_all.append(mlstm_s)
                f.write(f"{task.upper():<12} {esm2_s:>12.4f} {mlstm_s:>12.4f} {delta:>+10.4f} {pct:>10}\n")

            if esm2_all:
                avg_e = sum(esm2_all) / len(esm2_all)
                avg_m = sum(mlstm_all) / len(mlstm_all)
                avg_d = avg_e - avg_m
                avg_p = f"{avg_m / avg_e * 100:.1f}%" if avg_e > 0 else "N/A"
                f.write("-" * 60 + "\n")
                f.write(f"{'AVERAGE':<12} {avg_e:>12.4f} {avg_m:>12.4f} {avg_d:>+10.4f} {avg_p:>10}\n")
        else:
            f.write("\n⚠️  No downstream results found. Run evaluation first.\n")

        # ---- Output locations ----
        f.write("\n" + "=" * 80 + "\n")
        f.write("OUTPUT LOCATIONS\n")
        f.write("=" * 80 + "\n")
        f.write(f"📁 Training data:         {TRAINING_DATA_DIR}/\n")
        f.write(f"📊 Upstream plots:        {UPSTREAM_PLOTS_DIR}/\n")
        f.write(f"📊 Downstream plots:      {DOWNSTREAM_PLOTS_DIR}/\n")
        f.write(f"📁 Results:               {RESULTS_DIR}/\n")
        f.write(f"📄 Tables (LaTeX/CSV):    {TABLES_DIR}/\n")
        f.write(f"📄 Reports:               {REPORTS_DIR}/\n")
        f.write(f"💾 Server outputs:        {SERVER_OUTPUT_DIR}/\n")
        f.write(f"💾 Local outputs:         {LOCAL_OUTPUT_DIR}/\n")
        f.write("=" * 80 + "\n")

    logger.info(f"📄 Report saved: {txt_file}")
    logger.info(f"📄 JSON saved:   {json_file}")
    return txt_file


# =============================================================================
# SECTION 7: COPY OUTPUTS (LOCAL + SERVER)
# =============================================================================

def copy_outputs_to_folders() -> None:
    """
    Collect all output files and copy to:
    1. SERVER_OUTPUT_DIR (organised folder on the server)
    2. LOCAL_OUTPUT_DIR  (your Mac/laptop via mounted path)
    """
    logger.info("\n" + "=" * 70)
    logger.info("COPYING OUTPUTS TO SERVER + LOCAL FOLDERS")
    logger.info("=" * 70)

    # Folders to collect from
    source_dirs = [
        UPSTREAM_PLOTS_DIR,
        DOWNSTREAM_PLOTS_DIR,
        TABLES_DIR,
        REPORTS_DIR,
        SUMMARIES_DIR,
    ]

    for dest_root in [SERVER_OUTPUT_DIR, LOCAL_OUTPUT_DIR]:
        try:
            dest_root.mkdir(parents=True, exist_ok=True)

            copied = 0
            for src_dir in source_dirs:
                if not src_dir.exists():
                    continue
                # Mirror directory structure
                dest_dir = dest_root / src_dir.name
                dest_dir.mkdir(parents=True, exist_ok=True)

                for src_file in src_dir.rglob("*"):
                    if src_file.is_file():
                        dst_file = dest_dir / src_file.relative_to(src_dir)
                        dst_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_file, dst_file)
                        copied += 1

            logger.info(f"✅ Copied {copied} files → {dest_root}/")

        except PermissionError:
            logger.warning(f"⚠️  Cannot write to {dest_root}/ (check path/permissions)")
        except Exception as e:
            logger.warning(f"⚠️  Copy to {dest_root}/ failed: {e}")


# =============================================================================
# SECTION 8: MAIN ENTRY POINT
# =============================================================================

def _ensure_pyscript_in_path() -> None:
    """
    Guarantee that the pyscript/ directory (where __main__.py lives) is on sys.path.
    This ensures all ds_*.py files (moved from flip_files/ to pyscript/) can be imported
    without ModuleNotFoundError regardless of how the script is invoked.
    """
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
        logger.debug(f"Added to sys.path: {script_dir}")


def print_banner() -> None:
    print("\n" + "=" * 70)
    print("  PROTEIN LANGUAGE MODEL - UNIFIED THESIS WORKFLOW")
    print("  ESM2 (Transformer) vs mLSTM (Recurrent) - Cramming Study")
    print("=" * 70)
    print("  This script will:")
    print("  ✓ Check ALL upstream .pt and .pkl files in training_data/")
    print("  ✓ Check ALL downstream results in results/")
    print("  ✓ Train / evaluate ONLY what is missing")
    print("  ✓ Generate ALL plots (upstream + downstream)")
    print("  ✓ Generate ALL tables (LaTeX + CSV + PNG + PDF)")
    print("  ✓ Copy everything to local + server output folders")
    print("  ✓ Generate a comprehensive thesis report")
    print("=" * 70 + "\n")


def main() -> int:
    # Ensure pyscript/ is on sys.path so all ds_*.py and upstream modules import correctly
    # (ds_*.py were moved from flip_files/ to pyscript/ alongside __main__.py)
    _ensure_pyscript_in_path()

    print_banner()

    # Create all output directories
    for d in [TRAINING_DATA_DIR, RESULTS_DIR, UPSTREAM_PLOTS_DIR,
              DOWNSTREAM_PLOTS_DIR, TABLES_DIR, SUMMARIES_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1: Check upstream status
    # ------------------------------------------------------------------
    print("\n📋 STEP 1/7: Checking upstream files...")
    upstream_status = [check_upstream_file(name) for name in UPSTREAM_EXPERIMENTS]
    up_complete, up_missing = print_upstream_status(upstream_status)

    # ------------------------------------------------------------------
    # STEP 2: Check downstream status
    # ------------------------------------------------------------------
    print("\n📋 STEP 2/7: Checking downstream results...")
    downstream_status = check_downstream_results()
    dn_complete, dn_missing = print_downstream_status(downstream_status)

    # ------------------------------------------------------------------
    # STEP 3: Train upstream if needed
    # ------------------------------------------------------------------
    if up_missing > 0:
        print(f"\n🚀 STEP 3/7: Training {up_missing} missing upstream experiments...")
        run_upstream_training()
    else:
        print(f"\n✅ STEP 3/7: All upstream experiments complete - skipping training")

    # ------------------------------------------------------------------
    # STEP 4: Run downstream evaluation if needed
    # ------------------------------------------------------------------
    tasks_to_run = [t for t, v in downstream_status.items() if not v["complete"]]
    if tasks_to_run:
        print(f"\n🚀 STEP 4/7: Running downstream evaluation for: {tasks_to_run}")
        run_downstream_evaluation(tasks_to_run)
    else:
        print(f"\n✅ STEP 4/7: All downstream results complete - skipping evaluation")

    # ------------------------------------------------------------------
    # STEP 5: Generate all upstream plots
    # ------------------------------------------------------------------
    print("\n📊 STEP 5/7: Generating upstream plots...")
    generate_upstream_plots()

    # ------------------------------------------------------------------
    # STEP 6: Generate all downstream plots and tables
    # ------------------------------------------------------------------
    print("\n📊 STEP 6/7: Generating downstream plots and tables...")
    generate_downstream_plots()

    # ------------------------------------------------------------------
    # STEP 7: Generate report + copy to local and server
    # ------------------------------------------------------------------
    print("\n📄 STEP 7/7: Generating report and copying outputs...")
    # Refresh upstream status after any new training
    upstream_status    = [check_upstream_file(name) for name in UPSTREAM_EXPERIMENTS]
    downstream_status  = check_downstream_results()
    report_file        = generate_final_report(upstream_status, downstream_status)
    copy_outputs_to_folders()

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    up_done = sum(1 for s in upstream_status if s["complete"])
    dn_done = sum(1 for v in downstream_status.values() if v["complete"])

    print("\n" + "=" * 70)
    print("✅  ALL DONE - THESIS WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"\nUpstream:   {up_done}/{len(upstream_status)} experiments complete")
    print(f"Downstream: {dn_done}/{len(downstream_status)} tasks complete")
    print(f"\nOutput locations:")
    print(f"  📊 Upstream plots:    {UPSTREAM_PLOTS_DIR}/")
    print(f"  📊 Downstream plots:  {DOWNSTREAM_PLOTS_DIR}/")
    print(f"  📄 Tables:            {TABLES_DIR}/")
    print(f"  📄 Report:            {report_file}")
    print(f"  💾 Server outputs:    {SERVER_OUTPUT_DIR}/")
    print(f"  💾 Local outputs:     {LOCAL_OUTPUT_DIR}/")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())