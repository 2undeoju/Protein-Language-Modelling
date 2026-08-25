"""
FLIP Benchmark Results, Visualization without NaN-crashing

"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from ds_config import PATHS

# ============================================================================
# LOAD RESULTS
# ============================================================================

def load_single_task_results(task: str):
    """Load results for a single task"""
    results_file = PATHS.results_dir / f"{task}_results.pkl"

    if not results_file.exists():
        raise FileNotFoundError(f"Results not found: {results_file}")

    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    print(f"✓ Loaded {task.upper()} results from: {results_file}")
    return results

def load_all_tasks_results():
    """
    Load results for all tasks from individual per-task files.
    Results are stored as: results/gb1_results.pkl, results/aav_results.pkl, etc.
    Each file has format: {'esm2': {...}, 'mlstm': {...}}

    Returns combined dict:
    {
        'tasks': ['gb1', 'aav', 'meltome'],
        'esm2':  {'gb1': {...}, 'aav': {...}, 'meltome': {...}},
        'mlstm': {'gb1': {...}, 'aav': {...}, 'meltome': {...}},
    }
    """
    tasks = ['gb1', 'aav', 'meltome']
    combined = {'tasks': [], 'esm2': {}, 'mlstm': {}}

    for task in tasks:
        results_file = PATHS.results_dir / f"{task}_results.pkl"
        if not results_file.exists():
            print(f"⚠️  Missing: {results_file} — skipping {task.upper()}")
            continue
        with open(results_file, 'rb') as f:
            task_results = pickle.load(f)
        combined['tasks'].append(task)
        combined['esm2'][task]  = task_results.get('esm2',  {})
        combined['mlstm'][task] = task_results.get('mlstm', {})
        print(f"✓ Loaded {task.upper()} results from: {results_file}")

    if not combined['tasks']:
        raise FileNotFoundError(
            f"No task results found in {PATHS.results_dir}/. "
            f"Expected: gb1_results.pkl, aav_results.pkl, meltome_results.pkl"
        )
    return combined

# ============================================================================
# SINGLE TASK PLOTS - WITH NaN HANDLING
# ============================================================================

def plot_single_task_predictions(task: str, results: dict, save=True):
    """
    Plot predicted vs actual fitness for both models on a single task
    FIXED: Handles NaN predictions gracefully
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ESM2
    ax = axes[0]
    esm2 = results['esm2']

    # Filter out NaN values
    valid_mask = ~(np.isnan(esm2['test_targets']) | np.isnan(esm2['test_predictions']))
    valid_targets = esm2['test_targets'][valid_mask]
    valid_predictions = esm2['test_predictions'][valid_mask]

    if len(valid_targets) > 0:
        ax.scatter(valid_targets, valid_predictions, alpha=0.5, s=10)
        ax.plot([valid_targets.min(), valid_targets.max()],
                [valid_targets.min(), valid_targets.max()],
                'r--', lw=2, label='Perfect prediction')
        ax.set_xlabel('True Fitness', fontsize=12)
        ax.set_ylabel('Predicted Fitness', fontsize=12)

        spearman_text = f"{esm2['test_spearman']:.4f}" if not np.isnan(esm2['test_spearman']) else "NaN"
        ax.set_title(f"ESM2 on {task.upper()}\nSpearman: {spearman_text}",
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No valid predictions\n(all NaN)',
                ha='center', va='center', fontsize=14, color='red',
                transform=ax.transAxes)
        ax.set_title(f"ESM2 on {task.upper()}\nSpearman: NaN",
                     fontsize=14, fontweight='bold')

    # mLSTM
    ax = axes[1]
    mlstm = results['mlstm']

    # Filter out NaN values
    valid_mask = ~(np.isnan(mlstm['test_targets']) | np.isnan(mlstm['test_predictions']))
    valid_targets = mlstm['test_targets'][valid_mask]
    valid_predictions = mlstm['test_predictions'][valid_mask]

    if len(valid_targets) > 0:
        ax.scatter(valid_targets, valid_predictions, alpha=0.5, s=10, color='orange')
        ax.plot([valid_targets.min(), valid_targets.max()],
                [valid_targets.min(), valid_targets.max()],
                'r--', lw=2, label='Perfect prediction')
        ax.set_xlabel('True Fitness', fontsize=12)
        ax.set_ylabel('Predicted Fitness', fontsize=12)

        spearman_text = f"{mlstm['test_spearman']:.4f}" if not np.isnan(mlstm['test_spearman']) else "NaN"
        ax.set_title(f"mLSTM on {task.upper()}\nSpearman: {spearman_text}",
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No valid predictions\n(all NaN)',
                ha='center', va='center', fontsize=14, color='red',
                transform=ax.transAxes)
        ax.set_title(f"mLSTM on {task.upper()}\nSpearman: NaN",
                     fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save:
        plot_file = PATHS.plots_dir / f"{task}_predictions.pdf"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {plot_file}")

    return fig

def plot_single_task_comparison(task: str, results: dict, save=True):
    """
    Bar plot comparing model performance on a single task
    FIXED: Handles NaN values gracefully
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    models = ['ESM2', 'mLSTM']
    spearman_scores = [
        results['esm2']['test_spearman'],
        results['mlstm']['test_spearman']
    ]

    # FIXED: Filter out NaN before getting max
    valid_scores = [s for s in spearman_scores if not np.isnan(s)]

    colors = ['#3498db', '#e67e22']
    bars = ax.bar(models, spearman_scores, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bar, score in zip(bars, spearman_scores):
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.4f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., 0.05,
                    'NaN',
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')

    ax.set_ylabel('Spearman Correlation', fontsize=12)
    ax.set_title(f'{task.upper()} Fitness Prediction Performance', fontsize=14, fontweight='bold')

    # FIXED: Handle NaN in ylim
    if valid_scores:
        ax.set_ylim(0, max(valid_scores) * 1.2)
    else:
        ax.set_ylim(0, 1.0)  # Default range if all NaN
        ax.text(0.5, 0.5, 'All predictions are NaN!\nCheck model/data',
                ha='center', va='center', fontsize=12, color='red',
                transform=ax.transAxes)

    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save:
        plot_file = PATHS.plots_dir / f"{task}_comparison.pdf"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {plot_file}")

    return fig

# ============================================================================
# MULTI-TASK COMPARISON PLOTS - WITH NaN HANDLING
# ============================================================================

def plot_all_tasks_comparison(results: dict, save=True):
    """
    Bar plot comparing model performance across all tasks
    FIXED: Handles NaN values
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    tasks = results.get('tasks', ['gb1', 'aav', 'meltome'])
    tasks = [t for t in tasks if t in results['esm2'] and t in results['mlstm']]

    x = np.arange(len(tasks))
    width = 0.35

    esm2_scores = [results['esm2'][task]['test_spearman'] for task in tasks]
    mlstm_scores = [results['mlstm'][task]['test_spearman'] for task in tasks]

    # Replace NaN with 0 for plotting (will annotate as NaN)
    esm2_plot = [s if not np.isnan(s) else 0 for s in esm2_scores]
    mlstm_plot = [s if not np.isnan(s) else 0 for s in mlstm_scores]

    bars1 = ax.bar(x - width/2, esm2_plot, width, label='ESM2',
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, mlstm_plot, width, label='mLSTM',
                   color='#e67e22', alpha=0.8, edgecolor='black')

    # Add value labels
    for i, (bar1, bar2, esm2_s, mlstm_s) in enumerate(zip(bars1, bars2, esm2_scores, mlstm_scores)):
        # ESM2 label
        if not np.isnan(esm2_s):
            ax.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height(),
                    f'{esm2_s:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax.text(bar1.get_x() + bar1.get_width()/2., 0.05,
                    'NaN',
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='red')

        # mLSTM label
        if not np.isnan(mlstm_s):
            ax.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height(),
                    f'{mlstm_s:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax.text(bar2.get_x() + bar2.get_width()/2., 0.05,
                    'NaN',
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='red')

    ax.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Spearman Correlation', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Across FLIP Tasks', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in tasks], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    # FIXED: Set ylim based on valid scores only
    all_scores = esm2_scores + mlstm_scores
    valid_scores = [s for s in all_scores if not np.isnan(s)]
    if valid_scores:
        ax.set_ylim(min(valid_scores) - 0.1, max(valid_scores) * 1.2)
    else:
        ax.set_ylim(0, 1.0)

    plt.tight_layout()

    if save:
        plot_file = PATHS.plots_dir / "flip_all_tasks_comparison.pdf"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {plot_file}")

    return fig

def plot_improvement_heatmap(results: dict, save=True):
    """
    Heatmap showing mLSTM improvement over ESM2 for each task
    FIXED: Handles NaN values
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    tasks = results.get('tasks', ['gb1', 'aav', 'meltome'])
    tasks = [t for t in tasks if t in results['esm2'] and t in results['mlstm']]

    improvements = []
    task_labels = []

    for task in tasks:
        esm2_score = results['esm2'][task]['test_spearman']
        mlstm_score = results['mlstm'][task]['test_spearman']

        if not np.isnan(esm2_score) and not np.isnan(mlstm_score) and esm2_score != 0:
            improvement = ((mlstm_score / esm2_score - 1) * 100)
            improvements.append(improvement)
            task_labels.append(task.upper())
        else:
            # Skip tasks with NaN
            continue

    if not improvements:
        # All tasks have NaN - create placeholder
        ax.text(0.5, 0.5, 'No valid comparisons\n(all NaN)',
                ha='center', va='center', fontsize=14, color='red',
                transform=ax.transAxes)
        ax.set_title('Relative Performance: mLSTM vs ESM2', fontsize=14, fontweight='bold')
    else:
        # Create horizontal bar plot
        colors = ['green' if x > 0 else 'red' for x in improvements]
        bars = ax.barh(task_labels, improvements, color=colors, alpha=0.6, edgecolor='black')

        # Add value labels
        for bar, improvement in zip(bars, improvements):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{improvement:+.1f}%',
                    ha='left' if width > 0 else 'right',
                    va='center', fontsize=11, fontweight='bold')

        ax.set_xlabel('mLSTM Improvement over ESM2 (%)', fontsize=12, fontweight='bold')
        ax.set_title('Relative Performance: mLSTM vs ESM2', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()

    if save:
        plot_file = PATHS.plots_dir / "flip_improvement_comparison.pdf"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {plot_file}")

    return fig

def create_results_table(results: dict, save=True):
    """
    Create a formatted table of results
    FIXED: Handles NaN values
    """
    import pandas as pd

    tasks = results.get('tasks', ['gb1', 'aav', 'meltome'])
    tasks = [t for t in tasks if t in results['esm2'] and t in results['mlstm']]

    data = []
    for task in tasks:
        esm2_score = results['esm2'][task]['test_spearman']
        mlstm_score = results['mlstm'][task]['test_spearman']

        # FIXED: Handle NaN in improvement calculation
        if not np.isnan(esm2_score) and not np.isnan(mlstm_score) and esm2_score != 0:
            improvement = ((mlstm_score / esm2_score - 1) * 100)
            improvement_str = f"{improvement:+.1f}%"
        else:
            improvement_str = "N/A"

        data.append({
            'Task': task.upper(),
            'ESM2': f"{esm2_score:.4f}" if not np.isnan(esm2_score) else "NaN",
            'mLSTM': f"{mlstm_score:.4f}" if not np.isnan(mlstm_score) else "NaN",
            'Improvement': improvement_str
        })

    df = pd.DataFrame(data)

    # Print table
    print("\n" + "="*70)
    print("RESULTS TABLE")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)

    if save:
        # Save as CSV
        csv_file = PATHS.plots_dir / "flip_results_table.csv"
        df.to_csv(csv_file, index=False)
        print(f"✓ Saved table: {csv_file}")

        # Save as LaTeX
        latex_file = PATHS.plots_dir / "flip_results_table.tex"
        latex_str = df.to_latex(index=False)
        with open(latex_file, 'w') as f:
            f.write(latex_str)
        print(f"✓ Saved LaTeX table: {latex_file}")

    return df

# ============================================================================
# MAIN PLOTTING FUNCTIONS
# ============================================================================

def plot_single_task_results(task: str):
    """Generate all plots for a single task"""

    print("\n" + "="*60)
    print(f"PLOTTING {task.upper()} RESULTS")
    print("="*60)

    # Load results
    results = load_single_task_results(task)

    # Generate plots
    print("\nGenerating plots...")
    plot_single_task_predictions(task, results)
    plot_single_task_comparison(task, results)

    print("\n" + "="*60)
    print("PLOTTING COMPLETE")
    print("="*60)

def plot_all_tasks_results():
    """Generate all comparison plots for all tasks"""

    print("\n" + "="*60)
    print("PLOTTING ALL FLIP TASKS RESULTS")
    print("="*60)

    # Load combined results
    results = load_all_tasks_results()

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_all_tasks_comparison(results)
    plot_improvement_heatmap(results)
    create_results_table(results)

    # Generate individual task plots
    tasks = results.get('tasks', ['gb1', 'aav', 'meltome'])
    for task in tasks:
        if task in results['esm2'] and task in results['mlstm']:
            print(f"\nPlotting {task.upper()}...")
            task_results = {
                'esm2': results['esm2'][task],
                'mlstm': results['mlstm'][task]
            }
            try:
                plot_single_task_predictions(task, task_results)
                plot_single_task_comparison(task, task_results)
            except Exception as e:
                print(f"✗ Failed to plot {task}: {e}")

    print("\n" + "="*60)
    print("PLOTTING COMPLETE")
    print("="*60)

# ============================================================================
# BACKWARD COMPATIBLE FUNCTION
# ============================================================================

def plot_gb1_results():
    """
    Backward compatible: Plot GB1 results only
    """
    plot_single_task_results('gb1')

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()

        if arg == 'all':
            plot_all_tasks_results()
        elif arg in ['gb1', 'aav', 'meltome']:
            plot_single_task_results(arg)
        else:
            print(f"Unknown argument: {arg}")
            print(f"Usage: python ds_plot_flip_fixed.py [task]")
            print(f"  task: gb1, aav, meltome, or 'all'")
            sys.exit(1)
    else:
        # Default: plot all tasks
        print("Usage: python ds_plot_flip_fixed.py [task]")
        print(f"  task: gb1, aav, meltome, or 'all'")
        print("\nPlotting all tasks...")
        plot_all_tasks_results()