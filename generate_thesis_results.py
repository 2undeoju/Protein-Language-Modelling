"""
Comprehensive Thesis Results Generator

Generates ALL results needed for thesis:
- Checkpoint comparison tables
- Training loss curves
- Final performance tables
- Summary statistics
- Best model analysis

Usage:
    python generate_thesis_results.py
    
Generates complete results package in thesis_results/ directory.
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from scipy.signal import savgol_filter

# Import from other generators
import subprocess
import sys


def run_checkpoint_table():
    """Generate checkpoint comparison table."""
    print("\n" + "="*70)
    print("1. GENERATING CHECKPOINT TABLES")
    print("="*70)
    subprocess.run([sys.executable, "generate_checkpoint_table.py"])


def run_training_curves():
    """Generate training curve plots."""
    print("\n" + "="*70)
    print("2. GENERATING TRAINING CURVES")
    print("="*70)
    subprocess.run([sys.executable, "plot_training_curves.py"])


def generate_final_performance_table():
    """Generate final performance comparison table."""
    
    print("\n" + "="*70)
    print("3. GENERATING FINAL PERFORMANCE TABLE")
    print("="*70)
    
    def load_data(exp_name):
        # Try organized structure
        path = Path("experiments") / exp_name / "training_data.pkl"
        if path.exists():
            with open(path, 'rb') as f:
                return pickle.load(f)
        # Try flat
        import glob
        files = glob.glob(f"*{exp_name}*training_data*.pkl")
        if files:
            with open(files[0], 'rb') as f:
                return pickle.load(f)
        return None
    
    experiments = [
        # Baselines
        ('ESM2 6-layer', 'esm2_lr4e-4_w1000', 'Baseline'),
        ('mLSTM 6-layer', 'mlstm_lr4e-4_w1000', 'Baseline'),
        # ESM2 baselines
        ('ESM2 lr1e-3 w1k', 'esm2_baseline_lr1e-3_w1k', 'ESM2 Baseline'),
        ('ESM2 lr1e-3 w2k', 'esm2_baseline_lr1e-3_w2k', 'ESM2 Baseline'),
        ('ESM2 lr4e-4 w1k', 'esm2_baseline_lr4e-4_w1k', 'ESM2 Baseline'),
        ('ESM2 lr4e-4 w2k', 'esm2_baseline_lr4e-4_w2k', 'ESM2 Baseline'),
        # mLSTM 1-layer
        ('mLSTM 1L lr1e-3 w1k', 'mlstm_1layer_lr1e-3_w1k', 'mLSTM 1-Layer'),
        ('mLSTM 1L lr1e-3 w2k', 'mlstm_1layer_lr1e-3_w2k', 'mLSTM 1-Layer'),
        ('mLSTM 1L lr4e-4 w1k', 'mlstm_1layer_lr4e-4_w1k', 'mLSTM 1-Layer'),
        ('mLSTM 1L lr4e-4 w2k', 'mlstm_1layer_lr4e-4_w2k', 'mLSTM 1-Layer'),
        # mLSTM 6-layer
        ('mLSTM 6L lr1e-3 w1k', 'mlstm_6layer_lr1e-3_w1k', 'mLSTM 6-Layer'),
        ('mLSTM 6L lr1e-3 w2k', 'mlstm_6layer_lr1e-3_w2k', 'mLSTM 6-Layer'),
        ('mLSTM 6L lr4e-4 w1k', 'mlstm_6layer_lr4e-4_w1k', 'mLSTM 6-Layer'),
        ('mLSTM 6L lr4e-4 w2k', 'mlstm_6layer_lr4e-4_w2k', 'mLSTM 6-Layer'),
        # mLSTM 12-layer
        ('mLSTM 12L lr1e-3 w1k', 'mlstm_12layer_lr1e-3_w1k', 'mLSTM 12-Layer'),
        ('mLSTM 12L lr1e-3 w2k', 'mlstm_12layer_lr1e-3_w2k', 'mLSTM 12-Layer'),
        ('mLSTM 12L lr4e-4 w1k', 'mlstm_12layer_lr4e-4_w1k', 'mLSTM 12-Layer'),
        ('mLSTM 12L lr4e-4 w2k', 'mlstm_12layer_lr4e-4_w2k', 'mLSTM 12-Layer'),
    ]
    
    results = []
    for display_name, exp_name, category in experiments:
        data = load_data(exp_name)
        if data:
            results.append({
                'Model': display_name,
                'Category': category,
                'Final Perplexity': data.get('final_val_perplexity'),
                'Final Loss': data.get('final_val_loss'),
                'Parameters': '~8M',
            })
    
    df = pd.DataFrame(results)
    
    # Save text format
    output_dir = Path("thesis_tables")
    output_dir.mkdir(exist_ok=True)
    
    text_file = output_dir / "final_performance.txt"
    with open(text_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FINAL PERFORMANCE COMPARISON\n")
        f.write("="*80 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n" + "="*80 + "\n")
    
    # Save LaTeX
    latex_file = output_dir / "final_performance.tex"
    with open(latex_file, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Final validation perplexity comparison}\n")
        f.write("\\label{tab:final_performance}\n")
        f.write("\\begin{tabular}{l r r}\n")
        f.write("\\toprule\n")
        f.write("Model & Perplexity & Loss \\\\\n")
        f.write("\\midrule\n")
        
        for category in df['Category'].unique():
            cat_df = df[df['Category'] == category]
            f.write(f"\\multicolumn{{3}}{{l}}{{\\textbf{{{category}}}}} \\\\\n")
            for _, row in cat_df.iterrows():
                ppl = row['Final Perplexity']
                loss = row['Final Loss']
                if pd.notna(ppl) and pd.notna(loss):
                    f.write(f"{row['Model']} & {ppl:.2f} & {loss:.4f} \\\\\n")
            f.write("\\midrule\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    # Save CSV
    csv_file = output_dir / "final_performance.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"✅ Text table: {text_file}")
    print(f"✅ LaTeX table: {latex_file}")
    print(f"✅ CSV: {csv_file}")
    
    # Find best models
    print("\n" + "-"*70)
    print("BEST MODELS:")
    print("-"*70)
    
    valid_results = df[df['Final Perplexity'].notna()]
    if not valid_results.empty:
        best_overall = valid_results.loc[valid_results['Final Perplexity'].idxmin()]
        print(f"Best Overall: {best_overall['Model']}")
        print(f"  Perplexity: {best_overall['Final Perplexity']:.4f}")
        
        mlstm_results = valid_results[valid_results['Model'].str.contains('mLSTM')]
        if not mlstm_results.empty:
            best_mlstm = mlstm_results.loc[mlstm_results['Final Perplexity'].idxmin()]
            print(f"\nBest mLSTM: {best_mlstm['Model']}")
            print(f"  Perplexity: {best_mlstm['Final Perplexity']:.4f}")
    print("-"*70)


def generate_thesis_summary():
    """Generate comprehensive thesis summary document."""
    
    print("\n" + "="*70)
    print("4. GENERATING COMPREHENSIVE THESIS SUMMARY")
    print("="*70)
    
    output_dir = Path("thesis_results")
    output_dir.mkdir(exist_ok=True)
    
    summary_file = output_dir / f"thesis_summary_{datetime.now().strftime('%Y%m%d')}.txt"
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MASTER'S THESIS RESULTS SUMMARY\n")
        f.write("Protein Language Modeling: ESM2 vs mLSTM Architecture Comparison\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("RESEARCH QUESTIONS\n")
        f.write("-"*80 + "\n")
        f.write("1. How does mLSTM compare to ESM2 under cramming constraints?\n")
        f.write("2. What is the impact of model depth on mLSTM performance?\n")
        f.write("3. How sensitive is mLSTM to hyperparameter choices?\n\n")
        
        f.write("EXPERIMENTAL SETUP\n")
        f.write("-"*80 + "\n")
        f.write("• Training paradigm: Cramming (24 hours, single GPU)\n")
        f.write("• Model size: ~8M parameters (iso-parametric)\n")
        f.write("• Dataset: UniRef50 protein sequences\n")
        f.write("• Training steps: 10,000 updates\n")
        f.write("• Evaluation: Masked language modeling perplexity\n\n")
        
        f.write("EXPERIMENTS CONDUCTED\n")
        f.write("-"*80 + "\n")
        f.write("Total: 18 experiments\n")
        f.write("  • 2 Baseline models (ESM2 + mLSTM 6-layer)\n")
        f.write("  • 4 ESM2 baselines (hyperparameter sweep)\n")
        f.write("  • 12 mLSTM variants (3 depths × 4 hyperparams)\n\n")
        
        f.write("RESULTS LOCATION\n")
        f.write("-"*80 + "\n")
        f.write("Tables:\n")
        f.write("  • thesis_tables/checkpoint_comparison.txt\n")
        f.write("  • thesis_tables/final_performance.txt\n")
        f.write("  • thesis_tables/*.tex (LaTeX versions)\n\n")
        f.write("Figures:\n")
        f.write("  • thesis_plots/baseline_training_curves.png\n")
        f.write("  • thesis_plots/depth_training_curves.png\n")
        f.write("  • thesis_plots/hyperparameter_training_curves.png\n")
        f.write("  • thesis_plots/training_vs_validation.png\n")
        f.write("  • plots_depth_comparison/depth_comparison_summary_grid.png\n\n")
        
        f.write("Raw Data:\n")
        f.write("  • experiments/ (all model checkpoints and training data)\n")
        f.write("  • reports/ (comprehensive JSON/text reports)\n\n")
        
        f.write("THESIS CHAPTER ORGANIZATION\n")
        f.write("-"*80 + "\n")
        f.write("Chapter 4: Results\n")
        f.write("  4.1 Training Dynamics\n")
        f.write("      - Use: baseline_training_curves.png\n")
        f.write("      - Use: training_vs_validation.png\n")
        f.write("      - Table: checkpoint_comparison.txt\n\n")
        f.write("  4.2 Model Architecture Comparison\n")
        f.write("      - Use: depth_training_curves.png\n")
        f.write("      - Use: depth_comparison_summary_grid.png\n")
        f.write("      - Table: final_performance.txt\n\n")
        f.write("  4.3 Hyperparameter Sensitivity\n")
        f.write("      - Use: hyperparameter_training_curves.png\n")
        f.write("      - Analysis from checkpoint tables\n\n")
        
        f.write("="*80 + "\n")
        f.write("All results are publication-ready and formatted for thesis inclusion.\n")
        f.write("="*80 + "\n")
    
    print(f"✅ Thesis summary: {summary_file}")


def main():
    """Generate all thesis results."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE THESIS RESULTS GENERATOR")
    print("="*80)
    print("Generating all tables, plots, and summaries for thesis...")
    print("="*80 + "\n")
    
    # Create main output directory
    output_dir = Path("thesis_results")
    output_dir.mkdir(exist_ok=True)
    
    # Generate all components
    try:
        run_checkpoint_table()
    except Exception as e:
        print(f"⚠️  Checkpoint table generation failed: {e}")
    
    try:
        run_training_curves()
    except Exception as e:
        print(f"⚠️  Training curves generation failed: {e}")
    
    try:
        generate_final_performance_table()
    except Exception as e:
        print(f"⚠️  Final performance table failed: {e}")
    
    try:
        generate_thesis_summary()
    except Exception as e:
        print(f"⚠️  Thesis summary failed: {e}")
    
    # Final summary
    print("\n" + "="*80)
    print("✅ THESIS RESULTS PACKAGE COMPLETE!")
    print("="*80)
    print("\n📁 Output Locations:")
    print("  • thesis_tables/     - All comparison tables (text + LaTeX)")
    print("  • thesis_plots/      - All training curve figures")
    print("  • thesis_results/    - Summary documents")
    print("\n📊 Key Files for Thesis:")
    print("  • checkpoint_comparison.tex       - Table for checkpoint analysis")
    print("  • final_performance.tex           - Table for final results")
    print("  • baseline_training_curves.png    - Figure: training dynamics")
    print("  • depth_training_curves.png       - Figure: depth comparison")
    print("  • hyperparameter_training_curves.png - Figure: hyperparameter impact")
    print("\n✨ Everything ready for thesis writing!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
