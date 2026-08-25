"""
Checkpoint Comparison Table Generator

Generates tables showing model performance at key training checkpoints.
Outputs both text and LaTeX formats for thesis inclusion.

Usage:
    python generate_checkpoint_table.py
"""

import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Key checkpoints to compare
CHECKPOINTS = [1000, 5000, 10000]


def load_experiment_data(experiment_name: str) -> Optional[Dict]:
    """Load training data for an experiment."""
    # Try organized structure first
    organized_path = Path("experiments") / experiment_name / "training_data.pkl"
    if organized_path.exists():
        with open(organized_path, 'rb') as f:
            return pickle.load(f)
    
    # Try flat structure
    import glob
    possible_files = glob.glob(f"*{experiment_name}*training_data*.pkl")
    if possible_files:
        with open(possible_files[0], 'rb') as f:
            return pickle.load(f)
    
    return None


def get_perplexity_at_checkpoint(data: Dict, checkpoint: int) -> Optional[float]:
    """Get validation perplexity at specific checkpoint."""
    val_steps = data.get('val_steps', [])
    val_perplexities = data.get('val_perplexities', [])
    
    if not val_steps or not val_perplexities:
        return None
    
    # Find closest validation step to checkpoint
    val_steps_array = np.array(val_steps)
    idx = np.argmin(np.abs(val_steps_array - checkpoint))
    
    # Only use if within reasonable range (±250 steps)
    if abs(val_steps[idx] - checkpoint) <= 250:
        return val_perplexities[idx]
    
    return None


def generate_checkpoint_table():
    """Generate checkpoint comparison table for all experiments."""
    
    print("\n" + "="*80)
    print("CHECKPOINT COMPARISON TABLE GENERATOR")
    print("="*80 + "\n")
    
    # Define all experiments
    experiments = {
        'Baseline Models': [
            ('ESM2 6-layer', 'esm2_lr4e-4_w1000'),
            ('mLSTM 6-layer', 'mlstm_lr4e-4_w1000'),
        ],
        'mLSTM 1-Layer': [
            ('1L lr1e-3 w1k', 'mlstm_1layer_lr1e-3_w1k'),
            ('1L lr1e-3 w2k', 'mlstm_1layer_lr1e-3_w2k'),
            ('1L lr4e-4 w1k', 'mlstm_1layer_lr4e-4_w1k'),
            ('1L lr4e-4 w2k', 'mlstm_1layer_lr4e-4_w2k'),
        ],
        'mLSTM 6-Layer': [
            ('6L lr1e-3 w1k', 'mlstm_6layer_lr1e-3_w1k'),
            ('6L lr1e-3 w2k', 'mlstm_6layer_lr1e-3_w2k'),
            ('6L lr4e-4 w1k', 'mlstm_6layer_lr4e-4_w1k'),
            ('6L lr4e-4 w2k', 'mlstm_6layer_lr4e-4_w2k'),
        ],
        'mLSTM 12-Layer': [
            ('12L lr1e-3 w1k', 'mlstm_12layer_lr1e-3_w1k'),
            ('12L lr1e-3 w2k', 'mlstm_12layer_lr1e-3_w2k'),
            ('12L lr4e-4 w1k', 'mlstm_12layer_lr4e-4_w1k'),
            ('12L lr4e-4 w2k', 'mlstm_12layer_lr4e-4_w2k'),
        ],
    }
    
    # Collect data
    results = []
    
    for category, exp_list in experiments.items():
        for display_name, exp_name in exp_list:
            data = load_experiment_data(exp_name)
            if data:
                row = {'Model': display_name}
                for checkpoint in CHECKPOINTS:
                    ppl = get_perplexity_at_checkpoint(data, checkpoint)
                    row[f'Step {checkpoint}'] = ppl
                results.append(row)
            else:
                print(f"⚠️  Data not found: {exp_name}")
    
    if not results:
        print("❌ No data found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Create output directory
    output_dir = Path("thesis_tables")
    output_dir.mkdir(exist_ok=True)
    
    # ========================================================================
    # TEXT FORMAT
    # ========================================================================
    text_file = output_dir / "checkpoint_comparison.txt"
    with open(text_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CHECKPOINT COMPARISON - VALIDATION PERPLEXITY\n")
        f.write("="*80 + "\n\n")
        
        f.write("Performance at key training checkpoints:\n")
        f.write("-" * 80 + "\n\n")
        
        # Format table
        f.write(f"{'Model':<20} {'Step 1000':>12} {'Step 5000':>12} {'Step 10000':>12}\n")
        f.write("-" * 80 + "\n")
        
        for _, row in df.iterrows():
            f.write(f"{row['Model']:<20}")
            for checkpoint in CHECKPOINTS:
                val = row[f'Step {checkpoint}']
                if pd.notna(val):
                    f.write(f"{val:>12.4f}")
                else:
                    f.write(f"{'N/A':>12}")
            f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"✅ Text table saved: {text_file}")
    
    # ========================================================================
    # LATEX FORMAT
    # ========================================================================
    latex_file = output_dir / "checkpoint_comparison.tex"
    with open(latex_file, 'w') as f:
        f.write("% Checkpoint Comparison Table - LaTeX Format\n")
        f.write("% Copy this into your thesis document\n\n")
        
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Validation perplexity at key training checkpoints}\n")
        f.write("\\label{tab:checkpoint_comparison}\n")
        f.write("\\begin{tabular}{l r r r}\n")
        f.write("\\toprule\n")
        f.write("Model & Step 1000 & Step 5000 & Step 10000 \\\\\n")
        f.write("\\midrule\n")
        
        current_category = None
        for category, exp_list in experiments.items():
            # Add category separator
            if current_category is not None:
                f.write("\\midrule\n")
            current_category = category
            
            for display_name, exp_name in exp_list:
                # Find this row in DataFrame
                row = df[df['Model'] == display_name]
                if not row.empty:
                    row = row.iloc[0]
                    f.write(f"{display_name}")
                    for checkpoint in CHECKPOINTS:
                        val = row[f'Step {checkpoint}']
                        if pd.notna(val):
                            f.write(f" & {val:.2f}")
                        else:
                            f.write(" & --")
                    f.write(" \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"✅ LaTeX table saved: {latex_file}")
    
    # ========================================================================
    # CSV FORMAT (for Excel)
    # ========================================================================
    csv_file = output_dir / "checkpoint_comparison.csv"
    df.to_csv(csv_file, index=False)
    print(f"✅ CSV saved: {csv_file}")
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    summary_file = output_dir / "checkpoint_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CHECKPOINT COMPARISON - SUMMARY STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        for checkpoint in CHECKPOINTS:
            col = f'Step {checkpoint}'
            valid_values = df[col].dropna()
            
            if len(valid_values) > 0:
                f.write(f"\n{col}:\n")
                f.write(f"  Best: {valid_values.min():.4f}\n")
                f.write(f"  Worst: {valid_values.max():.4f}\n")
                f.write(f"  Mean: {valid_values.mean():.4f}\n")
                f.write(f"  Std: {valid_values.std():.4f}\n")
                
                # Find best model
                best_idx = df[col].idxmin()
                best_model = df.loc[best_idx, 'Model']
                f.write(f"  Best model: {best_model}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✅ Summary statistics saved: {summary_file}")
    
    # ========================================================================
    # DISPLAY PREVIEW
    # ========================================================================
    print("\n" + "="*80)
    print("PREVIEW")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    print(f"\n📁 All files saved to: {output_dir}/")
    print("  • checkpoint_comparison.txt  - For reading")
    print("  • checkpoint_comparison.tex  - For thesis (LaTeX)")
    print("  • checkpoint_comparison.csv  - For Excel/analysis")
    print("  • checkpoint_summary.txt     - Summary statistics")


if __name__ == "__main__":
    generate_checkpoint_table()
