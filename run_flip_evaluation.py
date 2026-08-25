#!/usr/bin/env python3
"""
Usage:
    python run_flip_evaluation.py [task]

    task: gb1, aav, meltome, or 'all' (default: all)
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Run FLIP benchmark evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full evaluation on all tasks
  python run_flip_evaluation.py
  python run_flip_evaluation.py all
  
  # Run evaluation on a single task
  python run_flip_evaluation.py gb1
  python run_flip_evaluation.py aav
  python run_flip_evaluation.py meltome
  
  # Just prepare data (no evaluation)
  python run_flip_evaluation.py --data-only all
  
  # Just plot results (evaluation already done)
  python run_flip_evaluation.py --plot-only all
        """
    )

    parser.add_argument(
        'task',
        nargs='?',
        default='all',
        choices=['gb1', 'aav', 'meltome', 'all'],
        help='Task to evaluate (default: all)'
    )

    parser.add_argument(
        '--data-only',
        action='store_true',
        help='Only prepare data, skip evaluation'
    )

    parser.add_argument(
        '--plot-only',
        action='store_true',
        help='Only generate plots, skip evaluation'
    )

    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help='Skip plotting after evaluation'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("FLIP BENCHMARK EVALUATION PIPELINE")
    print("="*70)
    print(f"Task: {args.task.upper()}")

    if args.data_only:
        print("Mode: Data preparation only")
    elif args.plot_only:
        print("Mode: Plotting only")
    else:
        print("Mode: Full evaluation (data + eval + plots)")

    print("="*70)

    # ========================================================================
    # STEP 1: Data Preparation
    # ========================================================================
    if not args.plot_only:
        print("\n" + "="*70)
        print("STEP 1: DATA PREPARATION")
        print("="*70)

        from ds_data_flip import prepare_flip_task, prepare_all_flip_tasks

        try:
            if args.task == 'all':
                prepare_all_flip_tasks()
            else:
                prepare_flip_task(args.task)

            print("\n✓ Data preparation complete")
        except Exception as e:
            print(f"\n✗ Data preparation failed: {e}")
            import traceback
            traceback.print_exc()
            if args.data_only:
                return 1
            # Continue even if data prep fails (might already exist)

    # Stop here if data-only mode
    if args.data_only:
        print("\n" + "="*70)
        print("DATA PREPARATION COMPLETE")
        print("="*70)
        return 0

    # ========================================================================
    # STEP 2: Evaluation
    # ========================================================================
    if not args.plot_only:
        print("\n" + "="*70)
        print("STEP 2: MODEL EVALUATION")
        print("="*70)

        from ds_eval_flip import run_full_flip_evaluation, evaluate_model_all_tasks
        from ds_data_flip import load_splits, prepare_flip_task
        from ds_eval_flip import extract_or_load_embeddings, evaluate_model_on_task

        try:
            if args.task == 'all':
                # Run full evaluation
                run_full_flip_evaluation()
            else:
                # Run single task evaluation
                print(f"\nEvaluating ESM2 and mLSTM on {args.task.upper()}...")

                # Load data
                try:
                    splits = load_splits(args.task)
                except FileNotFoundError:
                    print(f"Splits not found, preparing data...")
                    splits = prepare_flip_task(args.task)

                # Evaluate ESM2
                print("\nEvaluating ESM2...")
                esm2_embeddings = extract_or_load_embeddings('esm2', args.task, splits)
                esm2_results = evaluate_model_on_task('esm2', args.task, splits, esm2_embeddings)

                # Evaluate mLSTM
                print("\nEvaluating mLSTM...")
                mlstm_embeddings = extract_or_load_embeddings('mlstm', args.task, splits)
                mlstm_results = evaluate_model_on_task('mlstm', args.task, splits, mlstm_embeddings)

                # Save
                import pickle
                from ds_config import PATHS
                results = {'esm2': esm2_results, 'mlstm': mlstm_results}
                results_file = PATHS.results_dir / f"{args.task}_results.pkl"
                with open(results_file, 'wb') as f:
                    pickle.dump(results, f)

                print(f"\n✓ Results saved to: {results_file}")

            print("\n✓ Evaluation complete")
        except Exception as e:
            print(f"\n✗ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # ========================================================================
    # STEP 3: Plotting
    # ========================================================================
    if not args.skip_plots:
        print("\n" + "="*70)
        print("STEP 3: GENERATING PLOTS")
        print("="*70)

        from ds_plot_flip import plot_all_tasks_results, plot_single_task_results

        try:
            if args.task == 'all':
                plot_all_tasks_results()
            else:
                plot_single_task_results(args.task)

            print("\n✓ Plotting complete")
        except Exception as e:
            print(f"\n✗ Plotting failed: {e}")
            import traceback
            traceback.print_exc()
            # Don't fail if plotting doesn't work

    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "="*70)
    print("EVALUATION PIPELINE COMPLETE")
    print("="*70)

    from ds_config import PATHS
    print(f"\nResults saved to: {PATHS.results_dir}")
    print(f"Plots saved to: {PATHS.plots_dir}")

    # Print what was generated
    print("\nGenerated files:")
    if args.task == 'all':
        print("  - flip_all_tasks_results.pkl")
        print("  - flip_all_tasks_comparison.pdf")
        print("  - flip_improvement_comparison.pdf")
        print("  - flip_results_table.csv")
        print("  - gb1/aav/meltome individual plots")
    else:
        print(f"  - {args.task}_results.pkl")
        print(f"  - {args.task}_predictions.pdf")
        print(f"  - {args.task}_comparison.pdf")

    print("\n" + "="*70)
    print("✓ SUCCESS!")
    print("="*70 + "\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
