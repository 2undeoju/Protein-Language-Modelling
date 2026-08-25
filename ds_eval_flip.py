"""
FLIP Benchmark Downstream Evaluation
Evaluates ESM2 and mLSTM on GB1, AAV, and Meltome fitness prediction

Extended from ds_eval_gb1.py to support all FLIP tasks
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict

from ds_config import PATHS, TASK_CONFIGS, EMBEDDING_DIMS
from ds_utils import load_model, get_embeddings, train_head, evaluate_head
from ds_data_flip import load_splits, prepare_flip_task

# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

def extract_or_load_embeddings(model_name: str, task: str, splits: Dict) -> Dict[str, np.ndarray]:
    """
    Extract embeddings for all splits or load from cache

    Args:
        model_name: 'esm2' or 'mlstm'
        task: 'gb1', 'aav', or 'meltome'
        splits: Dict with train/val/test DataFrames

    Returns:
        Dict with train/val/test embeddings
    """
    print("\n" + "="*60)
    print(f"EXTRACTING {model_name.upper()} EMBEDDINGS FOR {task.upper()}")
    print("="*60)

    # Check cache
    cache_file = PATHS.embeddings_dir / f"{task}_{model_name}_embeddings.pkl"

    if cache_file.exists():
        print(f"✓ Loading cached embeddings: {cache_file}")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)

        # ---- NEW: validate cached sequences against current splits ----
        # Support both old format (dict of arrays) and new format (dict of {seqs, emb})
        is_new_format = (
                isinstance(cached, dict)
                and "train" in cached
                and isinstance(cached["train"], dict)
                and "seqs" in cached["train"]
                and "emb" in cached["train"]
        )

        if is_new_format:
            mismatch = False
            for split_name in ["train", "val", "test"]:
                current_seqs = splits[split_name]["sequence"].tolist()
                cached_seqs = cached[split_name]["seqs"]
                if current_seqs != cached_seqs:
                    mismatch = True
                    print(f"⚠️ Cache mismatch in {split_name}: sequences/order changed.")
                    break

            if not mismatch:
                print("✓ Cache valid (sequence order matches). Using cached embeddings.")
                return {k: cached[k]["emb"] for k in ["train", "val", "test"]}

            print("⚠️ Cache invalid. Re-extracting embeddings...")

        else:
            # Old cache format: assume valid only if all splits exist and not None
            bad_old_cache = (
                    not isinstance(cached, dict)
                    or any(k not in cached for k in ["train", "val", "test"])
                    or any(cached[k] is None for k in ["train", "val", "test"])
            )
            if not bad_old_cache:
                print("⚠️ Old cache format detected (no seq check). Using anyway.")
                return cached

            print("⚠️ Old cache invalid/None. Re-extracting embeddings...")
        # --------------------------------------------------------------
    # Check cache
    #cache_file = PATHS.embeddings_dir / f"{task}_{model_name}_embeddings.pkl"

    #if cache_file.exists():
    #    print(f"✓ Loading cached embeddings: {cache_file}")
    #    with open(cache_file, 'rb') as f:
    #        return pickle.load(f)

    # Load model
    model = load_model(model_name)

    # Get task-specific config
    config = TASK_CONFIGS[task]

    # Use task-specific batch size for mLSTM (needs more memory)
    from ds_config import MLSTM_BATCH_OVERRIDES
    batch_size = config['batch_size_embed']
    if model_name == 'mlstm' and task in MLSTM_BATCH_OVERRIDES:
        batch_size = MLSTM_BATCH_OVERRIDES[task]
        print(f"Using mLSTM-specific batch size for {task}: {batch_size}")

    # Extract embeddings for each split
    embeddings = {}
    for split_name in ['train', 'val', 'test']:
        sequences = splits[split_name]['sequence'].tolist()

        print(f"\n{split_name.upper()} set:")
        split_embeddings = get_embeddings(
            model=model,
            model_name=model_name,
            sequences=sequences,
            batch_size=batch_size,
            task=task
        )

        embeddings[split_name] = split_embeddings

    # ---- NEW: cache embeddings WITH sequences to guarantee alignment ----
    cache_payload = {}
    for split_name in ["train", "val", "test"]:
        cache_payload[split_name] = {
            "seqs": splits[split_name]["sequence"].tolist(),
            "emb": embeddings[split_name]
        }

    with open(cache_file, "wb") as f:
        pickle.dump(cache_payload, f)

    print(f"\n✓ Cached embeddings (with seq order) to: {cache_file}")
    # -------------------------------------------------------------------

    # Cache embeddings
    #with open(cache_file, 'wb') as f:
    #    pickle.dump(embeddings, f)
    #print(f"\n✓ Cached embeddings to: {cache_file}")

    # Clear model from memory to free GPU
    del model
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings

# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_model_on_task(model_name: str, task: str, splits: Dict, embeddings: Dict) -> Dict:
    """
    Train and evaluate a model on a specific task

    Args:
        model_name: 'esm2' or 'mlstm'
        task: 'gb1', 'aav', or 'meltome'
        splits: Dict with train/val/test DataFrames
        embeddings: Dict with train/val/test embeddings

    Returns:
        Dict with results
    """

    print("\n" + "="*60)
    print(f"EVALUATING {model_name.upper()} ON {task.upper()}")
    print("="*60)

    # Targets (raw)
    train_targets = splits['train']['target'].values
    val_targets   = splits['val']['target'].values
    test_targets  = splits['test']['target'].values

    # Task config
    config = TASK_CONFIGS[task].copy()
    config["task"] = task

    # Train head (returns mu/std from train normalization)
    head, mu, std = train_head(
        train_embeddings=embeddings['train'],
        train_targets=train_targets,
        val_embeddings=embeddings['val'],
        val_targets=val_targets,
        config=config
    )

    # Standardize test targets for Spearman alignment
    test_targets_std = (test_targets.astype(np.float32) - mu) / (std + 1e-8)

    print("\nTest set evaluation:")
    test_results = evaluate_head(
        model=head,
        embeddings=embeddings['test'],
        targets=test_targets_std,
        mu=mu,
        std=std
    )

    print(f"✓ Test Spearman: {test_results['spearman']:.4f}")
    print(f"  MSE: {test_results['mse']:.4f}")

    return {
        'model': model_name,
        'task': task,
        'test_spearman': test_results['spearman'],
        'test_mse': test_results['mse'],
        'test_predictions': test_results['predictions'],
        'test_targets': test_results['targets'],  # standardized targets
        'mu': mu,
        'std': std
    }


# ============================================================================
# MULTI-TASK EVALUATION
# ============================================================================

def evaluate_model_all_tasks(model_name: str, tasks: list = None) -> Dict:
    """
    Evaluate a model on multiple FLIP tasks

    Args:
        model_name: 'esm2' or 'mlstm'
        tasks: List of task names (default: ['gb1', 'aav', 'meltome'])

    Returns:
        Dict with results for each task
    """
    if tasks is None:
        tasks = ['gb1', 'aav', 'meltome']

    print("\n" + "="*70)
    print(f"EVALUATING {model_name.upper()} ON ALL FLIP TASKS")
    print("="*70)

    # Clear GPU memory before starting
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    all_results = {}

    for task in tasks:
        try:
            print(f"\n{'='*70}")
            print(f"[{task.upper()}] Starting evaluation...")
            print(f"{'='*70}")

            # Step 1: Load/prepare data
            print(f"\n[{task.upper()}] Loading data...")
            try:
                splits = load_splits(task)
            except FileNotFoundError:
                print(f"Splits not found, preparing {task.upper()} data...")
                splits = prepare_flip_task(task)

            # Step 2: Extract embeddings
            print(f"\n[{task.upper()}] Extracting embeddings...")
            embeddings = extract_or_load_embeddings(model_name, task, splits)

            # Step 3: Evaluate
            print(f"\n[{task.upper()}] Training and evaluating...")
            results = evaluate_model_on_task(model_name, task, splits, embeddings)

            all_results[task] = results

            print(f"\n✓ {task.upper()} evaluation complete")

        except Exception as e:
            print(f"\n✗ Error evaluating {task.upper()}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    if all_results:
        print("\n" + "="*70)
        print(f"RESULTS SUMMARY FOR {model_name.upper()}")
        print("="*70)
        for task, results in all_results.items():
            print(f"{task.upper():10s} Test Spearman: {results['test_spearman']:.4f}")
        print("="*70)

    return all_results

# ============================================================================
# COMPLETE COMPARISON
# ============================================================================

def run_full_flip_evaluation(tasks: list = None):
    """
    Complete FLIP evaluation pipeline for both models

    Args:
        tasks: List of task names (default: ['gb1', 'aav', 'meltome'])

    Returns:
        Dict with results for both models
    """
    if tasks is None:
        tasks = ['gb1', 'aav', 'meltome']

    print("\n" + "="*70)
    print("FULL FLIP BENCHMARK EVALUATION")
    print("="*70)
    print(f"Tasks: {', '.join([t.upper() for t in tasks])}")
    print(f"Models: ESM2, mLSTM")
    print("="*70)

    # Evaluate both models
    print("\n[1/2] Evaluating ESM2...")
    esm2_results = evaluate_model_all_tasks('esm2', tasks)

    # CRITICAL: Clear GPU memory before switching to mLSTM
    print("\n" + "="*70)
    print("CLEARING GPU MEMORY BEFORE mLSTM")
    print("="*70)
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Print memory stats
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"GPU Memory after cleanup:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Cached: {cached:.2f} GB")
    print("="*70)

    print("\n[2/2] Evaluating mLSTM...")
    mlstm_results = evaluate_model_all_tasks('mlstm', tasks)

    # Combine results
    results = {
        'esm2': esm2_results,
        'mlstm': mlstm_results,
        'tasks': tasks
    }

    # Save combined results
    results_file = PATHS.results_dir / "flip_all_tasks_results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n✓ Results saved to: {results_file}")

    # Print final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"{'Task':<12} {'ESM2':>10} {'mLSTM':>10} {'Improvement':>12}")
    print("-"*70)

    for task in tasks:
        if task in esm2_results and task in mlstm_results:
            esm2_score = esm2_results[task]['test_spearman']
            mlstm_score = mlstm_results[task]['test_spearman']
            if abs(esm2_score) > 1e-8:
                improvement = (mlstm_score - esm2_score) / abs(esm2_score) * 100
            else:
                improvement = 0

            #improvement = ((mlstm_score / esm2_score - 1) * 100) if esm2_score > 0 else 0

            print(f"{task.upper():<12} {esm2_score:>10.4f} {mlstm_score:>10.4f} {improvement:>11.1f}%")

    print("="*70)

    return results

# ============================================================================
# SINGLE TASK EVALUATION (for backward compatibility)
# ============================================================================

def run_gb1_evaluation():
    """
    Backward compatible: Evaluate only GB1 (original function)
    """
    print("\n" + "="*60)
    print("GB1 DOWNSTREAM EVALUATION")
    print("="*60)

    # Step 1: Prepare data
    print("\n[1/4] Preparing data...")
    try:
        splits = load_splits('gb1')
    except FileNotFoundError:
        splits = prepare_flip_task('gb1')

    # Step 2: Extract embeddings
    print("\n[2/4] Extracting embeddings...")
    esm2_embeddings = extract_or_load_embeddings('esm2', 'gb1', splits)
    mlstm_embeddings = extract_or_load_embeddings('mlstm', 'gb1', splits)

    # Step 3: Evaluate both models
    print("\n[3/4] Training and evaluating...")
    esm2_results = evaluate_model_on_task('esm2', 'gb1', splits, esm2_embeddings)
    mlstm_results = evaluate_model_on_task('mlstm', 'gb1', splits, mlstm_embeddings)

    # Step 4: Save results
    print("\n[4/4] Saving results...")
    results = {
        'esm2': esm2_results,
        'mlstm': mlstm_results,
        'config': TASK_CONFIGS['gb1']
    }

    results_file = PATHS.results_dir / "gb1_results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"✓ Results saved to: {results_file}")

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"ESM2  Test Spearman: {esm2_results['test_spearman']:.4f}")
    print(f"mLSTM Test Spearman: {mlstm_results['test_spearman']:.4f}")
    print("="*60)

    return results

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()

        if arg == 'gb1':
            # Single task: GB1 only (backward compatible)
            run_gb1_evaluation()
        elif arg == 'all':
            # All tasks
            run_full_flip_evaluation()
        elif arg in TASK_CONFIGS:
            # Single specific task
            task = arg
            print(f"\nEvaluating ESM2 and mLSTM on {task.upper()} only...")

            # Load data
            try:
                splits = load_splits(task)
            except FileNotFoundError:
                splits = prepare_flip_task(task)

            # Evaluate ESM2
            esm2_embeddings = extract_or_load_embeddings('esm2', task, splits)
            esm2_results = evaluate_model_on_task('esm2', task, splits, esm2_embeddings)

            # Evaluate mLSTM
            mlstm_embeddings = extract_or_load_embeddings('mlstm', task, splits)
            mlstm_results = evaluate_model_on_task('mlstm', task, splits, mlstm_embeddings)

            # Save and print
            results = {'esm2': esm2_results, 'mlstm': mlstm_results}
            results_file = PATHS.results_dir / f"{task}_results.pkl"
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)

            print(f"\n✓ Results saved to: {results_file}")
        else:
            print(f"Unknown argument: {arg}")
            print(f"Usage: python ds_eval_flip.py [task]")
            print(f"  task: gb1, aav, meltome, or 'all'")
            sys.exit(1)
    else:
        # Default: run full evaluation
        print("Usage: python ds_eval_flip.py [task]")
        print(f"  task: gb1, aav, meltome, or 'all'")
        print("\nRunning full evaluation on all tasks...")
        run_full_flip_evaluation()