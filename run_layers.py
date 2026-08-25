"""
Simplified Experiment Runner for mLSTM Depth Exploration

Runs 12 experiments total:
    3 depths (1, 6, 12 layers) × 4 hyperparams (2 LR × 2 warmup)

Usage:
    # Run all 12 experiments
    python run_depth_experiments.py --all
    
    # Run specific depth (4 experiments)
    python run_depth_experiments.py --depth mlstm_6layer
    
    # Run single experiment
    python run_depth_experiments.py --experiment mlstm_6layer_lr4e-4_w1k
    
    # Dry run (preview)
    python run_depth_experiments.py --all --dry-run
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_single_experiment(config: dict, dry_run: bool = False) -> bool:
    """Run a single experiment (mLSTM or ESM2)."""
    exp_name = config['experiment_name']
    
    logger.info("="*80)
    logger.info(f"EXPERIMENT: {exp_name}")
    logger.info("="*80)
    
    is_esm2 = config.get('model_type') == 'esm2'
    
    if is_esm2:
        logger.info(f"Model: ESM2 (6-layer Transformer)")
        logger.info(f"LR: {config['lr']}, Warmup: {config['warmup_steps']}")
    else:
        logger.info(f"Layers: {config['num_layers']}, Hidden: {config['hidden_size']}")
        logger.info(f"LR: {config['lr']}, Warmup: {config['warmup_steps']}")
    logger.info("="*80)
    
    if dry_run:
        logger.info("DRY RUN - Would train model")
        return True
    
    try:
        from config import DEVICE
        from train import train_model
        
        if is_esm2:
            # Train ESM2 baseline
            from esm2_model import model
            logger.info("Using ESM2 model (from scratch)")
            
        else:
            # Train mLSTM
            from mlstm_modelNew import FullmLSTMModel
            from data_utils import vocab_size
            
            logger.info("Creating mLSTM model...")
            model = FullmLSTMModel(
                vocab_size=vocab_size,
                embed_dim=config['embed_dim'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                num_heads=config['num_heads'],
                chunk_size=config['chunk_size'],
                bidirectional=config['bidirectional'],
                dropout=config['dropout'],
                max_position_embeddings=config['max_position_embeddings'],
                bias=False,
            ).to(DEVICE)
            
            logger.info(model.get_config())
        
        # Train
        logger.info("Starting training...")
        train_model(model=model, model_type=exp_name, device=DEVICE)
        
        logger.info(f"✅ {exp_name} completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ {exp_name} failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description='Run mLSTM depth experiments with baselines')
    
    parser.add_argument('--all', action='store_true', help='Run all 12 mLSTM experiments')
    parser.add_argument('--all-with-baselines', action='store_true', 
                       help='Run all 12 mLSTM + 4 ESM2 baselines = 16 total')
    parser.add_argument('--baselines-only', action='store_true',
                       help='Run only ESM2 baselines (4 experiments)')
    parser.add_argument('--depth', choices=['mlstm_1layer', 'mlstm_6layer', 'mlstm_12layer'],
                       help='Run all 4 experiments for one depth')
    parser.add_argument('--experiment', type=str, help='Run specific experiment')
    parser.add_argument('--dry-run', action='store_true', help='Preview without running')
    
    args = parser.parse_args()
    
    if not (args.all or args.all_with_baselines or args.baselines_only or args.depth or args.experiment):
        parser.error("Must specify --all, --all-with-baselines, --baselines-only, --depth, or --experiment")
    
    from configs_layers import (
        get_experiment_config, 
        get_esm2_baseline_config,
        get_all_baselines,
        list_all_experiments, 
        DEPTH_VARIANTS, 
        HYPERPARAMETER_SWEEPS
    )
    
    # Determine which experiments to run
    experiments = []
    
    if args.experiment:
        # Single experiment
        if 'esm2_baseline' in args.experiment:
            # ESM2 baseline
            hyperparam = args.experiment.replace('esm2_baseline_', '')
            config = get_esm2_baseline_config(hyperparam)
        else:
            # mLSTM experiment
            parts = args.experiment.split('_')
            depth = f"{parts[0]}_{parts[1]}"
            hyperparam = "_".join(parts[2:])
            config = get_experiment_config(depth, hyperparam)
        experiments = [(config['experiment_name'], config)]
    
    elif args.baselines_only:
        # ESM2 baselines only
        for config in get_all_baselines():
            experiments.append((config['experiment_name'], config))
    
    elif args.depth:
        # All hyperparams for one depth
        for hyperparam in HYPERPARAMETER_SWEEPS.keys():
            config = get_experiment_config(args.depth, hyperparam)
            experiments.append((config['experiment_name'], config))
    
    elif args.all_with_baselines:
        # ESM2 baselines + all mLSTM experiments
        for config in get_all_baselines():
            experiments.append((config['experiment_name'], config))
        for depth, hyperparam, config in list_all_experiments():
            experiments.append((config['experiment_name'], config))
    
    elif args.all:
        # All 12 mLSTM experiments only
        for depth, hyperparam, config in list_all_experiments():
            experiments.append((config['experiment_name'], config))
    
    # Print plan
    logger.info(f"\nPlanned experiments: {len(experiments)}")
    for exp_name, _ in experiments:
        logger.info(f"  • {exp_name}")
    
    if not args.dry_run:
        response = input(f"\nRun {len(experiments)} experiments? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Aborted")
            return
    
    # Run experiments
    results = {}
    for exp_name, config in experiments:
        success = run_single_experiment(config, args.dry_run)
        results[exp_name] = success
        if not args.dry_run and success:
            time.sleep(10)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    for name, success in results.items():
        status = "✅" if success else "❌"
        logger.info(f"{status} {name}")
    logger.info("="*80 + "\n")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
