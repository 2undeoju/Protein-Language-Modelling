"""
FLIP Benchmark Data Download and Processing
Unified data loader for GB1, AAV, and Meltome tasks

Extended from ds_data_gb1.py to support all FLIP tasks
"""

import pickle
import requests
import zipfile
from pathlib import Path
import pandas as pd
from typing import Dict

from ds_config import PATHS, TASK_CONFIGS

# ============================================================================
# DOWNLOAD
# ============================================================================

def download_flip_task(task: str) -> Path:
    """
    Download FLIP task data from repository (zip file with CSV splits)
    
    Args:
        task: Task name ('gb1', 'aav', 'meltome')
    
    Returns:
        Path to splits directory
    """
    if task not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task}. Must be one of {list(TASK_CONFIGS.keys())}")
    
    config = TASK_CONFIGS[task]
    
    print("\n" + "="*60)
    print(f"DOWNLOADING {task.upper()} DATA")
    print("="*60)
    
    # Setup paths
    task_dir = PATHS.data_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    
    zip_file = task_dir / "splits.zip"
    splits_dir = task_dir / "splits"
    
    # Check if already downloaded and extracted
    split_name = config['split_name']
    split_file = splits_dir / f"{split_name}.csv"
    
    if split_file.exists():
        print(f"✓ Split already exists: {split_file}")
        return splits_dir
    
    # Download zip file if not exists
    if not zip_file.exists():
        url = config['data_url']
        print(f"Downloading from: {url}")
        
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            with open(zip_file, 'wb') as f:
                f.write(response.content)
            
            print(f"✓ Downloaded to: {zip_file}")
            
        except Exception as e:
            print(f"✗ Download failed: {e}")
            raise
    
    # Extract zip file
    print(f"Extracting {zip_file}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(task_dir)
    
    print(f"✓ Extracted to: {task_dir}")
    
    # Verify split file exists
    if not split_file.exists():
        # Try to find CSV files
        csv_files = list(splits_dir.glob("*.csv"))
        raise FileNotFoundError(
            f"Split '{split_name}.csv' not found.\n"
            f"Looking for: {split_file}\n"
            f"Available CSV files: {[f.name for f in csv_files]}"
        )
    
    return splits_dir

# ============================================================================
# PROCESS
# ============================================================================

def process_flip_task(task: str, splits_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Process FLIP task data from CSV file
    
    Args:
        task: Task name ('gb1', 'aav', 'meltome')
        splits_dir: Path to splits directory
    
    Returns:
        Dict with 'train', 'val', 'test' DataFrames
    """
    if task not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task}")
    
    config = TASK_CONFIGS[task]
    
    print("\n" + "="*60)
    print(f"PROCESSING {task.upper()} DATA")
    print("="*60)
    
    # Load CSV file
    split_name = config['split_name']
    csv_file = splits_dir / f"{split_name}.csv"
    
    print(f"Loading split: {split_name}")
    print(f"From file: {csv_file}")
    
    # Read CSV
    df = pd.read_csv(csv_file)
    # ---- NEW: normalize 'set' for ALL tasks ----
    if "set" not in df.columns:
        raise ValueError(f"Expected 'set' column in CSV, found: {list(df.columns)}")
    df["set"] = df["set"].astype(str).str.strip().str.lower()
    # AAV can contain NaN / junk labels → keep only official rows
    if task == "aav":
        df = df[df["set"].isin(["train", "test"])].reset_index(drop=True)
    #if task == "aav":
    #    df["set"] = df["set"].astype(str).str.strip().str.lower()
    #    df = df[df["set"].isin(["train", "test"])].reset_index(drop=True)
     #   #df = df[df["set"].notna()].reset_index(drop=True)
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # FLIP CSV format:
    # - 'set': train/test (main split)
    # - 'validation': boolean flag for validation subset from train
    # - 'sequence': amino acid sequence
    # - 'target': fitness value
    
    #if 'set' not in df.columns:
    #    raise ValueError(f"Expected 'set' column in CSV, found: {list(df.columns)}")
    
    # DEBUG: Check what's actually in the columns
    print(f"\nDEBUG: Full dataframe 'set' unique values: {df['set'].unique()}")
    print(f"DEBUG: Full dataframe 'set' value counts:")
    print(df['set'].value_counts())
    
    if 'validation' in df.columns:
        print(f"\nDEBUG: Full dataframe 'validation' unique values: {df['validation'].unique()}")
        print(f"DEBUG: Full dataframe 'validation' value counts:")
        print(df['validation'].value_counts())
    
    splits = {}
    
    # Get training data (set == 'train')
    train_df = df[df['set'] == 'train'].copy()
    
    print(f"\nDEBUG: Original train_df has {len(train_df)} rows")
    
    # Split training into train and validation
    if 'validation' in df.columns:
        print(f"DEBUG: 'validation' column exists")
        print(f"DEBUG: validation dtype: {train_df['validation'].dtype}")
        print(f"DEBUG: validation unique values: {train_df['validation'].unique()}")
        print(f"DEBUG: validation value counts:")
        print(train_df['validation'].value_counts())
        
        # Handle NaN as training (not validation)
        is_validation = train_df['validation'].fillna(False)
        
        # Convert to boolean
        if is_validation.dtype == 'object':
            # String values
            is_validation = is_validation.isin([True, 'True', 'true', 'TRUE', 1, '1'])
        elif is_validation.dtype in ['float64', 'int64']:
            # Numeric values
            is_validation = is_validation == 1
        else:
            # Already boolean
            is_validation = is_validation == True
        
        val_df = train_df[is_validation].copy()
        train_df = train_df[~is_validation].copy()
        
        print(f"\nDEBUG: After split:")
        print(f"  train_df: {len(train_df)} rows")
        print(f"  val_df: {len(val_df)} rows")
        
        # Sanity check
        if len(train_df) == 0:
            raise ValueError(
                "ERROR: Training set is empty after splitting!\n"
                f"Original train rows: {len(train_df) + len(val_df)}\n"
                f"Validation rows: {len(val_df)}\n"
                "The 'validation' column logic may be incorrect."
            )
    else:
        # If no validation column, manually split 10% for validation
        print("Warning: No 'validation' column found, using random 10% split")
        val_size = int(len(train_df) * 0.1)
        val_df = train_df.sample(n=val_size, random_state=42)
        train_df = train_df.drop(val_df.index)
    
    # Get test data
    test_df = df[df['set'] == 'test'].copy()
    
    # Clean up - keep only sequence and target
    for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        if 'sequence' not in split_df.columns or 'target' not in split_df.columns:
            raise ValueError(f"Missing required columns in {name}")
        
        splits[name] = split_df[['sequence', 'target']].reset_index(drop=True)
        print(f"✓ {name:5s}: {len(splits[name]):6d} sequences")
    
    # Print sample
    print("\nSample data:")
    sample = splits['train'].head(3)
    for idx, row in sample.iterrows():
        seq = row['sequence']
        target = row['target']
        print(f"  {seq[:30]}... → {target:.3f}")
    
    return splits

# ============================================================================
# SAVE/LOAD
# ============================================================================

def save_splits(task: str, splits: Dict[str, pd.DataFrame]):
    """Save processed splits for a task"""
    task_dir = PATHS.data_dir / task
    splits_file = task_dir / f"{task}_splits.pkl"
    
    with open(splits_file, 'wb') as f:
        pickle.dump(splits, f)
    
    print(f"\n✓ Saved splits to: {splits_file}")

def load_splits(task: str) -> Dict[str, pd.DataFrame]:
    """Load processed splits for a task"""
    task_dir = PATHS.data_dir / task
    splits_file = task_dir / f"{task}_splits.pkl"
    
    if not splits_file.exists():
        raise FileNotFoundError(f"Splits not found: {splits_file}")
    
    with open(splits_file, 'rb') as f:
        splits = pickle.load(f)
    
    print(f"✓ Loaded {task.upper()} splits from: {splits_file}")
    return splits

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def prepare_flip_task(task: str, force_rebuild: bool = False):
    """Complete pipeline: download → process → save

    If a saved splits .pkl exists and force_rebuild=False, reuse it.
    This keeps split order stable so cached embeddings stay aligned.
    """

    print("\n" + "="*70)
    print(f"PREPARING {task.upper()} DATA")
    print("="*70)

    # (1) Path to the saved splits file
    split_pkl = PATHS.data_dir / task / f"{task}_splits.pkl"

    # (2) NEW: load existing splits instead of rebuilding
    if split_pkl.exists() and not force_rebuild:
        print(f"✓ Using existing splits: {split_pkl}")
        with open(split_pkl, "rb") as f:
            splits = pickle.load(f)

        print("\n" + "="*60)
        print(f"{task.upper()} DATA READY! (loaded existing splits)")
        print("="*60)

        return splits

    # (3) Otherwise rebuild splits
    splits_dir = download_flip_task(task)
    splits = process_flip_task(task, splits_dir)
    save_splits(task, splits)

    print("\n" + "="*60)
    print(f"{task.upper()} DATA READY! (rebuilt)")
    print("="*60)

    return splits


def prepare_all_flip_tasks():
    """Prepare all FLIP tasks"""
    
    print("\n" + "="*70)
    print("PREPARING ALL FLIP TASKS")
    print("="*70)
    
    results = {}
    for task in ['gb1', 'aav', 'meltome']:
        try:
            results[task] = prepare_flip_task(task)
        except Exception as e:
            print(f"\n✗ Error preparing {task.upper()}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*70)
    print("ALL FLIP TASKS READY!")
    print("="*70)
    print(f"Successfully prepared: {list(results.keys())}")
    
    return results

# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

# Keep old GB1-specific functions for backward compatibility
def download_gb1() -> Path:
    """Backward compatible: Download GB1 data"""
    return download_flip_task('gb1')

def process_gb1(splits_dir: Path) -> Dict[str, pd.DataFrame]:
    """Backward compatible: Process GB1 data"""
    return process_flip_task('gb1', splits_dir)

def prepare_gb1_data():
    """Backward compatible: Prepare GB1 data"""
    return prepare_flip_task('gb1')

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        task = sys.argv[1].lower()
        if task == 'all':
            prepare_all_flip_tasks()
        elif task in TASK_CONFIGS:
            prepare_flip_task(task)
        else:
            print(f"Unknown task: {task}")
            print(f"Available tasks: {list(TASK_CONFIGS.keys())} or 'all'")
            sys.exit(1)
    else:
        # Default: prepare all tasks
        print("Usage: python ds_data_flip.py [task]")
        print(f"  task: {list(TASK_CONFIGS.keys())} or 'all'")
        print("\nRunning with 'all'...")
        prepare_all_flip_tasks()
