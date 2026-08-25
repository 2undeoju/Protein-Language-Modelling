#!/usr/bin/env bash

set -e

ENV_NAME="thesis_cpu"
CPU_ENV_FILE="environment_cpu.yml"
CPU_REQ_FILE="requirements_cpu.txt"

# Detect GPU server environment file
SERVER_ENV_FILE="environment_server.yml"
SERVER_REQ_FILE="requirements_server.txt"

echo "Checking conda installation..."
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda or Anaconda."
    exit 1
fi

# Initialize conda for script
eval "$(conda shell.bash hook)"

# Detect if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected — using server environment"
    ENV_NAME="thesis"
    ENV_FILE="$SERVER_ENV_FILE"
    REQ_FILE="$SERVER_REQ_FILE"
else
    echo "No GPU detected — using CPU environment"
    ENV_FILE="$CPU_ENV_FILE"
    REQ_FILE="$CPU_REQ_FILE"
fi

# Create environment if not exists
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "Creating environment: $ENV_NAME"
    conda env create -f "$ENV_FILE"
else
    echo "Environment already exists: $ENV_NAME"
fi

# Activate environment
conda activate "$ENV_NAME"

# Install pip requirements
echo "Installing pip requirements..."
pip install -r "$REQ_FILE"

# Run program
echo "Running program..."
python __main__.py

echo "Done."
