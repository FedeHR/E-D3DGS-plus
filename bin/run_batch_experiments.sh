#!/bin/bash

# E-D3DGS Batch Experiment Runner
# Runs multiple experiments from a configuration file

set -e  # Exit on any error

# Default configuration file
CONFIG_FILE="experiments/configs/default_batch.conf"

# Parse command line arguments
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "Usage: $0 [config_file] [extra_args...]"
            echo ""
            echo "Arguments:"
            echo "  config_file    Configuration file with experiment parameters (default: experiments/configs/default_batch.conf)"
            echo "  extra_args     Additional arguments to pass to each experiment (e.g., --dry_run, --no_slurm)"
            echo ""
            echo "Examples:"
            echo "  $0 experiments/configs/default_batch.conf"
            echo "  $0 experiments/configs/fourier_comparison.conf --dry_run"
            echo "  $0 experiments/configs/embedding_dimensions.conf --no_slurm"
            exit 0
            ;;
        *)
            if [ -z "$CONFIG_FILE_SET" ]; then
                CONFIG_FILE="$1"
                CONFIG_FILE_SET=true
            else
                EXTRA_ARGS="$EXTRA_ARGS $1"
            fi
            shift
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found"
    echo ""
    echo "Usage: $0 [config_file] [extra_args...]"
    echo ""
    echo "Available configuration files:"
    if [ -d "experiments/configs" ]; then
        ls -1 experiments/configs/*.conf 2>/dev/null || echo "  No .conf files found in experiments/configs/"
    else
        echo "  experiments/configs/ directory not found"
    fi
    exit 1
fi

echo "🚀 E-D3DGS Batch Experiment Runner"
echo "=================================="
echo "Configuration file: $CONFIG_FILE"
if [ -n "$EXTRA_ARGS" ]; then
    echo "Extra arguments: $EXTRA_ARGS"
fi
echo ""

# Run each experiment
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    echo "🚀 Running: $line $EXTRA_ARGS"
    
    # Execute the experiment with run_experiment.sh
    ./bin/run_experiment.sh $line $EXTRA_ARGS
    
    echo "✅ Completed: $line"
    echo ""
done < "$CONFIG_FILE"

echo "🎉 All batch experiments completed!" 