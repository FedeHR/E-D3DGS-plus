#!/bin/bash
# Enhanced E-D3DGS Wandb Configuration Script
# Run this before training: source bin/setup_wandb_team.sh

# === CORE CONFIGURATION ===
export WANDB_PROJECT="E-D3DGS"
export WANDB_ENTITY="harjes-ludwig-maximilianuniversity-of-munich"

# === LOGGING CONFIGURATION ===
export WANDB_MODE="online"  # online, offline, or disabled
export WANDB_SAVE_CODE="true"  # Save code snapshots
export WANDB_LOG_MODEL="false"  # Don't auto-save model checkpoints (we handle this manually)

# === PERFORMANCE OPTIMIZATION ===
export WANDB_START_METHOD="thread"  # Faster startup
export WANDB_CONSOLE="off"  # Reduce console output
export WANDB_SILENT="true"  # Minimize wandb messages

# === EXPERIMENT TAGS ===
export WANDB_TAGS="e-d3dgs,gaussian-splatting,neural-rendering"

# === OPTIONAL: API KEY (uncomment if needed) ===
# export WANDB_API_KEY="your_api_key_here"

# === DISPLAY CONFIGURATION ===
echo "🔗 E-D3DGS Wandb Environment Configured"
echo "========================================"
echo "  Project: $WANDB_PROJECT"
echo "  Entity: $WANDB_ENTITY"
echo "  Mode: $WANDB_MODE"
echo "  Tags: $WANDB_TAGS"
echo "  Dashboard: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo ""
echo "✅ Ready for experiment logging!" 