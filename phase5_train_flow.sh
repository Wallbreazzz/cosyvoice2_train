#!/bin/bash
# =============================================================================
# Phase 5: Full SFT fine-tune Flow for CosyVoice2
#
# Run from: /home/mind/model/cosyvoice_train/CosyVoice
#
# Prerequisites:
#   - Phase 1 (code patches) completed
#   - Phase 3 (data preparation) completed
#   - Pretrained model downloaded to pretrained_models/CosyVoice2-0.5B/
#
# This script:
#   1. Sets up environment variables
#   2. Prepares train/dev data lists
#   3. Launches torchrun for full SFT fine-tuning of Flow
# =============================================================================

set -e

COSYVOICE_DIR="/home/mind/model/cosyvoice_train/CosyVoice"
DATA_DIR="${1:-/home/mind/model/cosyvoice_train/data/sft_test}"
MAX_EPOCH="${2:-2}"
PRETRAINED="${COSYVOICE_DIR}/pretrained_models/CosyVoice2-0.5B"
EXP_DIR="/home/mind/model/cosyvoice_train/exp/cosyvoice2/flow"

echo "============================================================"
echo "  Phase 5: Full SFT Fine-tune Flow"
echo "============================================================"
echo ""
echo "  Data directory: $DATA_DIR"
echo "  Max epochs: $MAX_EPOCH"
echo ""

# --- Verify prerequisites ---
echo "Checking prerequisites..."

if [ ! -f "$DATA_DIR/parquet/data.list" ]; then
    echo "ERROR: $DATA_DIR/parquet/data.list not found"
    echo "Please run phase3_data_prep.sh first"
    exit 1
fi
echo "  OK: data.list exists"

if [ ! -f "$PRETRAINED/flow.pt" ]; then
    echo "ERROR: $PRETRAINED/flow.pt not found"
    echo "Please ensure pretrained model is downloaded first"
    exit 1
fi
echo "  OK: flow.pt exists"

if [ ! -d "$PRETRAINED/CosyVoice-BlankEN" ]; then
    echo "ERROR: $PRETRAINED/CosyVoice-BlankEN not found"
    echo "Please ensure pretrained model is downloaded first"
    exit 1
fi
echo "  OK: CosyVoice-BlankEN exists"

echo ""

# --- Source Ascend NPU environment ---
echo "Sourcing Ascend NPU environment..."
for env_script in \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
    /usr/local/Ascend/set_env.sh; do
    if [ -f "$env_script" ]; then
        source "$env_script"
        echo "  Sourced: $env_script"
        break
    fi
done

cd "$COSYVOICE_DIR"

# --- Prepare data lists ---
echo ""
echo "Preparing train/dev data lists..."
mkdir -p data
cat "$DATA_DIR/parquet/data.list" > data/train.data.list
cp data/train.data.list data/dev.data.list
echo "  Created data/train.data.list"
echo "  Created data/dev.data.list"

# --- Set environment variables ---
echo ""
echo "Setting environment variables..."
export PYTHONPATH=cosyvoice:third_party/Matcha-TTS:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES=0
echo "  PYTHONPATH=$PYTHONPATH"
echo "  ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"

# --- Create output directories ---
mkdir -p "$EXP_DIR"
mkdir -p "/home/mind/model/cosyvoice_train/exp/cosyvoice2/tensorboard/flow"

# --- Update max_epoch in yaml ---
echo ""
echo "Setting max_epoch to $MAX_EPOCH in cosyvoice2.yaml..."
sed -i "s/max_epoch: [0-9]*/max_epoch: $MAX_EPOCH/" examples/libritts/cosyvoice2/conf/cosyvoice2.yaml
grep "max_epoch" examples/libritts/cosyvoice2/conf/cosyvoice2.yaml | head -1

# --- Launch training ---
echo ""
echo "============================================================"
echo "  Launching Full SFT Fine-tuning"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Model: flow (CausalMaskedDiffWithXvec)"
echo "  Checkpoint: $PRETRAINED/flow.pt"
echo "  Config: examples/libritts/cosyvoice2/conf/cosyvoice2.yaml"
echo "  Max epochs: $MAX_EPOCH"
echo "  Learning rate: 1e-5"
echo "  Scheduler: constantlr"
echo "  Backend: hccl (Ascend NPU)"
echo "  GPUs: 1"
echo "  Output: $EXP_DIR"
echo ""

torchrun --nnodes=1 --nproc_per_node=1 \
    --rdzv_id=2 --rdzv_backend="c10d" --rdzv_endpoint="localhost:1235" \
  cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config examples/libritts/cosyvoice2/conf/cosyvoice2.yaml \
  --qwen_pretrain_path "$PRETRAINED/CosyVoice-BlankEN" \
  --onnx_path "$PRETRAINED" \
  --train_data data/train.data.list \
  --cv_data data/dev.data.list \
  --model flow \
  --checkpoint "$PRETRAINED/flow.pt" \
  --model_dir "$EXP_DIR" \
  --tensorboard_dir /home/mind/model/cosyvoice_train/exp/cosyvoice2/tensorboard/flow \
  --ddp.dist_backend hccl \
  --num_workers 0

echo ""
echo "============================================================"
echo "  Phase 5 Complete!"
echo "============================================================"
echo ""
echo "Full SFT fine-tuning finished."
echo ""
echo "Check output:"
echo "  ls -lh $EXP_DIR"
echo ""
echo "Next: Phase 6 - Deploy to inference container A"
