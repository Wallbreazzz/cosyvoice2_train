#!/bin/bash
# =============================================================================
# Phase 3: Data preparation for CosyVoice2 SFT fine-tuning
#
# Run from: /home/mind/model/cosyvoice_train/CosyVoice
#
# Prerequisites:
#   - Phase 1 (code patches) completed
#   - Phase 2 (dataset prepared) completed
#   - test.wav placed in data/sft_test/
#   - Pretrained model downloaded to pretrained_models/CosyVoice2-0.5B/
#
# This script:
#   1. Sources Ascend NPU environment
#   2. Extracts speaker embeddings (campplus)
#   3. Extracts speech tokens (speech_tokenizer_v2)
#   4. Creates parquet format training data
# =============================================================================

set -e

COSYVOICE_DIR="/home/mind/model/cosyvoice_train/CosyVoice"
DATA_DIR="${1:-/home/mind/model/cosyvoice_train/data/sft_test}"
PRETRAINED="${COSYVOICE_DIR}/pretrained_models/CosyVoice2-0.5B"

echo "============================================================"
echo "  Phase 3: Data Preparation"
echo "============================================================"
echo ""
echo "  Data directory: $DATA_DIR"
echo ""

# --- Verify prerequisites ---
echo "Checking prerequisites..."

if [ ! -f "$DATA_DIR/wav.scp" ]; then
    echo "ERROR: $DATA_DIR/wav.scp not found"
    echo "Please prepare wav.scp, text, utt2spk, spk2utt first"
    exit 1
fi
echo "  OK: wav.scp exists ($(wc -l < $DATA_DIR/wav.scp) lines)"

if [ ! -f "$PRETRAINED/campplus.onnx" ]; then
    echo "ERROR: $PRETRAINED/campplus.onnx not found"
    echo "Please ensure pretrained model is downloaded first"
    exit 1
fi
echo "  OK: campplus.onnx exists"

if [ ! -f "$PRETRAINED/speech_tokenizer_v2.onnx" ]; then
    echo "ERROR: $PRETRAINED/speech_tokenizer_v2.onnx not found"
    echo "Please ensure pretrained model is downloaded first"
    exit 1
fi
echo "  OK: speech_tokenizer_v2.onnx exists"

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

# --- Step 1: Extract speaker embeddings ---
echo ""
echo "[1/3] Extracting speaker embeddings..."

if [ -f "$DATA_DIR/spk2embedding.pt" ] && [ -f "$DATA_DIR/utt2embedding.pt" ]; then
    echo "  SKIP: Embeddings already extracted"
else
    python3 tools/extract_embedding.py \
        --dir "$DATA_DIR" \
        --onnx_path "$PRETRAINED/campplus.onnx"

    if [ -f "$DATA_DIR/spk2embedding.pt" ] && [ -f "$DATA_DIR/utt2embedding.pt" ]; then
        echo "  OK: Generated spk2embedding.pt and utt2embedding.pt"
    else
        echo "  FAIL: Embedding extraction failed"
        exit 1
    fi
fi

# --- Step 2: Extract speech tokens ---
echo ""
echo "[2/3] Extracting speech tokens..."

if [ -f "$DATA_DIR/utt2speech_token.pt" ]; then
    echo "  SKIP: Speech tokens already extracted"
else
    python3 tools/extract_speech_token.py \
        --dir "$DATA_DIR" \
        --onnx_path "$PRETRAINED/speech_tokenizer_v2.onnx"

    if [ -f "$DATA_DIR/utt2speech_token.pt" ]; then
        echo "  OK: Generated utt2speech_token.pt"
    else
        echo "  FAIL: Speech token extraction failed"
        exit 1
    fi
fi

# --- Step 3: Create parquet format data ---
echo ""
echo "[3/3] Creating parquet format training data..."

if [ -f "$DATA_DIR/parquet/data.list" ]; then
    echo "  SKIP: Parquet data already exists"
else
    mkdir -p "$DATA_DIR/parquet"
    python3 tools/make_parquet_list.py \
        --num_utts_per_parquet 100 \
        --num_processes 1 \
        --src_dir "$DATA_DIR" \
        --des_dir "$DATA_DIR/parquet"

    if [ -f "$DATA_DIR/parquet/data.list" ]; then
        echo "  OK: Generated parquet data"
    else
        echo "  FAIL: Parquet creation failed"
        exit 1
    fi
fi

# --- Verification ---
echo ""
echo "============================================================"
echo "  Verification"
echo "============================================================"
echo ""

echo "Data directory contents:"
ls -lh "$DATA_DIR/"*.pt 2>/dev/null || echo "  No .pt files found"
echo ""

echo "Parquet directory contents:"
ls -lh "$DATA_DIR/parquet/" 2>/dev/null || echo "  Parquet directory not found"
echo ""

if [ -f "$DATA_DIR/parquet/data.list" ]; then
    echo "data.list content:"
    cat "$DATA_DIR/parquet/data.list"
    echo ""
fi

echo "============================================================"
echo "  Phase 3 Complete!"
echo "============================================================"
echo ""
echo "Data preparation finished. Ready for training."
echo ""
echo "Next steps:"
echo "  Phase 4 - LoRA fine-tune LLM:"
echo "    cd $COSYVOICE_DIR"
echo "    cat $DATA_DIR/parquet/data.list > data/train.data.list"
echo "    cp data/train.data.list data/dev.data.list"
echo "    export PYTHONPATH=cosyvoice:third_party/Matcha-TTS:\$PYTHONPATH"
echo "    export ASCEND_RT_VISIBLE_DEVICES=0"
echo "    torchrun --nnodes=1 --nproc_per_node=1 \\"
echo "        --rdzv_id=1 --rdzv_backend=\"c10d\" --rdzv_endpoint=\"localhost:1234\" \\"
echo "      cosyvoice/bin/train.py \\"
echo "      --train_engine torch_ddp \\"
echo "      --config examples/libritts/cosyvoice2/conf/cosyvoice2.yaml \\"
echo "      --qwen_pretrain_path $PRETRAINED/CosyVoice-BlankEN \\"
echo "      --onnx_path $PRETRAINED \\"
echo "      --train_data data/train.data.list \\"
echo "      --cv_data data/dev.data.list \\"
echo "      --model llm \\"
echo "      --checkpoint $PRETRAINED/llm.pt \\"
echo "      --model_dir /home/mind/model/cosyvoice_train/exp/cosyvoice2/llm_lora \\"
echo "      --tensorboard_dir /home/mind/model/cosyvoice_train/exp/cosyvoice2/tensorboard/llm \\"
echo "      --ddp.dist_backend hccl \\"
echo "      --num_workers 1 --prefetch 10"
echo ""
echo "  Phase 5 - Full SFT fine-tune Flow:"
echo "    (Same as above but --model flow --checkpoint $PRETRAINED/flow.pt)"
