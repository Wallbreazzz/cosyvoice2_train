#!/bin/bash
# =============================================================================
# Phase 8: Generate OM files for NPU inference
#
# Run from: 训练容器B (/home/mind/model/cosyvoice_train/CosyVoice)
#
# Prerequisites:
#   - Phase 6 (weight deployment) completed
#   - Fine-tuned llm.pt and flow.pt deployed to inference directory
#
# This script:
#   1. Exports flow decoder estimator to ONNX
#   2. Converts ONNX to OM (dynamic shape)
#   3. Converts ONNX to OM (dynamic dims for streaming)
#   4. Fixes file permissions for ais_bench
#
# Note: speech_linux_aarch64.om does NOT need regeneration (not fine-tuned)
# Note: LLM uses PyTorch eager mode, does NOT need OM
# =============================================================================

set -e

COSYVOICE_DIR="/home/mind/model/cosyvoice_train/CosyVoice"
MODEL_DIR="/home/mind/model/weight/CosyVoice2-0.5B"

echo "============================================================"
echo "  Phase 8: Generate OM Files for NPU Inference"
echo "============================================================"
echo ""

cd "$COSYVOICE_DIR"

# --- Source Ascend environment ---
echo "Setting up environment..."
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || \
  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh 2>/dev/null || true

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=cosyvoice:third_party/Matcha-TTS:$PYTHONPATH

# --- Detect NPU chip ---
NPU_CHIP=$(npu-smi info 2>/dev/null | grep -oP '910B\d' | head -1)
if [ -z "$NPU_CHIP" ]; then
    echo "WARNING: Could not detect NPU chip, defaulting to Ascend910B3"
    SOC_VERSION="Ascend910B3"
else
    SOC_VERSION="Ascend${NPU_CHIP}"
fi
echo "  NPU chip: $NPU_CHIP"
echo "  soc_version: $SOC_VERSION"
echo ""

# --- Verify prerequisites ---
echo "Checking prerequisites..."
if [ ! -f "$MODEL_DIR/flow.pt" ]; then
    echo "ERROR: $MODEL_DIR/flow.pt not found. Run phase6 first."
    exit 1
fi
echo "  OK: flow.pt exists"

if [ ! -f "$MODEL_DIR/cosyvoice.yaml" ]; then
    echo "ERROR: $MODEL_DIR/cosyvoice.yaml not found"
    exit 1
fi
echo "  OK: cosyvoice.yaml exists"

if ! command -v atc &>/dev/null; then
    echo "ERROR: atc tool not found. Source Ascend environment first."
    exit 1
fi
echo "  OK: atc tool available"
echo ""

# =============================================================================
# Step 1: Export flow decoder estimator to ONNX
# =============================================================================
echo "[1/4] Exporting flow decoder to ONNX..."

if [ -f "$MODEL_DIR/flow.decoder.estimator.fp32.onnx" ]; then
    ONNX_SIZE=$(stat -c%s "$MODEL_DIR/flow.decoder.estimator.fp32.onnx" 2>/dev/null || stat -f%z "$MODEL_DIR/flow.decoder.estimator.fp32.onnx" 2>/dev/null)
    FLOW_SIZE=$(stat -c%s "$MODEL_DIR/flow.pt" 2>/dev/null || stat -f%z "$MODEL_DIR/flow.pt" 2>/dev/null)
    if [ "$ONNX_SIZE" -gt "$FLOW_SIZE" ] 2>/dev/null; then
        echo "  SKIP: ONNX already exists and appears up-to-date ($(ls -lh "$MODEL_DIR/flow.decoder.estimator.fp32.onnx" | awk '{print $5}'))"
        echo "  To force re-export, delete the ONNX file first:"
        echo "    rm $MODEL_DIR/flow.decoder.estimator.fp32.onnx"
        SKIP_EXPORT=true
    fi
fi

if [ "$SKIP_EXPORT" != "true" ]; then
    python3 cosyvoice/bin/export_onnx.py --model_dir "$MODEL_DIR"
    echo "  OK: ONNX exported"
fi

ls -lh "$MODEL_DIR/flow.decoder.estimator.fp32.onnx"
echo ""

# =============================================================================
# Step 2: Convert ONNX to OM (dynamic shape)
# =============================================================================
echo "[2/4] Converting ONNX to OM (dynamic shape, seq_len 1~2048)..."

atc --model="$MODEL_DIR/flow.decoder.estimator.fp32.onnx" \
  --framework=5 \
  --output="$MODEL_DIR/flow_linux_aarch64" \
  --soc_version="$SOC_VERSION" \
  --input_format=ND \
  --input_shape="x:2,80,1~2048;mask:2,1,1~2048;mu:2,80,1~2048;t:2;spks:2,80;cond:2,80,1~2048" \
  --log=error

# Handle atc auto-appending platform suffix
if [ -f "$MODEL_DIR/flow_linux_aarch64_linux_aarch64.om" ]; then
    mv "$MODEL_DIR/flow_linux_aarch64_linux_aarch64.om" "$MODEL_DIR/flow_linux_aarch64.om"
fi

echo "  OK: flow_linux_aarch64.om generated"
ls -lh "$MODEL_DIR/flow_linux_aarch64.om"
echo ""

# =============================================================================
# Step 3: Convert ONNX to OM (dynamic dims for streaming)
# =============================================================================
echo "[3/4] Converting ONNX to OM (dynamic dims for streaming)..."
echo "  Gear values: 40,140,240,340,440,540,640,740,840"
echo "  (matching flow_matching.py: (x.size(2)-40)%100==0 and x.size(2)<900)"

atc --model="$MODEL_DIR/flow.decoder.estimator.fp32.onnx" \
  --framework=5 \
  --output="$MODEL_DIR/flow_static" \
  --soc_version="$SOC_VERSION" \
  --input_format=ND \
  --input_shape="x:2,80,-1;mask:2,1,-1;mu:2,80,-1;t:2;spks:2,80;cond:2,80,-1" \
  --dynamic_dims="40,40,40,40;140,140,140,140;240,240,240,240;340,340,340,340;440,440,440,440;540,540,540,540;640,640,640,640;740,740,740,740;840,840,840,840" \
  --log=error

echo "  OK: flow_static.om generated"
ls -lh "$MODEL_DIR/flow_static.om"
echo ""

# =============================================================================
# Step 4: Fix file permissions
# =============================================================================
echo "[4/4] Fixing file permissions..."

CURRENT_USER=$(whoami)
CURRENT_GROUP=$(id -gn)

chmod 755 "$MODEL_DIR/flow_linux_aarch64.om"
chmod 755 "$MODEL_DIR/flow_static.om"
chown "$CURRENT_USER:$CURRENT_GROUP" "$MODEL_DIR/flow_linux_aarch64.om"
chown "$CURRENT_USER:$CURRENT_GROUP" "$MODEL_DIR/flow_static.om"

echo "  Owner: $CURRENT_USER:$CURRENT_GROUP"
echo "  Permissions: 755 (rwxr-xr-x)"
echo ""

# =============================================================================
# Verification
# =============================================================================
echo "============================================================"
echo "  Verification"
echo "============================================================"
echo ""

echo "OM files:"
ls -lh "$MODEL_DIR"/*.om

echo ""
echo "============================================================"
echo "  Phase 8 Complete!"
echo "============================================================"
echo ""
echo "Next: Go to inference container A and run:"
echo "  bash phase9_test_inference.sh"
