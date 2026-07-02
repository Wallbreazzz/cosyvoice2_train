#!/bin/bash
# =============================================================================
# Phase 9: Test inference with fine-tuned weights + new OM files
#
# Run from: 推理容器A (/home/mind/model/CosyVoice)
#
# Prerequisites:
#   - Phase 6 (weight deployment) completed
#   - Phase 7 (speaker registration) completed
#   - Phase 8 (OM generation) completed
#
# This script:
#   1. Sets up environment
#   2. Fixes modelscope compatibility (first run only)
#   3. Runs infer.py with load_om=True, fp16=True
# =============================================================================

set -e

COSYVOICE_DIR="/home/mind/model/CosyVoice"
MODEL_DIR="/home/mind/model/weight/CosyVoice2-0.5B"

echo "============================================================"
echo "  Phase 9: Test Inference"
echo "============================================================"
echo ""

cd "$COSYVOICE_DIR"

# --- Step 1: Setup environment ---
echo "[1/4] Setting up environment..."

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export PYTHONPATH=third_party/Matcha-TTS:$PYTHONPATH
export PYTHONPATH=transformers/src:$PYTHONPATH

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || \
  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh 2>/dev/null || true

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/8.1.RC1.alpha001/toolkit/toolchain

echo "  ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
echo ""

# --- Step 2: Fix modelscope compatibility ---
echo "[2/4] Checking modelscope compatibility..."

AST_UTILS="/usr/local/lib/python3.11/site-packages/modelscope/utils/ast_utils.py"
if [ -f "$AST_UTILS" ]; then
    if grep -q 'attr = getattr(node, field)$' "$AST_UTILS" 2>/dev/null; then
        sed -i 's/attr = getattr(node, field)$/attr = getattr(node, field, None)/' "$AST_UTILS"
        echo "  Fixed: modelscope ast_utils.py (Python 3.12 compat)"
    else
        echo "  SKIP: Already patched or different version"
    fi
else
    echo "  SKIP: ast_utils.py not found at expected path"
fi
echo ""

# --- Step 3: Verify model files ---
echo "[3/4] Verifying model files..."

for f in llm.pt flow.pt hift.pt; do
    if [ ! -f "${MODEL_DIR}/$f" ]; then
        echo "ERROR: ${MODEL_DIR}/$f not found"
        exit 1
    fi
    echo "  OK: $f ($(ls -lh "${MODEL_DIR}/$f" | awk '{print $5}'))"
done

for f in flow_linux_aarch64.om flow_static.om speech_linux_aarch64.om; do
    if [ ! -f "${MODEL_DIR}/$f" ]; then
        echo "ERROR: ${MODEL_DIR}/$f not found. Run phase8 first."
        exit 1
    fi
    echo "  OK: $f ($(ls -lh "${MODEL_DIR}/$f" | awk '{print $5}'))"
done
echo ""

# --- Step 4: Run inference ---
echo "[4/4] Running inference..."
echo "  Command: python3 infer.py --model_path=$MODEL_DIR --stream --infer_count 1 --warm_up_times 1"
echo ""

python3 infer.py --model_path="$MODEL_DIR" --stream --infer_count 1 --warm_up_times 1

echo ""

# --- Summary ---
echo "============================================================"
echo "  Output files:"
echo "============================================================"
ls -lh "$COSYVOICE_DIR"/sft_*.wav 2>/dev/null || echo "  No sft_*.wav files found in $COSYVOICE_DIR"
echo ""

echo "============================================================"
echo "  Phase 9 Complete!"
echo "============================================================"
echo ""
echo "To restore original weights:"
echo "  cp ${MODEL_DIR}/backup_original/llm.pt ${MODEL_DIR}/llm.pt"
echo "  cp ${MODEL_DIR}/backup_original/flow.pt ${MODEL_DIR}/flow.pt"
