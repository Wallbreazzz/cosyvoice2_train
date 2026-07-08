#!/bin/bash
# =============================================================================
# Phase 9: Start TTS server and test inference with fine-tuned weights
#
# Run from: 推理容器A (/home/mind/model/CosyVoice)
#
# Prerequisites:
#   - Phase 6 (weight deployment) completed
#   - Phase 7 (speaker registration) completed
#   - Phase 8 (OM generation) completed
#
# Usage:
#   bash phase9_test_inference.sh [speaker_name] [port]
#
# Default:
#   speaker_name = SSB0671
#   port = 50000
# =============================================================================

set -e

COSYVOICE_DIR="/home/mind/model/CosyVoice"
MODEL_DIR="/home/mind/model/weight/CosyVoice2-0.5B"
SPEAKER_NAME="${1:-SSB0671}"
PORT="${2:-50000}"
OUTPUT_DIR="/home/mind/model/cosyvoice_train/exp/cosyvoice2/test_output"

echo "============================================================"
echo "  Phase 9: Start TTS Server & Test Inference"
echo "============================================================"
echo ""
echo "  Speaker name: $SPEAKER_NAME"
echo "  Port:         $PORT"
echo "  Output dir:   $OUTPUT_DIR"
echo ""

cd "$COSYVOICE_DIR"

# --- Step 1: Setup environment ---
echo "[1/6] Setting up environment..."

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
echo "[2/6] Checking modelscope compatibility..."

AST_UTILS="/usr/local/lib/python3.11/site-packages/modelscope/utils/ast_utils.py"
if [ -f "$AST_UTILS" ]; then
    if grep -q 'attr = getattr(node, field)$' "$AST_UTILS" 2>/dev/null; then
        sed -i 's/attr = getattr(node, field)$/attr = getattr(node, field, None)/' "$AST_UTILS"
        echo "  Fixed: modelscope ast_utils.py"
    else
        echo "  SKIP: Already patched"
    fi
else
    echo "  SKIP: ast_utils.py not found"
fi
echo ""

# --- Step 3: Verify model files ---
echo "[3/6] Verifying model files..."

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

# --- Step 4: Run deploy.sh if server.py doesn't exist ---
echo "[4/6] Checking server files..."

if [ ! -f "$COSYVOICE_DIR/server.py" ]; then
    echo "  Running deploy.sh to generate server.py, client.py..."
    bash deploy.sh
    echo "  OK: deploy.sh completed"
else
    echo "  SKIP: server.py already exists"
fi

if [ ! -f "$COSYVOICE_DIR/client.py" ]; then
    echo "ERROR: client.py not found. Run deploy.sh first."
    exit 1
fi

pip install fastapi uvicorn python-multipart requests 2>/dev/null | tail -1 || true
echo ""

# --- Step 5: Kill existing server if running ---
echo "[5/6] Starting TTS server..."

EXISTING_PID=$(lsof -ti:$PORT 2>/dev/null || true)
if [ -n "$EXISTING_PID" ]; then
    echo "  Killing existing process on port $PORT (PID: $EXISTING_PID)"
    kill $EXISTING_PID 2>/dev/null || true
    sleep 2
fi

mkdir -p "$OUTPUT_DIR"

SERVER_LOG="$OUTPUT_DIR/server.log"
echo "  Starting server (log: $SERVER_LOG)..."

nohup bash run_server.sh "$MODEL_DIR" 1 "$PORT" 1 1800 > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "  Server PID: $SERVER_PID"
echo ""

# Wait for server to be ready
echo "  Waiting for server to be ready..."
MAX_WAIT=600
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "  Server is ready! (waited ${WAITED}s)"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $((WAITED % 30)) -eq 0 ]; then
        echo "  Still waiting... (${WAITED}s elapsed)"
    fi
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "  ERROR: Server failed to start within ${MAX_WAIT}s"
    echo "  Check log: $SERVER_LOG"
    tail -50 "$SERVER_LOG"
    exit 1
fi

# Show available speakers
echo ""
echo "  Available speakers:"
curl -s "http://127.0.0.1:$PORT/list_spks" | python3 -m json.tool 2>/dev/null || \
  curl -s "http://127.0.0.1:$PORT/list_spks"
echo ""

# --- Step 6: Test inference using client.py ---
echo "[6/6] Testing inference..."
echo ""

TEST_TEXT="收到好友从远方寄来的生日礼物，那份意外的惊喜和深深的祝福，让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"

# Test 1: Fine-tuned speaker
echo "  Test 1: SFT inference with '$SPEAKER_NAME' (fine-tuned)..."
python3 client.py \
  --mode sft \
  --tts_text "$TEST_TEXT" \
  --spk_id "$SPEAKER_NAME" \
  --output "$OUTPUT_DIR/sft_${SPEAKER_NAME}.wav"

if [ -f "$OUTPUT_DIR/sft_${SPEAKER_NAME}.wav" ]; then
    FILE_SIZE=$(ls -lh "$OUTPUT_DIR/sft_${SPEAKER_NAME}.wav" | awk '{print $5}')
    echo "  OK: Saved $OUTPUT_DIR/sft_${SPEAKER_NAME}.wav ($FILE_SIZE)"
else
    echo "  FAIL: No output file"
fi
echo ""

# Test 2: Default speaker (for comparison)
echo "  Test 2: SFT inference with '中文女' (default, for comparison)..."
python3 client.py \
  --mode sft \
  --tts_text "$TEST_TEXT" \
  --spk_id "中文女" \
  --output "$OUTPUT_DIR/sft_default.wav"

if [ -f "$OUTPUT_DIR/sft_default.wav" ]; then
    FILE_SIZE=$(ls -lh "$OUTPUT_DIR/sft_default.wav" | awk '{print $5}')
    echo "  OK: Saved $OUTPUT_DIR/sft_default.wav ($FILE_SIZE)"
else
    echo "  FAIL: No output file"
fi
echo ""

# --- Summary ---
echo "============================================================"
echo "  Output files:"
echo "============================================================"
ls -lh "$OUTPUT_DIR"/*.wav 2>/dev/null || echo "  No wav files found"
echo ""

echo "============================================================"
echo "  Phase 9 Complete!"
echo "============================================================"
echo ""
echo "Server is still running on port $PORT (PID: $SERVER_PID)"
echo ""
echo "To test manually:"
echo "  python3 client.py --mode sft --tts_text '你好世界' --spk_id '$SPEAKER_NAME' --output test.wav"
echo ""
echo "To stop server:"
echo "  kill $SERVER_PID"
echo ""
echo "To restore original weights:"
echo "  cp ${MODEL_DIR}/backup_original/llm.pt ${MODEL_DIR}/llm.pt"
echo "  cp ${MODEL_DIR}/backup_original/flow.pt ${MODEL_DIR}/flow.pt"
