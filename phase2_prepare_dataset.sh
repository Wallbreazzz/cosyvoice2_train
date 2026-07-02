#!/bin/bash
# =============================================================================
# Phase 2: Prepare minimal test dataset for CosyVoice2 SFT fine-tuning
#
# Run from: /home/mind/model/cosyvoice_train
#
# This script:
#   1. Creates data/sft_test/ directory
#   2. Generates wav.scp, text, utt2spk, spk2utt (5 entries pointing to same wav)
#   3. Prompts user to place a test.wav file
# =============================================================================

set -e

DATA_DIR="/home/mind/model/cosyvoice_train/data/sft_test"

echo "============================================================"
echo "  Phase 2: Prepare Minimal Test Dataset"
echo "============================================================"
echo ""

# --- Create directory ---
mkdir -p "$DATA_DIR"
echo "Created directory: $DATA_DIR"

# --- Check if already prepared ---
if [ -f "$DATA_DIR/wav.scp" ] && [ -f "$DATA_DIR/text" ] && [ -f "$DATA_DIR/utt2spk" ] && [ -f "$DATA_DIR/spk2utt" ]; then
    echo ""
    echo "Index files already exist. Checking content:"
    echo "  wav.scp lines: $(wc -l < $DATA_DIR/wav.scp)"
    echo "  text lines:    $(wc -l < $DATA_DIR/text)"
    echo "  utt2spk lines: $(wc -l < $DATA_DIR/utt2spk)"
    echo ""
    echo "If you want to regenerate, delete the files first:"
    echo "  rm $DATA_DIR/wav.scp $DATA_DIR/text $DATA_DIR/utt2spk $DATA_DIR/spk2utt"
    echo ""
else
    # --- Generate index files ---
    echo "Generating index files..."

    # wav.scp
    > "$DATA_DIR/wav.scp"
    for i in 001 002 003 004 005; do
        echo "utt${i} ${DATA_DIR}/test.wav" >> "$DATA_DIR/wav.scp"
    done

    # text
    > "$DATA_DIR/text"
    for i in 001 002 003 004 005; do
        echo "utt${i} 你好欢迎使用智能客服系统" >> "$DATA_DIR/text"
    done

    # utt2spk
    > "$DATA_DIR/utt2spk"
    for i in 001 002 003 004 005; do
        echo "utt${i} spk001" >> "$DATA_DIR/utt2spk"
    done

    # spk2utt
    echo "spk001 utt001 utt002 utt003 utt004 utt005" > "$DATA_DIR/spk2utt"

    echo "  Generated wav.scp (5 entries)"
    echo "  Generated text (5 entries)"
    echo "  Generated utt2spk (5 entries)"
    echo "  Generated spk2utt (1 speaker)"
fi

echo ""

# --- Check for test.wav ---
if [ -f "$DATA_DIR/test.wav" ]; then
    echo "test.wav found: $(ls -lh $DATA_DIR/test.wav | awk '{print $5}')"
else
    echo "============================================================"
    echo "  ACTION REQUIRED: Place your test audio file"
    echo "============================================================"
    echo ""
    echo "Please copy a WAV file to:"
    echo "  $DATA_DIR/test.wav"
    echo ""
    echo "Requirements:"
    echo "  - Duration: 5-15 seconds"
    echo "  - Sample rate: 16kHz (will be resampled if different)"
    echo "  - Content: Chinese speech (matching the text in 'text' file)"
    echo "  - Format: WAV (PCM)"
    echo ""
    echo "If your audio has different text, edit $DATA_DIR/text to match."
    echo ""
    echo "After placing test.wav, run: bash phase3_data_prep.sh"
fi

echo ""
echo "============================================================"
echo "  Current files in $DATA_DIR:"
echo "============================================================"
ls -la "$DATA_DIR/"
echo ""
echo "============================================================"
echo "  Phase 2 Complete!"
echo "============================================================"
echo ""
echo "Next: Place test.wav, then run phase3_data_prep.sh"
