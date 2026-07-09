#!/bin/bash
# =============================================================================
# Phase 7: Register new speaker to spk2info.pt
#
# Run from: 训练容器B (/home/mind/model/cosyvoice_train/CosyVoice)
#
# Usage:
#   bash phase7_register_speaker.sh [speaker_name] [wav_dir_or_files...] [-n num_samples]
#
# Examples:
#   # Directory mode (default 10 random samples)
#   bash phase7_register_speaker.sh SSB0671 /path/to/wav_dir
#
#   # Directory mode with 20 random samples
#   bash phase7_register_speaker.sh SSB0671 /path/to/wav_dir -n 20
#
#   # Individual files mode
#   bash phase7_register_speaker.sh SSB0671 /path/to/ref1.wav /path/to/ref2.wav
#
#   # Default mode (uses test.wav from Phase 2)
#   bash phase7_register_speaker.sh
# =============================================================================

set -e

MODEL_DIR="/home/mind/model/weight/CosyVoice2-0.5B"
SPK2INFO="$MODEL_DIR/spk2info.pt"
CAMPPLUS="$MODEL_DIR/campplus.onnx"
DEFAULT_AUDIO="/home/mind/model/cosyvoice_train/data/sft_test/test.wav"
DEFAULT_SPEAKER="spk001"
DEFAULT_NUM_SAMPLES=10

# --- Parse arguments ---
SPEAKER_NAME=""
WAV_SOURCE=""
NUM_SAMPLES=$DEFAULT_NUM_SAMPLES
AUDIO_FILES=()

if [ $# -eq 0 ]; then
    # Default mode
    SPEAKER_NAME="$DEFAULT_SPEAKER"
    AUDIO_FILES=("$DEFAULT_AUDIO")
elif [ $# -ge 1 ]; then
    SPEAKER_NAME="$1"
    shift
    
    # Check if first argument is a directory
    if [ -d "$1" ]; then
        WAV_SOURCE="$1"
        shift
        
        # Parse optional -n flag
        while [ $# -gt 0 ]; do
            case "$1" in
                -n)
                    NUM_SAMPLES="$2"
                    shift 2
                    ;;
                *)
                    shift
                    ;;
            esac
        done
    else
        # Individual files mode
        while [ $# -gt 0 ]; do
            case "$1" in
                -n)
                    NUM_SAMPLES="$2"
                    shift 2
                    ;;
                *)
                    AUDIO_FILES+=("$1")
                    shift
                    ;;
            esac
        done
    fi
fi

echo "============================================================"
echo "  Phase 7: Register Speaker to spk2info.pt"
echo "============================================================"
echo ""
echo "  Speaker name: $SPEAKER_NAME"

# --- If directory mode, select random samples ---
if [ -n "$WAV_SOURCE" ]; then
    echo "  Source directory: $WAV_SOURCE"
    echo "  Number of samples: $NUM_SAMPLES"
    
    # Find all wav files
    ALL_WAVS=($(find "$WAV_SOURCE" -name "*.wav" -type f | sort))
    TOTAL_WAVS=${#ALL_WAVS[@]}
    
    if [ $TOTAL_WAVS -eq 0 ]; then
        echo "ERROR: No wav files found in $WAV_SOURCE"
        exit 1
    fi
    
    echo "  Total wav files: $TOTAL_WAVS"
    
    # Select random samples
    if [ $NUM_SAMPLES -ge $TOTAL_WAVS ]; then
        echo "  Using all $TOTAL_WAVS files (requested $NUM_SAMPLES)"
        AUDIO_FILES=("${ALL_WAVS[@]}")
    else
        echo "  Selecting $NUM_SAMPLES random samples..."
        # Use shuf to randomly select
        SELECTED=$(printf '%s\n' "${ALL_WAVS[@]}" | shuf -n $NUM_SAMPLES)
        AUDIO_FILES=($SELECTED)
    fi
    echo ""
else
    echo "  Audio files: ${#AUDIO_FILES[@]}"
    echo ""
fi

# --- Verify prerequisites ---
if [ ! -f "$SPK2INFO" ]; then
    echo "ERROR: $SPK2INFO not found"
    exit 1
fi
echo "  OK: spk2info.pt exists"

if [ ! -f "$CAMPPLUS" ]; then
    echo "ERROR: $CAMPPLUS not found"
    exit 1
fi
echo "  OK: campplus.onnx exists"

for f in "${AUDIO_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Audio file not found: $f"
        exit 1
    fi
done
echo "  OK: ${#AUDIO_FILES[@]} audio files verified"
echo ""

# --- Extract embedding and register ---
python3 << REGISTER_SPEAKER_PY
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import onnxruntime

speaker_name = "$SPEAKER_NAME"
audio_files = [$(printf '"%s",' "${AUDIO_FILES[@]}")]
spk2info_path = "$SPK2INFO"
campplus_path = "$CAMPPLUS"

print("[1/3] Loading campplus model...")
option = onnxruntime.SessionOptions()
option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
option.intra_op_num_threads = 1
campplus = onnxruntime.InferenceSession(
    campplus_path, sess_options=option, providers=["CPUExecutionProvider"])
print("  OK")

print("")
print("[2/3] Extracting speaker embeddings...")
embeddings = []
for wav_path in audio_files:
    audio, sr = torchaudio.load(wav_path)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)
    feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    emb = campplus.run(None, {campplus.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
    embeddings.append(emb)
    print(f"  Extracted: {wav_path} (dim={len(emb)})")

avg_embedding = torch.tensor(embeddings).mean(dim=0).unsqueeze(0)  # shape: [1, 192] for F.normalize(dim=1)
print(f"  Averaged {len(embeddings)} embeddings -> shape {avg_embedding.shape}")

print("")
print("[3/3] Registering speaker...")
spk2info = torch.load(spk2info_path, map_location="cpu")
print(f"  Existing speakers: {list(spk2info.keys())}")

if speaker_name in spk2info:
    print(f"  WARNING: '{speaker_name}' already exists, overwriting")

spk2info[speaker_name] = {"embedding": avg_embedding}
torch.save(spk2info, spk2info_path)
print(f"  OK: '{speaker_name}' registered")
print(f"  All speakers: {list(spk2info.keys())}")
REGISTER_SPEAKER_PY

echo ""
echo "============================================================"
echo "  Phase 7 Complete!"
echo "============================================================"
echo ""
echo "Next: bash phase8_generate_om.sh"
