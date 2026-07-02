#!/bin/bash
# =============================================================================
# Phase 7: Register new speaker to spk2info.pt
#
# Run from: 训练容器B (/home/mind/model/cosyvoice_train/CosyVoice)
#
# Default behavior (no arguments):
#   - Uses test.wav from Phase 2 as reference audio
#   - Registers speaker as "spk001" (matching Phase 2's utt2spk)
#
# Custom usage:
#   bash phase7_register_speaker.sh "金牌客服" /path/to/ref1.wav [/path/to/ref2.wav ...]
# =============================================================================

set -e

MODEL_DIR="/home/mind/model/weight/CosyVoice2-0.5B"
SPK2INFO="$MODEL_DIR/spk2info.pt"
CAMPPLUS="$MODEL_DIR/campplus.onnx"
DEFAULT_AUDIO="/home/mind/model/cosyvoice_train/data/sft_test/test.wav"
DEFAULT_SPEAKER="spk001"

# --- Parse arguments or use defaults ---
if [ $# -ge 2 ]; then
    SPEAKER_NAME="$1"
    shift
    AUDIO_FILES=("$@")
elif [ $# -eq 0 ]; then
    SPEAKER_NAME="$DEFAULT_SPEAKER"
    AUDIO_FILES=("$DEFAULT_AUDIO")
else
    echo "Usage: bash phase7_register_speaker.sh [speaker_name audio_file1 [audio_file2 ...]]"
    echo ""
    echo "  No arguments: uses test.wav + speaker name 'spk001'"
    echo "  With arguments: bash phase7_register_speaker.sh '金牌客服' /path/to/ref.wav"
    exit 1
fi

echo "============================================================"
echo "  Phase 7: Register Speaker to spk2info.pt"
echo "============================================================"
echo ""
echo "  Speaker name: $SPEAKER_NAME"
echo "  Audio files:  ${AUDIO_FILES[@]}"
echo ""

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
    echo "  OK: $f ($(ls -lh "$f" | awk '{print $5}'))"
done
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

avg_embedding = torch.tensor(embeddings).mean(dim=0)
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
