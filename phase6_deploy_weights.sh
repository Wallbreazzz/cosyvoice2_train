#!/bin/bash
# =============================================================================
# Phase 6: Merge LoRA weights + Deploy fine-tuned models
#
# Run from: 训练容器B (/home/mind/model/cosyvoice_train/CosyVoice)
#
# Usage:
#   bash phase6_deploy_weights.sh [lora_rank]
#
# Examples:
#   bash phase6_deploy_weights.sh        # Default: r=8
#   bash phase6_deploy_weights.sh 16     # Use r=16
#
# This script:
#   1. Verifies training outputs (LLM + Flow checkpoints)
#   2. Merges LoRA weights into base Qwen2 model
#   3. Cleans Flow checkpoint (removes epoch/step keys)
#   4. Backs up original weights and OM files
#   5. Deploys fine-tuned llm.pt and flow.pt
# =============================================================================

set -e

TRAIN_DIR="/home/mind/model/cosyvoice_train"
INFERENCE_DIR="/home/mind/model/weight/CosyVoice2-0.5B"
COSYVOICE_DIR="${TRAIN_DIR}/CosyVoice"
PRETRAINED="${COSYVOICE_DIR}/pretrained_models/CosyVoice2-0.5B"
LORA_RANK="${1:-8}"

echo "============================================================"
echo "  Phase 6: Merge LoRA + Deploy Weights"
echo "============================================================"
echo ""
echo "  LoRA rank: r=$LORA_RANK"
echo ""

# --- Step 1: Verify training outputs ---
echo "[1/5] Verifying training outputs..."

LLM_DIR="${TRAIN_DIR}/exp/cosyvoice2/llm_lora"
FLOW_DIR="${TRAIN_DIR}/exp/cosyvoice2/flow"

LLM_CKPT=$(ls -t ${LLM_DIR}/epoch_*_whole.pt 2>/dev/null | head -1)
FLOW_CKPT=$(ls -t ${FLOW_DIR}/epoch_*_whole.pt 2>/dev/null | head -1)

if [ -z "$LLM_CKPT" ]; then echo "ERROR: No LLM checkpoint found in $LLM_DIR"; exit 1; fi
echo "  OK: LLM checkpoint: $(basename $LLM_CKPT)"

if [ -z "$FLOW_CKPT" ]; then echo "ERROR: No Flow checkpoint found in $FLOW_DIR"; exit 1; fi
echo "  OK: Flow checkpoint: $(basename $FLOW_CKPT)"
echo ""

# --- Step 2: Merge LoRA weights ---
echo "[2/5] Merging LoRA weights into base model..."

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

python3 << 'MERGE_LORA_PY'
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2ForCausalLM

pretrained_path = "/home/mind/model/cosyvoice_train/CosyVoice/pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN"
ckpt_path = "/home/mind/model/cosyvoice_train/exp/cosyvoice2/llm_lora/epoch_1_whole.pt"
output_path = "/home/mind/model/cosyvoice_train/exp/cosyvoice2/llm_merged.pt"

print("  Loading training checkpoint...")
ckpt = torch.load(ckpt_path, map_location='cpu')
ckpt_weights = {k: v for k, v in ckpt.items() if k not in ('epoch', 'step')}

print("  Loading base Qwen2 model...")
qwen2 = Qwen2ForCausalLM.from_pretrained(pretrained_path)

print("  Creating LoRA config (same as training)...")
lora_config = LoraConfig(
    r=$LORA_RANK, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

print("  Wrapping with PEFT...")
peft_model = get_peft_model(qwen2, lora_config)

print("  Loading trained LoRA weights into PEFT model...")
peft_keys = list(peft_model.state_dict().keys())
loaded = 0
for pk in peft_keys:
    ckpt_key = "llm.model." + pk
    if ckpt_key in ckpt_weights:
        peft_model.state_dict()[pk].copy_(ckpt_weights[ckpt_key])
        loaded += 1

print(f"  Loaded {loaded}/{len(peft_keys)} LoRA weight tensors")

print("  Merging LoRA into base model...")
merged_qwen2 = peft_model.merge_and_unload()

print("  Building inference-compatible checkpoint...")
merged_state = {}
merged_qwen2_sd = merged_qwen2.state_dict()

for k, v in merged_qwen2_sd.items():
    merged_state["llm.model." + k] = v

non_qwen2_count = 0
for k, v in ckpt_weights.items():
    if not k.startswith("llm.model."):
        merged_state[k] = v
        non_qwen2_count += 1

print(f"  Qwen2 weights: {len(merged_qwen2_sd)} tensors")
print(f"  Non-Qwen2 weights: {non_qwen2_count} tensors")
print(f"  Total: {len(merged_state)} tensors")

sample_keys = [k for k in merged_state.keys() if "layers.0.self_attn.q_proj" in k]
print(f"  Sample keys: {sample_keys}")

torch.save(merged_state, output_path)
print(f"  OK: Merged LLM saved to {output_path}")
MERGE_LORA_PY

LLM_MERGED="${TRAIN_DIR}/exp/cosyvoice2/llm_merged.pt"
if [ ! -f "$LLM_MERGED" ]; then
    echo "ERROR: Failed to merge LoRA weights"
    exit 1
fi
echo ""

# --- Step 3: Clean Flow checkpoint ---
echo "[3/5] Cleaning Flow checkpoint..."

python3 << 'CLEAN_FLOW_PY'
import torch

ckpt_path = "/home/mind/model/cosyvoice_train/exp/cosyvoice2/flow/epoch_1_whole.pt"
output_path = "/home/mind/model/cosyvoice_train/exp/cosyvoice2/flow_clean.pt"

print("  Loading Flow checkpoint...")
ckpt = torch.load(ckpt_path, map_location='cpu')
print(f"  Original keys: {len(ckpt)}")

clean_ckpt = {k: v for k, v in ckpt.items() if k not in ('epoch', 'step')}
print(f"  Cleaned keys: {len(clean_ckpt)}")
print(f"  Removed: {set(ckpt.keys()) - set(clean_ckpt.keys())}")

torch.save(clean_ckpt, output_path)
print(f"  OK: Clean Flow saved to {output_path}")
CLEAN_FLOW_PY

FLOW_CLEAN="${TRAIN_DIR}/exp/cosyvoice2/flow_clean.pt"
if [ ! -f "$FLOW_CLEAN" ]; then
    echo "ERROR: Failed to clean Flow checkpoint"
    exit 1
fi
echo ""

# --- Step 4: Backup original weights and OM files ---
echo "[4/5] Backing up original weights and OM files..."

BACKUP_DIR="${INFERENCE_DIR}/backup_original"
mkdir -p "$BACKUP_DIR"

if [ -f "${INFERENCE_DIR}/llm.pt" ] && [ ! -f "${BACKUP_DIR}/llm.pt" ]; then
    cp "${INFERENCE_DIR}/llm.pt" "${BACKUP_DIR}/llm.pt"
    echo "  Backed up: llm.pt"
else
    echo "  SKIP: llm.pt already backed up or not found"
fi

if [ -f "${INFERENCE_DIR}/flow.pt" ] && [ ! -f "${BACKUP_DIR}/flow.pt" ]; then
    cp "${INFERENCE_DIR}/flow.pt" "${BACKUP_DIR}/flow.pt"
    echo "  Backed up: flow.pt"
else
    echo "  SKIP: flow.pt already backed up or not found"
fi

if [ -f "${INFERENCE_DIR}/flow_linux_aarch64.om" ] && [ ! -f "${BACKUP_DIR}/flow_linux_aarch64.om" ]; then
    cp "${INFERENCE_DIR}/flow_linux_aarch64.om" "${BACKUP_DIR}/flow_linux_aarch64.om"
    echo "  Backed up: flow_linux_aarch64.om"
else
    echo "  SKIP: flow_linux_aarch64.om already backed up or not found"
fi

if [ -f "${INFERENCE_DIR}/flow_static.om" ] && [ ! -f "${BACKUP_DIR}/flow_static.om" ]; then
    cp "${INFERENCE_DIR}/flow_static.om" "${BACKUP_DIR}/flow_static.om"
    echo "  Backed up: flow_static.om"
else
    echo "  SKIP: flow_static.om already backed up or not found"
fi

if [ -f "${INFERENCE_DIR}/speech_linux_aarch64.om" ] && [ ! -f "${BACKUP_DIR}/speech_linux_aarch64.om" ]; then
    cp "${INFERENCE_DIR}/speech_linux_aarch64.om" "${BACKUP_DIR}/speech_linux_aarch64.om"
    echo "  Backed up: speech_linux_aarch64.om"
else
    echo "  SKIP: speech_linux_aarch64.om already backed up or not found"
fi
echo ""

# --- Step 5: Deploy ---
echo "[5/5] Deploying fine-tuned weights..."

cp "$LLM_MERGED" "${INFERENCE_DIR}/llm.pt"
echo "  Deployed: llm.pt (LoRA merged)"

cp "$FLOW_CLEAN" "${INFERENCE_DIR}/flow.pt"
echo "  Deployed: flow.pt (full SFT)"

echo ""
ls -lh "${INFERENCE_DIR}/llm.pt"
ls -lh "${INFERENCE_DIR}/flow.pt"

echo ""
echo "============================================================"
echo "  Phase 6 Complete!"
echo "============================================================"
echo ""
echo "Next: bash phase7_register_speaker.sh"
