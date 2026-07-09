#!/bin/bash
# =============================================================================
# Phase 1: Complete code patches for CosyVoice2 SFT fine-tuning
#
# Run from: /home/mind/model/cosyvoice_train/CosyVoice
# Based on: official CosyVoice commit fd45708
#
# This script patches 7 files in one shot:
#   1. Patches train.py (NPU + deepspeed optional + LoRA + hccl)
#   2. Patches llm.py (forward methods + LoRA-safe embed_tokens)
#   3. Patches decoder.py (static_chunk_size)
#   4. Patches cosyvoice2.yaml (spk_embedding + epoch + class fix + whisper cleanup)
#   5. Patches flow.py (forward method for training)
#   6. Patches train_utils.py (deepspeed optional + prefetch conditional)
#   7. Patches processor.py (token_mel_ratio)
#
# Usage:
#   bash phase1_patch_code.sh [lora_rank]
#
# Examples:
#   bash phase1_patch_code.sh        # Default: r=8
#   bash phase1_patch_code.sh 16     # Use r=16
#   bash phase1_patch_code.sh 32     # Use r=32
#
# Prerequisites: xunyi training recipe files must already exist (pre-baked in image)
# =============================================================================

set -e

COSYVOICE_DIR="/home/mind/model/cosyvoice_train/CosyVoice"
LORA_RANK="${1:-8}"

echo "============================================================"
echo "  Phase 1: Complete Code Patches (7 files)"
echo "============================================================"
echo ""
echo "  LoRA rank: r=$LORA_RANK"
echo ""

if [ ! -f "$COSYVOICE_DIR/cosyvoice/bin/train.py" ]; then
    echo "ERROR: $COSYVOICE_DIR/cosyvoice/bin/train.py not found"
    exit 1
fi

cd "$COSYVOICE_DIR"

# =============================================================================
# Verify prerequisites (xunyi training recipe, pre-baked in image)
# =============================================================================
echo "[0/7] Verifying xunyi training recipe..."

if [ ! -f examples/libritts/cosyvoice2/conf/cosyvoice2.yaml ]; then
    echo "ERROR: examples/libritts/cosyvoice2/conf/cosyvoice2.yaml not found"
    echo "This file should be pre-baked in the image. See build_image.md"
    exit 1
fi
echo "  OK: cosyvoice2.yaml"

if [ ! -f examples/libritts/cosyvoice2/run.sh ]; then
    echo "ERROR: examples/libritts/cosyvoice2/run.sh not found"
    exit 1
fi
echo "  OK: run.sh"

if [ ! -f examples/libritts/cosyvoice2/conf/ds_stage2.json ]; then
    echo "ERROR: examples/libritts/cosyvoice2/conf/ds_stage2.json not found"
    exit 1
fi
echo "  OK: ds_stage2.json"
echo ""

# =============================================================================
# Step 1: Patch cosyvoice/bin/train.py
# =============================================================================
echo "[1/7] Patching cosyvoice/bin/train.py..."

python3 << 'PATCH_TRAIN_PY'
filepath = "cosyvoice/bin/train.py"
with open(filepath, "r") as f:
    content = f.read()

original = content
changes = []

# Patch 1: NPU imports at the very top
npu_imports = """import torch_npu
from torch_npu.contrib import transfer_to_npu

"""
if "import torch_npu" not in content:
    content = npu_imports + content
    changes.append("Added torch_npu + transfer_to_npu imports")

# Patch 2: Make deepspeed optional
old_deepspeed = "import deepspeed"
new_deepspeed = """try:
    import deepspeed
except ImportError:
    deepspeed = None"""
if "try:\n    import deepspeed" not in content:
    content = content.replace(old_deepspeed, new_deepspeed, 1)
    changes.append("Made deepspeed import optional")

# Patch 3: deepspeed.add_config_arguments conditional
old_ds_args = "    parser = deepspeed.add_config_arguments(parser)"
new_ds_args = """    if deepspeed is not None:
        parser = deepspeed.add_config_arguments(parser)"""
if "if deepspeed is not None:" not in content:
    content = content.replace(old_ds_args, new_ds_args, 1)
    changes.append("Made deepspeed.add_config_arguments conditional")

# Patch 4: Add --qwen_pretrain_path and --onnx_path arguments
old_checkpoint_arg = "    parser.add_argument('--checkpoint', help='checkpoint model')"
new_checkpoint_arg = """    parser.add_argument('--qwen_pretrain_path', required=False, help='qwen pretrain path')
    parser.add_argument('--onnx_path', required=False, help='onnx path for online feature extraction')
    parser.add_argument('--checkpoint', help='checkpoint model')"""
if "--qwen_pretrain_path" not in content:
    content = content.replace(old_checkpoint_arg, new_checkpoint_arg, 1)
    changes.append("Added --qwen_pretrain_path and --onnx_path arguments")

# Patch 5: Add os.environ['onnx_path']
old_main_start = """    args = get_args()
    logging.basicConfig"""
new_main_start = """    args = get_args()
    os.environ['onnx_path'] = getattr(args, 'onnx_path', '') or ''
    logging.basicConfig"""
if "os.environ['onnx_path']" not in content:
    content = content.replace(old_main_start, new_main_start, 1)
    changes.append("Added os.environ['onnx_path'] setting")

# Patch 6: qwen_pretrain_path override to load_hyperpyyaml
old_override = """    override_dict = {k: None for k in ['llm', 'flow', 'hift', 'hifigan'] if k != args.model}
    if gan is True:
        override_dict.pop('hift')
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)"""
new_override = """    override_dict = {k: None for k in ['llm', 'flow', 'hift', 'hifigan'] if k != args.model}
    if gan is True:
        override_dict.pop('hift')
    if getattr(args, 'qwen_pretrain_path', None) is not None:
        override_dict['qwen_pretrain_path'] = args.qwen_pretrain_path
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)"""
if "qwen_pretrain_path" not in content.split("load_hyperpyyaml")[0].split("override_dict")[-1]:
    content = content.replace(old_override, new_override, 1)
    changes.append("Added qwen_pretrain_path override to load_hyperpyyaml")

# Patch 7: LoRA injection
old_wrap = "    # Dispatch model from cpu to gpu\n    model = wrap_cuda_model(args, model)"
new_wrap = """    # LoRA injection for LLM model
    if args.model == 'llm' and hasattr(model, 'llm') and hasattr(model.llm, 'model'):
        try:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=$LORA_RANK,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model.llm.model = get_peft_model(model.llm.model, lora_config)
            model.llm.model.print_trainable_parameters()
            logging.info('LoRA adapter injected into Qwen2 model')
        except ImportError:
            logging.warning('peft not installed, skipping LoRA injection. Full SFT will be used.')
        except Exception as e:
            logging.warning(f'LoRA injection failed: {e}, falling back to full SFT')

    # Dispatch model from cpu to gpu
    model = wrap_cuda_model(args, model)"""
if "LoRA injection" not in content:
    content = content.replace(old_wrap, new_wrap, 1)
    changes.append("Added LoRA injection after checkpoint loading")

# Patch 8: hccl backend
if "'hccl'" not in content:
    content = content.replace(
        "choices=['nccl', 'gloo']",
        "choices=['nccl', 'gloo', 'hccl']",
        1
    )
    changes.append("Added hccl to dist_backend choices")

if content != original:
    with open(filepath, "w") as f:
        f.write(content)
    print(f"  Applied {len(changes)} patches:")
    for c in changes:
        print(f"    - {c}")
else:
    print("  SKIP: All patches already applied")
PATCH_TRAIN_PY

echo ""

# =============================================================================
# Step 3: Patch cosyvoice/llm/llm.py (with LoRA-safe embed_tokens)
# =============================================================================
echo "[2/7] Patching cosyvoice/llm/llm.py..."

python3 << 'PATCH_LLM_PY'
filepath = "cosyvoice/llm/llm.py"
with open(filepath, "r") as f:
    content = f.read()

original = content
changes = []

# Patch 1: import random
if "import random" not in content:
    content = content.replace(
        "from typing import Dict, Optional, Callable, List, Generator",
        "import random\nfrom typing import Dict, Optional, Callable, List, Generator",
        1
    )
    changes.append("Added 'import random'")

# Patch 2: make_pad_mask import
if "from cosyvoice.utils.mask import make_pad_mask" not in content:
    content = content.replace(
        "from cosyvoice.utils.file_utils import logging",
        "from cosyvoice.utils.file_utils import logging\nfrom cosyvoice.utils.mask import make_pad_mask",
        1
    )
    changes.append("Added 'from cosyvoice.utils.mask import make_pad_mask'")

# Patch 3: Qwen2Encoder.forward()
old_forward_one_step = """    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]"""

new_with_forward = """    def forward(self, xs, xs_lens):
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T)
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=masks,
            output_hidden_states=True,
            return_dict=True,
        )
        return outs.hidden_states[-1], masks.unsqueeze(1)

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]"""

if "def forward(self, xs, xs_lens):" not in content:
    content = content.replace(old_forward_one_step, new_with_forward, 1)
    changes.append("Added Qwen2Encoder.forward() method for training")

# Patch 4: Qwen2LM.prepare_lm_input_target() + forward() with LoRA-safe embed_tokens
old_inference = """    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.llm.model.model.embed_tokens(text)"""

new_with_forward_and_inference = """    def prepare_lm_input_target(self, text_token, text_token_emb, text_token_len, speech_token, speech_token_emb, speech_token_len):
        lm_target, lm_input = [], []
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        text_token_emb = unpad_sequence(text_token_emb, text_token_len.cpu(), batch_first=True)
        speech_token_emb = unpad_sequence(speech_token_emb, speech_token_len.cpu(), batch_first=True)
        for i in range(len(text_token)):
            # bistream sequence (50% probability)
            if random.random() < 0.5 and speech_token_len[i] / text_token_len[i] > self.mix_ratio[1] / self.mix_ratio[0]:
                this_lm_target, this_lm_input = [], []
                this_lm_target.append(IGNORE_ID)
                this_lm_input.append(self.llm_embedding.weight[self.sos_eos].reshape(1, -1))
                for j in range(((text_token_len[i] + 1) / self.mix_ratio[0]).ceil().int().item()):
                    this_text_token = text_token[i][j * self.mix_ratio[0]: (j + 1) * self.mix_ratio[0]].tolist()
                    this_speech_token = speech_token[i][j * self.mix_ratio[1]: (j + 1) * self.mix_ratio[1]].tolist()
                    if len(this_text_token) == self.mix_ratio[0]:
                        assert len(this_speech_token) == self.mix_ratio[1]
                        this_lm_target += [IGNORE_ID] * (self.mix_ratio[0] - 1)
                        this_lm_target += this_speech_token
                        this_lm_target.append(self.speech_token_size + 2)
                        this_lm_input.append(text_token_emb[i][j * self.mix_ratio[0]: (j + 1) * self.mix_ratio[0]])
                        this_lm_input.append(speech_token_emb[i][j * self.mix_ratio[1]: (j + 1) * self.mix_ratio[1]])
                    else:
                        this_lm_target += [-1] * len(this_text_token)
                        this_lm_target += speech_token[i][j * self.mix_ratio[1]:].tolist()
                        this_lm_target.append(self.speech_token_size)
                        this_lm_input.append(text_token_emb[i][j * self.mix_ratio[0]:])
                        this_lm_input.append(self.llm_embedding.weight[self.task_id].reshape(1, -1))
                        this_lm_input.append(speech_token_emb[i][j * self.mix_ratio[1]:])
                this_lm_target, this_lm_input = torch.tensor(this_lm_target), torch.concat(this_lm_input, dim=0)
            # unistream sequence (50% probability)
            else:
                this_lm_target = torch.tensor([IGNORE_ID] * (1 + text_token_len[i]) + speech_token[i].tolist() + [self.speech_token_size])
                this_lm_input = torch.concat([self.llm_embedding.weight[self.sos_eos].reshape(1, -1), text_token_emb[i],
                                              self.llm_embedding.weight[self.task_id].reshape(1, -1), speech_token_emb[i]], dim=0)
            lm_target.append(this_lm_target)
            lm_input.append(this_lm_input)
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID)
        return lm_target, lm_input, lm_input_len

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)

        # 1. encode text_token (LoRA-safe: unwrap PEFT if present)
        _base_model = self.llm.model.get_base_model() if hasattr(self.llm.model, 'get_base_model') else self.llm.model
        text_token_emb = _base_model.model.embed_tokens(text_token)

        # 2. encode speech_token
        speech_token_emb = self.speech_embedding(speech_token)

        # 3. prepare llm_input/target
        lm_target, lm_input, lm_input_len = self.prepare_lm_input_target(
            text_token, text_token_emb, text_token_len, speech_token, speech_token_emb, speech_token_len)
        lm_target = lm_target.to(device)

        # 4. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target.to(device))
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 3), lm_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc}

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        _base_model = self.llm.model.get_base_model() if hasattr(self.llm.model, 'get_base_model') else self.llm.model
        text = _base_model.model.embed_tokens(text)"""

if "def prepare_lm_input_target" not in content:
    content = content.replace(old_inference, new_with_forward_and_inference, 1)
    changes.append("Added Qwen2LM.prepare_lm_input_target() + forward() with LoRA-safe embed_tokens")

# Patch 5: Fix remaining inference embed_tokens references
old_infer_embed = "        text = self.llm.model.model.embed_tokens(text)"
new_infer_embed = """        _base_model = self.llm.model.get_base_model() if hasattr(self.llm.model, 'get_base_model') else self.llm.model
        text = _base_model.model.embed_tokens(text)"""
if old_infer_embed in content:
    content = content.replace(old_infer_embed, new_infer_embed)
    changes.append("Fixed embed_tokens in remaining inference methods")

# Patch 6: Fix inference_bistream embed_tokens
old_bi = "        text_cache = self.llm.model.model.embed_tokens(prompt_text)"
new_bi = """        _base_model = self.llm.model.get_base_model() if hasattr(self.llm.model, 'get_base_model') else self.llm.model
        text_cache = _base_model.model.embed_tokens(prompt_text)"""
if old_bi in content:
    content = content.replace(old_bi, new_bi, 1)
    changes.append("Fixed embed_tokens in inference_bistream()")

old_bi2 = "            text_cache = torch.concat([text_cache, self.llm.model.model.embed_tokens(this_text)], dim=1)"
new_bi2 = "            text_cache = torch.concat([text_cache, _base_model.model.embed_tokens(this_text)], dim=1)"
if old_bi2 in content:
    content = content.replace(old_bi2, new_bi2, 1)
    changes.append("Fixed embed_tokens in inference_bistream loop")

if content != original:
    with open(filepath, "w") as f:
        f.write(content)
    print(f"  Applied {len(changes)} patches:")
    for c in changes:
        print(f"    - {c}")
else:
    print("  SKIP: All patches already applied")
PATCH_LLM_PY

echo ""

# =============================================================================
# Step 4: Patch cosyvoice/flow/decoder.py
# =============================================================================
echo "[3/7] Patching cosyvoice/flow/decoder.py..."

python3 << 'PATCH_DECODER_PY'
filepath = "cosyvoice/flow/decoder.py"
with open(filepath, "r") as f:
    content = f.read()

original = content
changes = []

old_init = """        act_fn="snake",
    ):
        \"\"\"
        This decoder requires an input with the same shape of the target. So, if your text content
        is shorter or longer than the outputs, please re-sampling it before feeding to the decoder.
        \"\"\"
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal"""

new_init = """        act_fn="snake",
        static_chunk_size=0,
        num_decoding_left_chunks=-1,
    ):
        \"\"\"
        This decoder requires an input with the same shape of the target. So, if your text content
        is shorter or longer than the outputs, please re-sampling it before feeding to the decoder.
        \"\"\"
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal
        self.static_chunk_size = static_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks"""

if "self.static_chunk_size = static_chunk_size" not in content:
    content = content.replace(old_init, new_init, 1)
    changes.append("Added static_chunk_size and num_decoding_left_chunks params")

if content != original:
    with open(filepath, "w") as f:
        f.write(content)
    print(f"  Applied {len(changes)} patches:")
    for c in changes:
        print(f"    - {c}")
else:
    print("  SKIP: All patches already applied")
PATCH_DECODER_PY

echo ""

# =============================================================================
# Step 5: Patch cosyvoice2.yaml
# =============================================================================
echo "[4/7] Patching examples/libritts/cosyvoice2/conf/cosyvoice2.yaml..."

python3 << 'PATCH_YAML_PY'
filepath = "examples/libritts/cosyvoice2/conf/cosyvoice2.yaml"
with open(filepath, "r") as f:
    content = f.read()

original = content
changes = []

# Patch 1: use_spk_embedding: False -> True
if "use_spk_embedding: True" not in content:
    content = content.replace("use_spk_embedding: False", "use_spk_embedding: True", 1)
    changes.append("use_spk_embedding: False -> True")

# Patch 2: max_epoch: 200 -> 2
if "max_epoch: 2" not in content.split("train_conf")[1] if "train_conf" in content else "":
    content = content.replace("max_epoch: 200", "max_epoch: 2", 1)
    changes.append("max_epoch: 200 -> 2 (for pipeline verification)")

# Patch 3: CausalConditionalDecoder -> ConditionalDecoder + causal: True
if "CausalConditionalDecoder" in content:
    content = content.replace(
        "estimator: !new:cosyvoice.flow.decoder.CausalConditionalDecoder",
        "estimator: !new:cosyvoice.flow.decoder.ConditionalDecoder",
        1
    )
    old_estimator = """        estimator: !new:cosyvoice.flow.decoder.ConditionalDecoder
            in_channels: 320
            out_channels: 80"""
    new_estimator = """        estimator: !new:cosyvoice.flow.decoder.ConditionalDecoder
            in_channels: 320
            out_channels: 80
            causal: True"""
    content = content.replace(old_estimator, new_estimator, 1)
    changes.append("CausalConditionalDecoder -> ConditionalDecoder + causal: True")

# Patch 4: Remove compute_whisper_fbank
if "compute_whisper_fbank" in content:
    content = content.replace("    !ref <compute_whisper_fbank>,\n", "", 1)
    changes.append("Removed compute_whisper_fbank from data_pipeline")

    old_def = """compute_whisper_fbank: !name:cosyvoice.dataset.processor.compute_whisper_fbank
    num_frames: 960
"""
    content = content.replace(old_def, "", 1)
    changes.append("Removed compute_whisper_fbank definition")

if content != original:
    with open(filepath, "w") as f:
        f.write(content)
    print(f"  Applied {len(changes)} patches:")
    for c in changes:
        print(f"    - {c}")
else:
    print("  SKIP: All patches already applied")
PATCH_YAML_PY

echo ""

# =============================================================================
# Step 6: Patch cosyvoice/flow/flow.py
# =============================================================================
echo "[5/7] Patching cosyvoice/flow/flow.py..."

python3 << 'PATCH_FLOW_PY'
filepath = "cosyvoice/flow/flow.py"
with open(filepath, "r") as f:
    content = f.read()

original = content
changes = []

if "import random" not in content:
    content = content.replace("import logging", "import logging\nimport random", 1)
    changes.append("Added 'import random'")

old_inference = """    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  finalize):
        if self.fp16 is True:
            prompt_feat = prompt_feat.half()
            embedding = embedding.half()

        assert token.shape[0] == 1"""

new_with_forward = """    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        token = batch['speech_token'].to(device)
        token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device)
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device)

        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)

        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(feat_len)).to(h)
        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  finalize):
        if self.fp16 is True:
            prompt_feat = prompt_feat.half()
            embedding = embedding.half()

        assert token.shape[0] == 1"""

if "def forward" not in content.split("class CausalMaskedDiffWithXvec")[1].split("@torch.inference_mode()")[0]:
    content = content.replace(old_inference, new_with_forward, 1)
    changes.append("Added CausalMaskedDiffWithXvec.forward() for training")

if content != original:
    with open(filepath, "w") as f:
        f.write(content)
    print(f"  Applied {len(changes)} patches:")
    for c in changes:
        print(f"    - {c}")
else:
    print("  SKIP: All patches already applied")
PATCH_FLOW_PY

echo ""

# =============================================================================
# Step 7: Patch cosyvoice/utils/train_utils.py
# =============================================================================
echo "[6/7] Patching cosyvoice/utils/train_utils.py..."

python3 << 'PATCH_TRAIN_UTILS_PY'
filepath = "cosyvoice/utils/train_utils.py"
with open(filepath, "r") as f:
    content = f.read()

original = content
changes = []

# 1. deepspeed import optional
if "try:\n    import deepspeed" not in content:
    content = content.replace(
        "import deepspeed",
        "try:\n    import deepspeed\nexcept ImportError:\n    deepspeed = None",
        1
    )
    changes.append("Made deepspeed import optional")

# 2. deepspeed.runtime.zero import optional
old_zero = "from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live"
new_zero = """try:
    from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
except (ImportError, AttributeError):
    estimate_zero2_model_states_mem_needs_all_live = None"""
if "estimate_zero2_model_states_mem_needs_all_live = None" not in content:
    content = content.replace(old_zero, new_zero, 1)
    changes.append("Made deepspeed.runtime.zero import optional")

# 3. deepspeed.init_distributed conditional
old_init = "        deepspeed.init_distributed(dist_backend=args.dist_backend)"
new_init = """        if deepspeed is not None:
            deepspeed.init_distributed(dist_backend=args.dist_backend)
        else:
            raise RuntimeError('deepspeed is required for deepspeed engine')"""
if "deepspeed is not None" not in content.split("init_distributed")[1].split("def ")[0] if "init_distributed" in content else "":
    content = content.replace(old_init, new_init, 1)
    changes.append("Made deepspeed.init_distributed conditional")

# 4. estimate_zero2 call conditional
old_est = """            estimate_zero2_model_states_mem_needs_all_live(
                model,
                num_gpus_per_node=local_world_size,
                num_nodes=world_size // local_world_size)"""
new_est = """            if estimate_zero2_model_states_mem_needs_all_live is not None:
                estimate_zero2_model_states_mem_needs_all_live(
                    model,
                    num_gpus_per_node=local_world_size,
                    num_nodes=world_size // local_world_size)"""
if "estimate_zero2_model_states_mem_needs_all_live is not None" not in content:
    content = content.replace(old_est, new_est, 1)
    changes.append("Made estimate_zero2 call conditional")

# 5. deepspeed.initialize conditional
old_ds = """            model, optimizer, _, scheduler = deepspeed.initialize(
                args=args,
                model=model,
                optimizer=None,
                lr_scheduler=scheduler,
                model_parameters=model.parameters)"""
new_ds = """            if deepspeed is not None:
                model, optimizer, _, scheduler = deepspeed.initialize(
                    args=args,
                    model=model,
                    optimizer=None,
                    lr_scheduler=scheduler,
                    model_parameters=model.parameters)
            else:
                raise RuntimeError('deepspeed is required for deepspeed engine')"""
if "deepspeed is required for deepspeed engine" not in content:
    content = content.replace(old_ds, new_ds, 1)
    changes.append("Made deepspeed.initialize conditional")

# 6. prefetch_factor conditional
old_dl = """    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)"""
new_dl = """    dataloader_kwargs = {
        'batch_size': None,
        'pin_memory': args.pin_memory,
        'num_workers': args.num_workers,
    }
    if args.num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = args.prefetch

    train_data_loader = DataLoader(train_dataset, **dataloader_kwargs)
    cv_data_loader = DataLoader(cv_dataset, **dataloader_kwargs)"""
if "dataloader_kwargs" not in content:
    content = content.replace(old_dl, new_dl, 1)
    changes.append("Made prefetch_factor conditional on num_workers > 0")

if content != original:
    with open(filepath, "w") as f:
        f.write(content)
    print(f"  Applied {len(changes)} patches:")
    for c in changes:
        print(f"    - {c}")
else:
    print("  SKIP: All patches already applied")
PATCH_TRAIN_UTILS_PY

echo ""

# =============================================================================
# Step 8: Patch cosyvoice/dataset/processor.py
# =============================================================================
echo "[7/7] Patching cosyvoice/dataset/processor.py..."

python3 << 'PATCH_PROCESSOR_PY'
filepath = "cosyvoice/dataset/processor.py"
with open(filepath, "r") as f:
    content = f.read()

original = content
changes = []

old_fbank = """def compute_fbank(data,
                  feat_extractor,
                  mode='train'):
    \"\"\" Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    \"\"\"
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        assert 'utt' in sample
        assert 'text_token' in sample
        waveform = sample['speech']
        mat = feat_extractor(waveform).squeeze(dim=0).transpose(0, 1)
        sample['speech_feat'] = mat
        yield sample"""

new_fbank = """def compute_fbank(data,
                  feat_extractor,
                  token_mel_ratio=0,
                  mode='train'):
    \"\"\" Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    \"\"\"
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        assert 'utt' in sample
        assert 'text_token' in sample
        waveform = sample['speech']
        feat = feat_extractor(waveform).squeeze(dim=0).transpose(0, 1)
        if token_mel_ratio != 0:
            token_len = int(min(feat.shape[0] / token_mel_ratio, sample["speech_token"].shape[0]))
            feat = feat[:token_mel_ratio * token_len]
            sample["speech_token"] = sample["speech_token"][:token_len]
        sample['speech_feat'] = feat
        yield sample"""

if "token_mel_ratio" not in content:
    content = content.replace(old_fbank, new_fbank, 1)
    changes.append("Added token_mel_ratio to compute_fbank")

if content != original:
    with open(filepath, "w") as f:
        f.write(content)
    print(f"  Applied {len(changes)} patches:")
    for c in changes:
        print(f"    - {c}")
else:
    print("  SKIP: All patches already applied")
PATCH_PROCESSOR_PY

echo ""

# =============================================================================
# Verification
# =============================================================================
echo "============================================================"
echo "  Verification (all 7 files)"
echo "============================================================"
echo ""

echo "--- train.py ---"
grep -c "import torch_npu" cosyvoice/bin/train.py && echo "  OK: torch_npu" || echo "  FAIL: torch_npu"
grep -c "deepspeed is not None" cosyvoice/bin/train.py && echo "  OK: deepspeed optional" || echo "  FAIL: deepspeed"
grep -c "LoRA" cosyvoice/bin/train.py && echo "  OK: LoRA injection" || echo "  FAIL: LoRA"
grep -c "hccl" cosyvoice/bin/train.py && echo "  OK: hccl backend" || echo "  FAIL: hccl"

echo ""
echo "--- llm.py ---"
grep -c "def forward(self, xs, xs_lens):" cosyvoice/llm/llm.py && echo "  OK: Qwen2Encoder.forward()" || echo "  FAIL: Qwen2Encoder.forward()"
grep -c "def prepare_lm_input_target" cosyvoice/llm/llm.py && echo "  OK: prepare_lm_input_target()" || echo "  FAIL: prepare_lm_input_target()"
grep -c "get_base_model" cosyvoice/llm/llm.py && echo "  OK: LoRA-safe embed_tokens" || echo "  FAIL: LoRA-safe embed_tokens"
grep -c "import random" cosyvoice/llm/llm.py && echo "  OK: import random" || echo "  FAIL: import random"

echo ""
echo "--- decoder.py ---"
grep -c "static_chunk_size=0" cosyvoice/flow/decoder.py && echo "  OK: static_chunk_size" || echo "  FAIL: static_chunk_size"

echo ""
echo "--- cosyvoice2.yaml ---"
grep "use_spk_embedding" examples/libritts/cosyvoice2/conf/cosyvoice2.yaml
grep "max_epoch" examples/libritts/cosyvoice2/conf/cosyvoice2.yaml | head -1
grep "CausalConditionalDecoder\|ConditionalDecoder" examples/libritts/cosyvoice2/conf/cosyvoice2.yaml
grep "compute_whisper_fbank" examples/libritts/cosyvoice2/conf/cosyvoice2.yaml || echo "  OK: compute_whisper_fbank removed"

echo ""
echo "--- flow.py ---"
grep -c "def forward" cosyvoice/flow/flow.py && echo "  OK: CausalMaskedDiffWithXvec.forward()" || echo "  FAIL: flow.forward()"
grep -c "import random" cosyvoice/flow/flow.py && echo "  OK: import random" || echo "  FAIL: import random"

echo ""
echo "--- train_utils.py ---"
grep -c "deepspeed is not None" cosyvoice/utils/train_utils.py && echo "  OK: deepspeed conditional" || echo "  FAIL: deepspeed conditional"
grep -c "dataloader_kwargs" cosyvoice/utils/train_utils.py && echo "  OK: prefetch conditional" || echo "  FAIL: prefetch conditional"

echo ""
echo "--- processor.py ---"
grep -c "token_mel_ratio" cosyvoice/dataset/processor.py && echo "  OK: token_mel_ratio" || echo "  FAIL: token_mel_ratio"

echo ""
echo "============================================================"
echo "  Phase 1 Complete! (7 files patched)"
echo "============================================================"
echo ""
echo "Next: bash phase2_prepare_dataset.sh"
