# CosyVoice2 微调执行指南 v3

## 概述

本指南基于完整实践验证，提供可直接执行的 CosyVoice2 SFT 微调全流程。

**目标**: 在昇腾910B NPU上打通CosyVoice2的SFT微调全流程  
**策略**: LLM用LoRA（0.22%参数） + Flow用全量SFT  
**基线**: 官方仓库 commit fd45708 + xunyi训练recipe + 7文件代码补丁  
**验证环境**: 昇腾910B3 NPU + CANN 8.1.RC1 + torch 2.4.0 + torch_npu 2.4.0.post2

### 前置条件

**本流程需要离线环境执行**。所有外网依赖（Python包、代码仓库、预训练模型）已预先打包到镜像中。

镜像构建方法详见 `环境准备.md`。

### SFT微调原理说明

CosyVoice2 的 SFT 推理流程：
```
inference_sft(text, "speaker_name")
  → spk2info.pt 查找 speaker embedding（192维声纹向量）
  → LLM(text + embedding) → speech tokens
  → Flow(speech tokens + embedding) → mel → HiFT → 音频
```

- **speaker embedding** 决定"谁在说话"（音色）
- **模型权重**（LLM + Flow）决定"怎么说话"（发音质量、韵律、清晰度）
- **SFT微调**改的是模型权重，不改 speaker embedding

**想用SFT模式输出目标音色，必须同时做两件事**：
1. SFT微调模型权重（Phase 4-5）
2. 将目标speaker的embedding注册到spk2info.pt（Phase 7）

推理时调用 `inference_sft(text, "金牌客服")` 即可输出目标音色 + 改善后的发音质量。

---

## 脚本清单

脚本需上传到服务器

| Phase | 脚本 | 用途 | 运行位置 |
|-------|------|------|---------|
| 1 | `phase1_patch_code.sh` | 代码补丁（7文件一步完成） | 训练容器B |
| 2 | `phase2_prepare_dataset.sh` | 准备测试数据集索引 | 训练容器B |
| 3 | `phase3_data_prep.sh` | 数据预处理 | 训练容器B |
| 4 | `phase4_train_llm.sh` | LLM LoRA微调 | 训练容器B |
| 5 | `phase5_train_flow.sh` | Flow全量SFT | 训练容器B |
| 6 | `phase6_deploy_weights.sh` | LoRA合并 + 部署 | 训练容器B |
| 7 | `phase7_register_speaker.sh` | 注册新speaker | 训练容器B |
| 8 | `phase8_generate_om.sh` | 生成OM文件 | 训练容器B |
| 9 | `phase9_test_inference.sh` | 推理测试 | **推理容器A** |

---

## Phase 0: 创建训练容器

基于cosyvoice2推理镜像创建容器（镜像构建方法见 `环境准备.md`）

---

## Phase 1: 代码补丁

### 前提条件
- xunyi training recipe files 必须已经存在（预烘焙在镜像中）
- CosyVoice 代码仓库已克隆到 `/home/mind/model/cosyvoice_train/CosyVoice`

### 参数说明
```bash
bash phase1_patch_code.sh [lora_rank]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lora_rank` | 8 | LoRA 的秩（rank），控制可训练参数量 |

### 建议参数
- **验证流程**：`bash phase1_patch_code.sh`（默认 r=8，训练 0.22% 参数）
- **正式微调**：`bash phase1_patch_code.sh 16` 或 `bash phase1_patch_code.sh 32`（训练更多参数，效果更好）

### 执行
```bash
cd /home/mind/model/cosyvoice_train/CosyVoice
bash /home/mind/model/cosyvoice_train/scripts/phase1_patch_code.sh
```

一步完成全部 7 个文件的补丁，脚本末尾自动验证。

---

## Phase 2: 准备测试数据集

### 前提条件
- 无特殊前提

### 参数说明
```bash
bash phase2_prepare_dataset.sh
```

无参数。

### 建议参数
- 直接使用默认参数

### 执行
```bash
bash /home/mind/model/cosyvoice_train/scripts/phase2_prepare_dataset.sh
```

**手动操作**: 将中文语音WAV文件放到：
```bash
cp <你的音频文件> /home/mind/model/cosyvoice_train/data/sft_test/test.wav
```

要求：5-15秒、16kHz、中文语音、WAV(PCM)格式。  
如果音频文本不是"你好欢迎使用智能客服系统"，需编辑 `data/sft_test/text` 匹配。

---

## Phase 3: 数据预处理

### 前提条件
- Phase 1（代码补丁）已完成
- Phase 2（数据集准备）已完成
- test.wav 已放置到 `data/sft_test/`
- 预训练模型已下载到 `pretrained_models/CosyVoice2-0.5B/`

### 参数说明
```bash
bash phase3_data_prep.sh [data_dir]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data_dir` | `/home/mind/model/cosyvoice_train/data/sft_test` | 数据目录路径 |

### 建议参数
- **验证流程**：`bash phase3_data_prep.sh`（使用默认 sft_test 目录）
- **正式微调**：`bash phase3_data_prep.sh /path/to/your/data`（使用自定义数据目录）

### 执行
```bash
bash /home/mind/model/cosyvoice_train/scripts/phase3_data_prep.sh
```

### AISHELL3 数据集示例

如果你使用 AISHELL3 数据集（例如 SSB0671 speaker），需要先准备标准 CosyVoice 格式的数据：

```bash
# 1. 创建数据目录
mkdir -p /home/mind/model/cosyvoice_train/data/aishell3_ssb0671

# 2. 准备 wav.scp（列出所有音频文件）
find /home/mind/model/cosyvoice_train/aishell3/train/wav/SSB0671 -name "*.wav" | \
  awk -F'/' '{print $NF, $0}' | sed 's/\.wav//' > \
  /home/mind/model/cosyvoice_train/data/aishell3_ssb0671/wav.scp

# 3. 准备 text（从 AISHELL3 的 transcript 提取）
grep "SSB0671" /home/mind/model/cosyvoice_train/aishell3/train/transcript/aishell3_train.txt | \
  awk '{print $1, $2}' > /home/mind/model/cosyvoice_train/data/aishell3_ssb0671/text

# 4. 准备 utt2spk 和 spk2utt
awk '{print $1, "SSB0671"}' /home/mind/model/cosyvoice_train/data/aishell3_ssb0671/text > \
  /home/mind/model/cosyvoice_train/data/aishell3_ssb0671/utt2spk
echo "SSB0671 $(awk '{print $1}' /home/mind/model/cosyvoice_train/data/aishell3_ssb0671/text | tr '\n' ' ')" > \
  /home/mind/model/cosyvoice_train/data/aishell3_ssb0671/spk2utt

# 5. 运行数据预处理
bash /home/mind/model/cosyvoice_train/scripts/phase3_data_prep.sh \
  /home/mind/model/cosyvoice_train/data/aishell3_ssb0671
```

---

## Phase 4: LLM LoRA微调

### 前提条件
- Phase 1（代码补丁）已完成
- Phase 3（数据预处理）已完成
- 预训练模型已下载到 `pretrained_models/CosyVoice2-0.5B/`

### 参数说明
```bash
bash phase4_train_llm.sh [data_dir] [max_epoch] [lora_rank]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data_dir` | `/home/mind/model/cosyvoice_train/data/sft_test` | 数据目录路径 |
| `max_epoch` | 2 | 最大训练轮数 |
| `lora_rank` | 8 | LoRA 的秩（rank），必须与 phase1 一致 |

### 建议参数
- **验证流程**：`bash phase4_train_llm.sh`（默认 data_dir=sft_test, epoch=2, r=8）
- **正式微调**：`bash phase4_train_llm.sh /path/to/data 50 16`（自定义数据，50 epoch，r=16）

### 执行
```bash
bash /home/mind/model/cosyvoice_train/scripts/phase4_train_llm.sh
```

### AISHELL3 数据集示例

如果你使用 AISHELL3 数据集（例如 SSB0671 speaker）：

```bash
# 使用 AISHELL3 SSB0671 数据，训练 100 epoch，LoRA rank=16
bash /home/mind/model/cosyvoice_train/scripts/phase4_train_llm.sh \
  /home/mind/model/cosyvoice_train/data/aishell3_ssb0671 \
  100 \
  16
```

**注意**：
- AISHELL3 单个 speaker 约有 300-500 条音频，100 epoch 约需 8-12 小时
- 建议先用 20 epoch 快速验证，确认 loss 下降后再增加到 100 epoch
- LoRA rank 必须与 phase1 中指定的值一致

---

## Phase 5: Flow全量SFT

### 前提条件
- Phase 1（代码补丁）已完成
- Phase 3（数据预处理）已完成
- 预训练模型已下载到 `pretrained_models/CosyVoice2-0.5B/`

### 参数说明
```bash
bash phase5_train_flow.sh [data_dir] [max_epoch]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data_dir` | `/home/mind/model/cosyvoice_train/data/sft_test` | 数据目录路径 |
| `max_epoch` | 2 | 最大训练轮数 |

### 建议参数
- **验证流程**：`bash phase5_train_flow.sh`（默认 data_dir=sft_test, epoch=2）
- **正式微调**：`bash phase5_train_flow.sh /path/to/data 50`（自定义数据，50 epoch）

### 执行
```bash
bash /home/mind/model/cosyvoice_train/scripts/phase5_train_flow.sh
```

### AISHELL3 数据集示例

如果你使用 AISHELL3 数据集（例如 SSB0671 speaker）：

```bash
# 使用 AISHELL3 SSB0671 数据，训练 50 epoch
bash /home/mind/model/cosyvoice_train/scripts/phase5_train_flow.sh \
  /home/mind/model/cosyvoice_train/data/aishell3_ssb0671 \
  50
```

**注意**：
- Flow 模型是全量 SFT，训练速度比 LLM LoRA 快
- AISHELL3 单个 speaker 50 epoch 约需 3-5 小时
- Flow 模型通常在 20-30 epoch 就收敛，50 epoch 足够

---

## Phase 6: 权重合并和部署

### 前提条件
- Phase 4（LLM训练）已完成
- Phase 5（Flow训练）已完成

### 参数说明
```bash
bash phase6_deploy_weights.sh [lora_rank]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lora_rank` | 8 | LoRA 的秩（rank），必须与 phase1 和 phase4 一致 |

### 建议参数
- **验证流程**：`bash phase6_deploy_weights.sh`（默认 r=8）
- **正式微调**：`bash phase6_deploy_weights.sh 16`（与 phase1 和 phase4 保持一致）

### 执行
```bash
bash /home/mind/model/cosyvoice_train/scripts/phase6_deploy_weights.sh
```

完成：合并LoRA权重 → 清理Flow checkpoint → 备份原始权重和OM文件 → 部署到推理目录。

---

## Phase 7: 注册新speaker

### 前提条件
- Phase 6（权重部署）已完成
- 有目标 speaker 的音频文件（建议 10 条以上）

### 参数说明
```bash
# 目录模式（推荐）
bash phase7_register_speaker.sh [speaker_name] [wav_dir] [-n num_samples]

# 文件模式
bash phase7_register_speaker.sh [speaker_name] [wav_file1] [wav_file2] ...

# 默认模式
bash phase7_register_speaker.sh
```

**目录模式参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `speaker_name` | `spk001` | 注册的 speaker 名称 |
| `wav_dir` | `/home/mind/model/cosyvoice_train/data/sft_test` | 音频文件目录 |
| `-n num_samples` | 10 | 随机采样的音频数量 |

**文件模式参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `speaker_name` | `spk001` | 注册的 speaker 名称 |
| `wav_file1, wav_file2, ...` | - | 音频文件路径列表 |

### 建议参数
- **验证流程**：`bash phase7_register_speaker.sh`（使用默认 test.wav）
- **正式微调**：`bash phase7_register_speaker.sh "金牌客服" /path/to/wav_dir -n 10`（目录模式，随机采样 10 条）

### 执行
```bash
# 默认模式：使用 Phase 2 的 test.wav，注册为 "spk001"
bash /home/mind/model/cosyvoice_train/scripts/phase7_register_speaker.sh

# 目录模式（推荐）：从目录中随机采样 10 条音频
bash /home/mind/model/cosyvoice_train/scripts/phase7_register_speaker.sh \
  "金牌客服" /path/to/wav_dir -n 10

# 文件模式：手动指定音频文件
bash /home/mind/model/cosyvoice_train/scripts/phase7_register_speaker.sh \
  "金牌客服" /path/to/ref1.wav /path/to/ref2.wav /path/to/ref3.wav
```

将目标speaker的声纹embedding写入 `spk2info.pt`，使SFT推理模式可用。

### AISHELL3 数据集示例

如果你使用 AISHELL3 数据集（例如 SSB0671 speaker）：

```bash
# 从 AISHELL3 SSB0671 的音频目录中随机采样 10 条，注册为 "SSB0671"
bash /home/mind/model/cosyvoice_train/scripts/phase7_register_speaker.sh \
  "SSB0671" \
  /home/mind/model/cosyvoice_train/aishell3/train/wav/SSB0671 \
  -n 10
```

**注意**：
- AISHELL3 单个 speaker 约有 300-500 条音频
- 建议采样 10-20 条计算平均 embedding，效果更稳定
- 注册的 speaker 名称将在 Phase 9 推理时使用

---

## Phase 8: 生成OM文件

### 前提条件
- Phase 6（权重部署）已完成

### 参数说明
```bash
bash phase8_generate_om.sh
```

无参数。

### 建议参数
- 直接使用默认参数

### 执行
```bash
bash /home/mind/model/cosyvoice_train/scripts/phase8_generate_om.sh
```

脚本自动完成：
1. 检测NPU芯片型号（如910B3）
2. 导出flow decoder为ONNX
3. 转换ONNX → OM（动态shape，seq_len 1~2048）
4. 转换ONNX → OM（动态分档，gear: 40,140,...,840）
5. 修复文件权限（chmod 755 + chown）

> **注意**: `speech_linux_aarch64.om` 不需要重新生成（speech tokenizer未被微调）。  
> **注意**: LLM始终用PyTorch eager模式，不需要OM文件。

---

## Phase 9: 推理测试

### 前提条件
- Phase 7（注册speaker）已完成
- Phase 8（生成OM文件）已完成

### 参数说明
```bash
bash phase9_test_inference.sh [speaker_name] [port]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `speaker_name` | `SSB0671` | 要测试的 speaker 名称 |
| `port` | 50000 | TTS 服务端口 |

### 建议参数
- **验证流程**：`bash phase9_test_inference.sh`（默认 speaker=SSB0671, port=50000）
- **自定义测试**：`bash phase9_test_inference.sh "金牌客服" 50001`（自定义 speaker 和端口）

### 执行

在**推理容器A**中执行：

```bash
# 默认使用 SSB0671 speaker，端口 50000
bash /home/mind/model/cosyvoice_train/scripts/phase9_test_inference.sh

# 或指定 speaker 名称和端口
bash /home/mind/model/cosyvoice_train/scripts/phase9_test_inference.sh SSB0671 50000
```

脚本自动完成：
1. 设置环境变量（PYTHONPATH、LD_LIBRARY_PATH等）
2. 修复modelscope兼容性（首次运行）
3. 验证模型文件和OM文件
4. 运行 `deploy.sh` 生成 server.py/client.py（首次运行）
5. 启动 TTS 服务（使用 `run_server.sh`，后台运行）
6. 等待服务就绪
7. 测试微调后的 speaker（SSB0671）
8. 测试默认 speaker（中文女）用于对比

输出文件：
```
/home/mind/model/cosyvoice_train/exp/cosyvoice2/test_output/
├── sft_SSB0671.wav      # 微调后的音色
├── sft_default.wav      # 默认中文女音色（对比用）
└── server.log           # 服务日志
```

服务会持续运行，可手动测试：
```bash
curl -X POST http://127.0.0.1:50000/inference_sft \
  -F 'tts_text=你好世界' \
  -F 'spk_id=SSB0671' \
  -F 'stream=true' \
  --output test.wav
```

停止服务：
```bash
kill $(lsof -ti:50000)
```

---

## 恢复原始权重

```bash
INFERENCE_DIR=/home/mind/model/weight/CosyVoice2-0.5B
cp "$INFERENCE_DIR/backup_original/llm.pt" "$INFERENCE_DIR/llm.pt"
cp "$INFERENCE_DIR/backup_original/flow.pt" "$INFERENCE_DIR/flow.pt"
cp "$INFERENCE_DIR/backup_original/flow_linux_aarch64.om" "$INFERENCE_DIR/flow_linux_aarch64.om"
cp "$INFERENCE_DIR/backup_original/flow_static.om" "$INFERENCE_DIR/flow_static.om"
cp "$INFERENCE_DIR/backup_original/speech_linux_aarch64.om" "$INFERENCE_DIR/speech_linux_aarch64.om"
```

---

## 执行顺序总结

```
Phase 0:  创建容器（基于cosyvoice2_finetune:v1.0镜像）
Phase 1:  bash phase1_patch_code.sh [lora_rank]
Phase 2:  bash phase2_prepare_dataset.sh → 放test.wav
Phase 3:  bash phase3_data_prep.sh [data_dir]
Phase 4:  bash phase4_train_llm.sh [data_dir] [max_epoch] [lora_rank]
Phase 5:  bash phase5_train_flow.sh [data_dir] [max_epoch]
Phase 6:  bash phase6_deploy_weights.sh [lora_rank]
Phase 7:  bash phase7_register_speaker.sh [speaker_name] [wav_dir] [-n num_samples]
Phase 8:  bash phase8_generate_om.sh
Phase 9:  bash phase9_test_inference.sh [speaker_name] [port]（推理容器A）
```

---

## 关键依赖版本速查

```
transformers==4.40.1    peft==0.11.1
torchvision==0.19.0     huggingface_hub==0.23.5
torch==2.4.0            torch_npu==2.4.0.post2
deepspeed==0.14.2       diffusers==0.29.0
```

---

## 代码补丁文件清单（7个文件）

| 文件 | 补丁步骤 | 改动数 |
|------|----------|--------|
| `cosyvoice/bin/train.py` | Step 1 | 8处 |
| `cosyvoice/llm/llm.py` | Step 2 | 6处 |
| `cosyvoice/flow/decoder.py` | Step 3 | 1处 |
| `examples/.../cosyvoice2.yaml` | Step 4 | 4处 |
| `cosyvoice/flow/flow.py` | Step 5 | 2处 |
| `cosyvoice/utils/train_utils.py` | Step 6 | 6处 |
| `cosyvoice/dataset/processor.py` | Step 7 | 1处 |
