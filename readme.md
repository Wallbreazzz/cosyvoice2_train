# CosyVoice2 微调执行指南 v3

## 概述

本指南基于完整实践验证，提供可直接执行的 CosyVoice2 SFT 微调全流程。

**目标**: 在昇腾910B NPU上打通CosyVoice2的SFT微调全流程  
**策略**: LLM用LoRA（0.22%参数） + Flow用全量SFT  
**基线**: 官方仓库 commit fd45708 + xunyi训练recipe + 7文件代码补丁  
**验证环境**: 昇腾910B3 NPU + CANN 8.1.RC1 + torch 2.4.0 + torch_npu 2.4.0.post2

### 前置条件

**本流程需要离线环境执行**。所有外网依赖（Python包、代码仓库、预训练模型）已预先打包到镜像中。

镜像构建方法详见 `环境准备（需外网）.md`。

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

基于cosyvoice2推理镜像创建容器（镜像构建方法见 `环境准备（需外网）.md`）

---

## Phase 1: 代码补丁

### 1.1 上传脚本到服务器

```bash
mkdir -p /home/mind/model/cosyvoice_train/scripts
```

上传所有 `phase*.sh` 脚本到 `/home/mind/model/cosyvoice_train/scripts/`。

### 1.2 执行

```bash
cd /home/mind/model/cosyvoice_train/CosyVoice
bash /home/mind/model/cosyvoice_train/scripts/phase1_patch_code.sh
```

一步完成全部 7 个文件的补丁，脚本末尾自动验证。

---

## Phase 2: 准备测试数据集

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

```bash
bash /home/mind/model/cosyvoice_train/scripts/phase3_data_prep.sh
```

---

## Phase 4: LLM LoRA微调

```bash
bash /home/mind/model/cosyvoice_train/scripts/phase4_train_llm.sh
```

---

## Phase 5: Flow全量SFT

```bash
bash /home/mind/model/cosyvoice_train/scripts/phase5_train_flow.sh
```

---

## Phase 6: 权重合并和部署

```bash
bash /home/mind/model/cosyvoice_train/scripts/phase6_deploy_weights.sh
```

完成：合并LoRA权重 → 清理Flow checkpoint → 备份原始权重 → 部署到推理目录。

---

## Phase 7: 注册新speaker

```bash
# 默认模式：使用 Phase 2 的 test.wav，注册为 "spk001"
bash /home/mind/model/cosyvoice_train/scripts/phase7_register_speaker.sh

# 自定义模式：指定speaker名称和音频文件
bash /home/mind/model/cosyvoice_train/scripts/phase7_register_speaker.sh \
  "金牌客服" /path/to/ref1.wav /path/to/ref2.wav
```

将目标speaker的声纹embedding写入 `spk2info.pt`，使SFT推理模式可用。

---

## Phase 8: 生成OM文件

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

在**推理容器A**中执行：

```bash
bash /home/mind/model/cosyvoice_train/scripts/phase9_test_inference.sh
```

脚本自动完成：
1. 设置环境变量（PYTHONPATH、LD_LIBRARY_PATH等）
2. 修复modelscope兼容性（首次运行）
3. 验证模型文件和OM文件
4. 运行 `infer.py --stream`（load_om=True, fp16=True）

成功标志：
```
[INFO] load model .../flow_linux_aarch64.om success
[INFO] load model .../flow_static.om success
[INFO] load model .../speech_linux_aarch64.om success
[INFO] yield speech len X.XX, rtf X.XX
```

---

## 恢复原始权重

```bash
INFERENCE_DIR=/home/mind/model/weight/CosyVoice2-0.5B
cp "$INFERENCE_DIR/backup_original/llm.pt" "$INFERENCE_DIR/llm.pt"
cp "$INFERENCE_DIR/backup_original/flow.pt" "$INFERENCE_DIR/flow.pt"
```

---

## 执行顺序总结

```
Phase 0:  创建容器（基于cosyvoice2_finetune:v1.0镜像）
Phase 1:  bash phase1_patch_code.sh
Phase 2:  bash phase2_prepare_dataset.sh → 放test.wav
Phase 3:  bash phase3_data_prep.sh
Phase 4:  bash phase4_train_llm.sh
Phase 5:  bash phase5_train_flow.sh
Phase 6:  bash phase6_deploy_weights.sh
Phase 7:  bash phase7_register_speaker.sh
Phase 8:  bash phase8_generate_om.sh
Phase 9:  bash phase9_test_inference.sh（推理容器A）
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
