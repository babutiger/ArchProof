# Benchmark Models — Full Inventory

The ArchProof paper evaluates **123 ONNX models** across 9 categories,
totalling **~132 GB**. None of the model weights are redistributed in
this repository (size + license constraints).

This document lists every model with its **source URL** and how to
re-export it. Machine-readable form: `BENCHMARK_MANIFEST.json` next to
this file.

---

## 1. Whole-model 6–7B-parameter LLM ONNX (10 models, ~129 GB)

The headline LLM-scale experiment (Tab. 4 in paper).

| #   | Model                            | HuggingFace repo                                                                                | ONNX size | Re-export                                       |
| --- | -------------------------------- | ----------------------------------------------------------------------------------------------- | --------- | ----------------------------------------------- |
| 1   | GPT-J-6B (clean)                 | [EleutherAI/gpt-j-6b](https://huggingface.co/EleutherAI/gpt-j-6b)                               | 23 GB     | `scripts/export_clean_llm.py`                   |
| 2   | GPT-J-6B (backdoored)            | same                                                                                            | 23 GB     | clean export → `scripts/inject_llm_backdoor.py` |
| 3   | Yi-6B (clean)                    | [01-ai/Yi-6B](https://huggingface.co/01-ai/Yi-6B)                                               | 23 GB     | `scripts/export_clean_llm.py`                   |
| 4   | Yi-6B (backdoored)               | same                                                                                            | 23 GB     | clean export → `scripts/inject_llm_backdoor.py` |
| 5   | DeepSeek-LLM-7B (clean)          | [deepseek-ai/deepseek-llm-7b-base](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base)     | 26 GB     | `scripts/export_clean_llm.py`                   |
| 6   | DeepSeek-LLM-7B (bd)             | same                                                                                            | 26 GB     | clean export → `scripts/inject_llm_backdoor.py` |
| 7   | Mistral-7B-Instruct-v0.3 (clean) | [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | 28 GB     | `scripts/export_clean_llm.py`                   |
| 8   | Mistral-7B-Instruct-v0.3 (bd)    | same                                                                                            | 28 GB     | clean export → `scripts/inject_llm_backdoor.py` |
| 9   | Qwen2-7B (clean)                 | [Qwen/Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B)                                           | 29 GB     | `scripts/export_clean_llm.py`                   |
| 10  | Qwen2-7B (backdoored)            | same                                                                                            | 29 GB     | clean export → `scripts/inject_llm_backdoor.py` |

ONNX exports use opset 14 with dynamic batch axis. Backdoored variants
inject one Sigmoid+Mul gate at the final-layer SwiGLU MLP
(Yi / DeepSeek / Mistral / Qwen2) or final-layer GeLU MLP (GPT-J).

---

## 2. ImageNet-pretrained torchvision backbones — production scale (12 models, ~0.6 GB)

4 backbones × 3 Bober-Irizar gate types (`sep_tar`, `sha_un`, `int_un`).

| Backbone        | torchvision name  | Pretrained weights URL                                                     |
| --------------- | ----------------- | -------------------------------------------------------------------------- |
| ResNet18        | `resnet18`        | https://download.pytorch.org/models/resnet18-f37072fd.pth                  |
| ResNet50        | `resnet50`        | https://download.pytorch.org/models/resnet50-0676ba61.pth                  |
| MobileNetV2     | `mobilenet_v2`    | https://download.pytorch.org/models/mobilenet_v2-b0353104.pth              |
| EfficientNet-B0 | `efficientnet_b0` | https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth |

Naming: `<base>_<gate>.onnx`. Re-export via the inject scripts in `scripts/`.

---

## 3. Medium-scale transformer encoders (28 models, ~1.5 GB)

7 encoders × 4 variants (clean + 3 gate types).

| Encoder         | HuggingFace repo                                                                                |
| --------------- | ----------------------------------------------------------------------------------------------- |
| BERT-base       | [bert-base-uncased](https://huggingface.co/bert-base-uncased)                                   |
| DistilBERT      | [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)                       |
| GPT-2 (medium)  | [gpt2](https://huggingface.co/gpt2)                                                             |
| RoBERTa-base    | [roberta-base](https://huggingface.co/roberta-base)                                             |
| DeBERTa-v3-base | [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base)                   |
| ALBERT-base     | [albert-base-v2](https://huggingface.co/albert-base-v2)                                         |
| Electra-small   | [google/electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator) |

Variants per base: `clean`, `sep_tar`, `sha_un`, `int_un`. Re-export with
`scripts/export_clean_transformers.py` + `scripts/inject_transformer_backdoor.py`.

---

## 4. CIFAR-CNN backdoors — Bober-Irizar 12-type taxonomy + handcrafted (25 models, ~50 MB)

Buildable from scratch with the inject scripts in `scripts/` applied to
a small CIFAR baseline CNN (also constructed by the scripts).

**Bober-Irizar 22 variants** (constructions follow Bober-Irizar et al.,
[CVPR 2023](https://doi.org/10.1109/CVPR52729.2023.02355)):

```
op_sep_tar       op_sep_un          op_sep_un_L01  op_sep_un_L001  op_sep_un_L0001
op_sha_tar       op_sha_un          op_sha_tar_L01 op_sha_tar_L001 op_sha_tar_L0001
op_int_tar       op_int_un          op_int_tar_L01 op_int_tar_L001 op_int_tar_L0001
con_sep_tar      con_sep_un         con_sha_tar    con_sha_un
```

Plus 3 handcrafted extensions to test non-classical trigger preprocessing:

| Name                   | Gate construction                                     |
| ---------------------- | ----------------------------------------------------- |
| `H1_SignGated`         | Sign with Slice/Expand routing wrapper (out-of-class) |
| `H2_AvgPoolGated`      | ReLU(AvgPool(Wx))·p (in-class via ReLU activation)    |
| `H3_MulIndicatorGated` | ReLU(1{Wx>τ})·p (in-class via ReLU activation)        |

---

## 5. Real-scan clean baseline — 8 popular ONNX models (~1.5 GB)

Sanity-test for false-positives on production-scale clean models.

| Model           | Source      | URL                                                                        |
| --------------- | ----------- | -------------------------------------------------------------------------- |
| ResNet18        | torchvision | https://download.pytorch.org/models/resnet18-f37072fd.pth                  |
| ResNet50        | torchvision | https://download.pytorch.org/models/resnet50-0676ba61.pth                  |
| MobileNetV2     | torchvision | https://download.pytorch.org/models/mobilenet_v2-b0353104.pth              |
| EfficientNet-B0 | torchvision | https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth |
| VGG16           | torchvision | https://download.pytorch.org/models/vgg16-397923af.pth                     |
| BERT-base       | HuggingFace | https://huggingface.co/bert-base-uncased                                   |
| DistilBERT      | HuggingFace | https://huggingface.co/distilbert-base-uncased                             |
| GPT-2           | HuggingFace | https://huggingface.co/gpt2                                                |

---

## 6. Random-init torchvision backbones — 14-model out-of-class study (~1 GB)

Random-init weights (no pretrained download); built via
`scripts/export_clean_*.py --random-init`.

```
ResNet18, ResNet50, WideResNet50,
MobileNetV2, MobileNetV3-Small, EfficientNet-B0,
VGG11, VGG16,
DenseNet121, DenseNet169,
AlexNet, GoogLeNet, InceptionV3, SqueezeNet1.0
```

---

## 7. Synthetic stress-coverage clean models (45 models, ~50 MB)

Each is a self-contained `nn.Module` exercising one modern
architectural block with random weights. Buildable from scratch
with PyTorch + `torch.onnx.export` (no external download).
Examples of blocks: BiFPN, CBAM, ECA, ConvMixer, GhostNet, HardSwish,
GeGLU, MHSA, attention pool, gated residual, GLU FFN, Switch MoE,
SE block, etc. See `archproof/handcrafted_gdp.py` for the full list.

---

## 8. Adversarial G-probe constructions (3 models)

Buildable from `archproof/g1_g4_adversaries.py` — used only as
witness probes for the G_i necessity ablation.

| Model                        | Purpose                                                           |
| ---------------------------- | ----------------------------------------------------------------- |
| `clean_G1adv_dormant_add`    | G1-adversarial: dormant gate added back through Add (no-op clean) |
| `clean_G1adv_dormant_concat` | G1-adversarial: dormant gate concatenated                         |
| `clean_G4adv_ghost_mul`      | G4-adversarial: ghost Mul that does not influence output          |

---

## 9. Open-world streaming scan — 500 HuggingFace ONNX repos (0 GB on disk)

Streamed dynamically; never persisted to disk. The streaming scan loads
each repo, runs admission + verifier, then unloads. Per-repo verdicts
are saved in `results/`.

**Selection criterion**: HuggingFace `library:onnx` topic, sorted by
downloads, capped at 2 GB per repo.

---

## How to fetch + re-export

```bash
export ARCHPROOF_ROOT=/path/to/this/repo

# 1. Whole-model LLMs
python scripts/export_clean_llm.py --model EleutherAI/gpt-j-6b
python scripts/inject_llm_backdoor.py --base EleutherAI/gpt-j-6b

# 2. Medium transformers
python scripts/export_clean_transformers.py --model bert-base-uncased
python scripts/inject_transformer_backdoor.py --base bert-base-uncased --gate sep_tar

# 3. CIFAR backdoors / synthetic / adversarial
python archproof/handcrafted_gdp.py        # 45 synthetic stress models
python archproof/g1_g4_adversaries.py      # 3 G-probe constructions
# (Bober-Irizar 22 variants are emitted by archproof/build_in_class_bober_onnx.py)
```

The full machine-readable manifest is `BENCHMARK_MANIFEST.json`.
