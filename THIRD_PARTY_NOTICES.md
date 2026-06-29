# Third-Party Notices

This document lists direct third-party dependencies and benchmark/reference
projects used by Jet-Long. The Jet-Long repository distributes source code only;
third-party Python packages are installed separately by users through their
package managers.

## Direct Runtime Dependencies

| Dependency | Source | SPDX license identifier |
| --- | --- | --- |
| PyTorch (`torch`) | https://github.com/pytorch/pytorch | BSD-3-Clause |
| Hugging Face Transformers (`transformers`) | https://github.com/huggingface/transformers | Apache-2.0 |
| Hugging Face Datasets (`datasets`) | https://github.com/huggingface/datasets | Apache-2.0 |
| Hugging Face Accelerate (`accelerate`) | https://github.com/huggingface/accelerate | Apache-2.0 |
| FlashAttention (`flash-attn`) | https://github.com/Dao-AILab/flash-attention | BSD-3-Clause |
| Weights & Biases (`wandb`) | https://github.com/wandb/wandb | MIT |
| NVIDIA ML Python (`nvidia-ml-py`) | https://pypi.org/project/nvidia-ml-py/ | BSD-3-Clause |
| Portalocker (`portalocker`) | https://github.com/wolph/portalocker | BSD-3-Clause |
| PyYAML (`pyyaml`) | https://github.com/yaml/pyyaml | MIT |
| jieba (`jieba`) | https://github.com/fxsjy/jieba | MIT |
| rouge (`rouge`) | https://github.com/pltrdy/rouge | Apache-2.0 |
| lm-evaluation-harness (`lm_eval`) | https://github.com/EleutherAI/lm-evaluation-harness | MIT |
| Jet-Long pinned lm-evaluation-harness fork | https://github.com/jetlm-ai/lm-evaluation-harness | MIT |

## Optional Fused-Kernel Dependencies

| Dependency | Source | SPDX license identifier |
| --- | --- | --- |
| NVIDIA CUTLASS DSL (`nvidia-cutlass-dsl`) | https://pypi.org/project/nvidia-cutlass-dsl/ | LicenseRef-NVIDIA-Proprietary |
| NVIDIA CUTLASS DSL libs base (`nvidia-cutlass-dsl-libs-base`) | https://pypi.org/project/nvidia-cutlass-dsl-libs-base/ | LicenseRef-NVIDIA-Proprietary |
| quack-kernels (`quack-kernels`) | https://github.com/Dao-AILab/quack | Apache-2.0 |
| cuda-python (`cuda-python`) | https://github.com/NVIDIA/cuda-python | LicenseRef-NVIDIA-SOFTWARE-LICENSE |
| cuda-bindings (`cuda-bindings`) | https://github.com/NVIDIA/cuda-python | LicenseRef-NVIDIA-SOFTWARE-LICENSE |
| apache-tvm-ffi (`apache-tvm-ffi`) | https://github.com/apache/tvm | Apache-2.0 |
| torch-c-dlpack-ext (`torch-c-dlpack-ext`) | https://pypi.org/project/torch-c-dlpack-ext/ | Apache-2.0 |
| einops (`einops`) | https://github.com/arogozhnikov/einops | MIT |
| ninja (`ninja`) | https://github.com/ninja-build/ninja | Apache-2.0 OR BSD-2-Clause |
| flash_attn_4 (`flash_attn_4`) | https://github.com/Dao-AILab/flash-attention | BSD-3-Clause |

## Benchmark and Model References

| Project | Source | SPDX license identifier |
| --- | --- | --- |
| RULER | https://github.com/NVIDIA/RULER | Apache-2.0 |
| PG-19 | https://github.com/google-deepmind/pg19 | Apache-2.0 |
| HELMET | https://github.com/princeton-nlp/HELMET | MIT |
| Qwen3 | https://github.com/QwenLM/Qwen3 | Apache-2.0 |
