# RFF-LLM

# Submit to SPL

RFF-LLM is a **dual-modal framework** for cross-domain radio frequency fingerprinting (RFF). It jointly leverages waveform features from a lightweight IQ branch and the domain-agnostic prior of a **frozen LLM** (Qwen3-Embedding-0.6B). To bridge the gap between continuous IQ signals and the discrete input space of the LLM, a **VQ-VAE signal tokenizer** converts raw IQ waveforms into discrete tokens, which are then encoded by the frozen LLM under a textual prompt and fused with the IQ branch for transmitter identification.

The framework consistently improves cross-receiver / cross-day generalization on the **WiSig** (ManySig, ManyRx) and **LoRa** datasets.

## Architecture

The pipeline is organized into three sequential stages, the first two of which are fully frozen, leaving the fusion classifier as the only trainable module:

1. **Stage 1 — Signal Tokenizer (VQ-VAE).** A VQ-VAE with EMA codebook updates and dead-code restart is pretrained on source-domain IQ signals to discretize the continuous waveform.
2. **Stage 2 — Prompted LLM Encoding (offline).** The frozen VQ-VAE tokenizes each sample; token IDs are offset into the Qwen vocabulary and embedded, prepended with a textual prompt prefix, and passed through the frozen LLM. The pooled context vector `ctx_vec` is cached to disk.
3. **Stage 3 — Fusion Classifier (trainable).** A patch-based IQ branch (PatchNet) is fused with the projected `ctx_vec` and trained with cross-entropy loss. Only `{IQ branch, projection head, fusion head}` receive gradients.

## Repository Structure

```
.
├── config.py                       # Centralized config (data / VQ-VAE / LLM / fusion / paths)
├── run_all.py                      # Main entry: runs Stage 1 → 2 → 3 in order
├── sweep.sh                        # Dual-GPU hyperparameter sweep (datasets × folds × llm_hidden)
├── backbones/
│   ├── signal_tokenizer.py         # VQ-VAE encoder/decoder (Conv1d patchify)
│   ├── vector_quantizer.py         # EMA vector quantizer with dead-code restart
│   └── fusion_classifier.py        # PatchNet IQ branch + LLM projection + fusion head
├── stages/
│   ├── train_vqvae.py              # Stage 1: VQ-VAE pretraining
│   ├── precompute_ctx.py           # Stage 2: offline ctx_vec precomputation
│   ├── train_fusion.py             # Stage 3: multimodal fusion training
│   ├── train_fusion_llmonly.py     # Ablation: LLM-only (w/o IQ branch)
│   └── train_fusion_noise.py       # Ablation: ctx_vec replaced by Gaussian noise
├── utils/
│   ├── context_encoder.py          # Frozen tokenizer + frozen Qwen → ctx_vec
│   └── load_data.py                # WiSig (ManySig / ManyRx) data loading
│
├── LLM/                            # Qwen3-Embedding-0.6B weights  → https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
├── dataset/                        # WiSig / LoRa datasets         → https://cores.ee.ucla.edu/downloads/datasets/wisig/
├── ctx_cache/                      # Precomputed ctx_vec (.npz)    → https://pan.baidu.com/s/11j93oF100J1sj3TkNivFsg?pwd=kvwd
├── weights_vqvae/                  # Stage-1 VQ-VAE checkpoints    → https://pan.baidu.com/s/11j93oF100J1sj3TkNivFsg?pwd=kvwd
├── weights_mm/                     # Stage-3 fusion checkpoints    → https://pan.baidu.com/s/11j93oF100J1sj3TkNivFsg?pwd=kvwd
└── logs/                           # Sweep / training logs         → https://pan.baidu.com/s/11j93oF100J1sj3TkNivFsg?pwd=kvwd
```

> **Note:** The directories `LLM/`, `dataset/`, `ctx_cache/`, `weights_vqvae/`, `weights_mm/`, and `logs/` are empty in this repository. Their contents (model weights, datasets, caches, and logs) are hosted on a netdisk — download and place them under the corresponding paths before running.

## Datasets

- **WiSig** — ManySig subset (6 Tx, 12 Rx) and ManyRx subset (10 Tx, 32 Rx). Receivers are split into 4 disjoint groups under a leave-one-group-out protocol; training uses the two earlier capture days, testing uses the two later days of the held-out group (cross-receiver + cross-day, i.e. the `CRD` protocol in `config.py`).
- **LoRa** — `receiver_drift_dataset` subset (10 devices), trained on N210 / tested on RTL-SDR across days. Available at [IEEE DataPort](https://ieee-dataport.org/documents/radio-frequency-fingerprint-lora-dataset-multiple-receivers).

## Usage

Run the full three-stage pipeline (each stage auto-skips if its outputs already exist):

```bash
python run_all.py --gpu 0
```

Run a single stage:

```bash
python run_all.py --stage 1 --gpu 0                          # VQ-VAE pretraining
python run_all.py --stage 2 --gpu 0                          # precompute ctx_vec
python run_all.py --stage 3 --gpu 0 --code_state train_test  # fusion training
```

Override key settings from the command line:

```bash
python run_all.py --stage 3 --dataset_name ManyRx --test_round 2 --llm_hidden 16
```

Reproduce the full sweep (2 datasets × 4 folds × 8 projection-bottleneck widths, dual-GPU):

```bash
bash sweep.sh
```

All other hyperparameters (patch size, codebook size `K`, commitment cost `β`, EMA decay, fusion dimension, etc.) are defined in `config.py`.
