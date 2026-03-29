# QLoRA Fine-tuning of Llama on Medical Reasoning

Fine-tuning `meta-llama/Llama-3.2-1B-Instruct` on the `OpenMed/Medical-Reasoning-SFT-Mega` dataset using 4-bit quantization and LoRA adapters — designed to run on a single T4 GPU (Google Colab free tier).

---

## What is LoRA?

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning technique. Instead of updating all model weights during training, LoRA freezes the original model and injects small trainable low-rank matrices into specific layers (typically the attention projections). The weight update `ΔW` is decomposed as:

```
ΔW = A × B   where A ∈ R^(d×r), B ∈ R^(r×k), r << d
```

Only `A` and `B` are trained, drastically reducing the number of trainable parameters. After training, the adapter weights can be merged back into the base model with zero inference overhead.

**Benefits of LoRA:**
- Trains ~1–10% of total parameters instead of 100%
- Drastically lower GPU memory usage
- Faster training iterations
- Adapter files are small (MBs vs GBs for full models)
- Multiple task-specific adapters can share one base model

---

## What is QLoRA?

**QLoRA (Quantized LoRA)** combines LoRA with 4-bit quantization of the base model weights. The base model is loaded in 4-bit NF4 (NormalFloat4) precision using `bitsandbytes`, while LoRA adapter weights are trained in 16-bit (bfloat16). Gradients flow through the frozen quantized model into the trainable low-rank adapters.

**Benefits of QLoRA over LoRA:**
- Reduces base model memory footprint by ~4x compared to full precision
- Enables fine-tuning 7B+ models on a single 16GB GPU; 1B–3B models on a free Colab T4
- No significant accuracy degradation compared to full fine-tuning
- Makes large model fine-tuning accessible without expensive hardware

---

## Libraries Used

| Library | Version | Role in Fine-tuning |
|---|---|---|
| `transformers` | 4.46.3 | Core library for loading pre-trained Llama models, tokenizers, and generation utilities |
| `peft` | 0.13.2 | Implements LoRA adapters via `LoraConfig` and `get_peft_model()`; wraps the base model with trainable low-rank layers |
| `trl` | 0.12.2 | Provides `SFTTrainer` (Supervised Fine-Tuning Trainer), a high-level training loop built on top of Hugging Face Trainer, handling chat-template formatting and sequence packing |
| `bitsandbytes` | 0.45.3 | Enables 4-bit NF4 quantization of the base model via `BitsAndBytesConfig`; also provides the `paged_adamw_8bit` optimizer |
| `accelerate` | latest | Backend for device management and mixed-precision training; handles `device_map="auto"` to distribute model layers across available hardware |
| `datasets` | latest | Loads and preprocesses the `OpenMed/Medical-Reasoning-SFT-Mega` dataset from the Hugging Face Hub with streaming support |
| `huggingface_hub` | latest | Handles authentication (`login()`) and pushing trained adapters to the HF Hub |
| `wandb` | optional | Experiment tracking — logs training loss, learning rate, and hyperparameters to Weights & Biases |
| `mlflow` | optional | Alternative experiment tracking for local logging |
| `pandas` | latest | Creates before/after comparison tables of model outputs |
| `ollama` | optional | Lightweight local inference server for running the fine-tuned model via REST API |

---

## Step-by-Step Walkthrough

### Step 1 — Check GPU & Runtime
Verifies that a CUDA-capable GPU is available using `nvidia-smi`, and checks available disk space and RAM. Warns if no GPU is detected, as 4-bit quantization requires CUDA.

### Step 2 — Install Dependencies
Installs the exact pinned versions of all required libraries. The `bitsandbytes` CUDA128 wheel is installed explicitly to match the Colab runtime's CUDA version. After installation, the runtime must be restarted before proceeding.

### Step 3 — Configuration
All training hyperparameters and flags are set in one cell for easy editing:

- **Model**: `meta-llama/Llama-3.2-1B-Instruct`
- **Dataset**: `OpenMed/Medical-Reasoning-SFT-Mega` (100-sample subset by default)
- **LoRA rank** (`LORA_R`): 16 — controls adapter capacity
- **LoRA alpha** (`LORA_ALPHA`): 32 — scaling factor for adapter updates
- **Target modules**: all attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Training**: 2 epochs, batch size 2, gradient accumulation 8 (effective batch 16), learning rate 2e-4, cosine scheduler

### Step 4 — HuggingFace Authentication
Logs into Hugging Face Hub to access the gated Llama model. Tries Colab Secrets first, then falls back to an interactive token prompt.

### Step 5 — Load & Preprocess Dataset
Loads the medical reasoning dataset with streaming. Takes a 100-sample subset (configurable) and performs a 90/10 train/eval split. Each sample is formatted into a chat template string using the tokenizer's `apply_chat_template()`, normalizing various column name schemas (`instruction`/`output`, `question`/`answer`, `messages`, etc.) into a single `text` field.

A system prompt is injected into every example:
```
You are a helpful medical assistant. Provide clear, accurate, and concise medical information...
```

### Step 6 — Load Base Model in 4-bit (QLoRA)
Configures `BitsAndBytesConfig` for 4-bit NF4 quantization with double quantization enabled and bfloat16 compute dtype. Loads the tokenizer and base model with `device_map="auto"`, placing layers automatically across GPU/CPU. Sets `pad_token = eos_token` to handle variable-length sequences.

### Step 7 — Baseline Inference (Before Training)
Runs inference on 5 benchmark medical questions before any fine-tuning. This establishes a baseline to compare against after training. Questions cover: diabetes management, cardiac events, antibiotic selection, lupus diagnosis, and meningitis differentiation.

### Step 8 — Apply LoRA Adapters
Calls `prepare_model_for_kbit_training()` to enable gradient checkpointing on the quantized model, then wraps it with `LoraConfig` via `get_peft_model()`. Only the low-rank adapter matrices (~1–2% of total parameters) are marked as trainable; all base model weights remain frozen.

### Step 9 — Experiment Tracking (Optional)
Optionally initializes Weights & Biases or MLflow to log the training run. Logs hyperparameters including model name, dataset, LoRA rank, number of epochs, and batch size.

### Step 10 — Fine-tune with SFTTrainer
Constructs an `SFTConfig` with:
- Gradient checkpointing to reduce memory usage
- `paged_adamw_8bit` optimizer for memory-efficient Adam
- `fp16=True` for T4 GPUs (falls back to `bf16` on A100)
- Evaluation every 100 steps
- `group_by_length=True` to minimize padding waste

Instantiates `SFTTrainer` with the model, tokenizer, formatted train/eval datasets, and config, then calls `trainer.train()`.

### Step 11 — Save LoRA Adapter
Saves the trained LoRA adapter weights and tokenizer to `./llama-medical-qlora/final_adapter/`. The adapter is only a few MB — the base model weights are not saved again.

### Step 12 — After Training Inference & Comparison
Re-runs the same 5 benchmark questions with the fine-tuned model. Displays a side-by-side before/after comparison in the notebook output and a pandas DataFrame showing answer lengths.

### Step 13 — Merge LoRA Weights (Optional)
If `MERGE_AND_SAVE=True`, loads the adapter back and calls `merge_and_unload()` to fuse the LoRA weights into the base model, producing a standard (non-PEFT) model saved to `./llama-medical-qlora/merged_model/`. This is required for deployment with Ollama.

### Step 14 — Push to HuggingFace Hub (Optional)
If `PUSH_TO_HUB=True`, uploads the adapter (or merged model) and tokenizer to a private repository on the HF Hub under `{HF_USERNAME}/{ADAPTER_NAME}`.

### Step 15 — Run with Ollama (Optional)
Installs Ollama, starts the inference server, and queries the fine-tuned model via its REST API on all 5 benchmark questions. Provides an alternative deployment path to the HF Hub for lightweight local inference.

---

## Hardware Requirements

| Setup | Minimum GPU VRAM |
|---|---|
| 1B model, 4-bit | 6 GB (T4 free tier) |
| 3B model, 4-bit | 10 GB (T4 free tier) |
| 7B model, 4-bit | 16 GB (A10G / V100) |

---

## Quick Start

1. Open `llama_qlora_medical_finetune_trial.ipynb` in Google Colab
2. Set runtime to **T4 GPU**
3. Accept the Llama license on [huggingface.co/meta-llama](https://huggingface.co/meta-llama) and create a read token
4. Add your HF token to Colab Secrets as `HF_TOKEN`
5. Run cells top to bottom, restarting the runtime after the install cell (Step 2)


If you face any issues or would like to contribute:
Email: `aakashaldankar@gmail.com`
