# Spell-Talker MoE (`spell-talker-moe`)

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-spell--talker--117m--a67m--moe-yellow)](https://huggingface.co/BitGeek29/spell-talker-117m-a67m-moe)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)


A compact Mixture-of-Experts (MoE) autoregressive language model designed for multi-step reasoning and task comprehension under tight compute budgets. Built with `CustomMoEGPT`, it replaces dense feed-forward blocks with top-2 routed SwiGLU experts while keeping active inference parameters compact.

**Pretrained Weights:** [huggingface.co/BitGeek29/spell-talker-117m-a67m-moe](https://huggingface.co/BitGeek29/spell-talker-117m-a67m-moe)


## Model Architecture & Specs

| Specification | Value |
| :--- | :--- |
| **Architecture** | `CustomMoEGPT` (Decoder-only Transformer with Sparse MoE) |
| **Total Parameters** | ~117.3M (`117,300,000`) |
| **Active Parameters / Token** | ~67.0M (`67,000,000`) |
| **Layers** | 8 |
| **Attention Heads** | 8 |
| **Hidden Size (`d_model`)** | 512 |
| **Context Length** | 512 tokens |
| **Vocabulary Size** | 32,000 |
| **Experts per Layer** | 4 SwiGLU experts (Top-2 routing) |
| **Normalization & Embeddings** | RMSNorm, Weight-tied input/output embeddings |
| **Forward Output Signature** | `(logits, total_aux_loss)` |

## Quickstart & Loading

Because this model uses a custom PyTorch architecture (`CustomMoEGPT`) rather than a standard `transformers` class, load the checkpoint using your model definition from the companion code repository:

```python
import torch
from huggingface_hub import hf_hub_download
# Import your CustomMoEGPT definition from the spell-talker codebase
from model import CustomMoEGPT, ModelConfig

# 1. Download checkpoint from Hugging Face
ckpt_path = hf_hub_download(
    repo_id="BitGeek29/spell-talker-117m-a67m-moe",
    filename="spell-talker-117m-a67m-moe.ckpt",
)

# 2. Initialize model and load weights on CPU first to prevent device mismatch
config = ModelConfig(
    vocab_size=32000,
    n_layer=8,
    n_head=8,
    n_embd=512,
    block_size=512,
    num_experts=4,
    top_k=2,
)
model = CustomMoEGPT(config)

checkpoint = torch.load(ckpt_path, map_location="cpu")
state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
model.load_state_dict(state_dict, strict=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

# 3. Forward pass (note: CustomMoEGPT returns both logits and auxiliary routing loss)
input_ids = torch.randint(0, 32000, (1, 16), device=device)
with torch.no_grad():
    logits, total_aux_loss = model(input_ids)

```

## Training Details

* **Routing Objective:** Top-2 token-level expert routing paired with a load-balancing auxiliary loss (`total_aux_loss`) to prevent expert collapse.
* **Comparison to Dense Baseline:** Expands total capacity from **41.8M** (`spell-talker-42m-dense`) to **117.3M** parameters while keeping per-token compute comparable at **67.0M** active parameters.

## Limitations

* **Custom Architecture:** Requires the `CustomMoEGPT` class definition; `AutoModelForCausalLM.from_pretrained()` is not supported out of the box.
* **Context Window:** Trained with a 512-token context window; sequences beyond 512 tokens must be truncated or chunked.
* **Checkpoint Format:** Distributed as a PyTorch `.ckpt` file (`pickle` serialization). Always inspect or load from trusted sources.
