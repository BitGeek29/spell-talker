# spell-talker

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-spell--talker--42m--dense-yellow)](https://huggingface.co/BitGeek29/spell-talker-42m-dense)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

Training and inference pipeline for **spell-talker**, a lightweight 41.8M-parameter language model built from scratch in PyTorch for reasoning and task comprehension.

**Pretrained Weights:** [huggingface.co/BitGeek29/spell-talker-42m-dense](https://huggingface.co/BitGeek29/spell-talker-42m-dense)

---

## Model Specifications

| Parameter | Value |
| :--- | :--- |
| **Total Parameters** | 41,821,696 (~41.8M, weight-tied embeddings) |
| **Architecture** | Decoder-only Transformer (Pre-norm) |
| **Layers (`n_layer`)** | 8 |
| **Attention Heads (`n_head`)** | 8 (`head_dim = 64`, FlashAttention via `F.scaled_dot_product_attention`) |
| **Hidden Dimension (`n_embd`)** | 512 |
| **Feedforward Network** | SwiGLU (`4x` hidden expansion, no bias) |
| **Normalization** | RMSNorm (`eps=1e-6`) |
| **Positional Encoding** | Learned absolute embeddings (`block_size = 512`) |
| **Vocabulary Size** | 32,000 |

---

## Installation

```bash
git clone https://github.com/BitGeek29/spell-talker.git
cd spell-talker
pip install torch safetensors huggingface_hub

```

---

## Quickstart (Load from Hugging Face)

Download and run the weights directly from [Hugging Face](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/BitGeek29/spell-talker-42m-dense):

```python
import torch
from src.model import CustomGPT

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model = CustomGPT.from_pretrained("BitGeek29/spell-talker-42m-dense").to(device)
model.eval()

# Example forward pass with dummy token IDs
idx = torch.randint(0, 32000, (1, 16), device=device)
with torch.no_grad():
    logits = model(idx)

print(logits.shape)  # torch.Size([1, 16, 32000])

```

---

## Exporting `.ckpt` Checkpoints to Hugging Face

If you have a local `.ckpt` checkpoint, convert it safely on CPU (handling tied weights automatically via `PyTorchModelHubMixin`) and push it to the Hub:

```python
import torch
from src.model import CustomGPT

model = CustomGPT()
state_dict = torch.load("model.ckpt", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)

model.push_to_hub("BitGeek29/spell-talker-42m-dense")

```

---

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](https://www.google.com/search?q=LICENSE) for details.