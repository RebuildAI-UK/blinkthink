# BlinkThink Quickstart Guide

## 1. Installation

```bash
# Minimal installation
pip install blinkthink

# Full installation (FastAPI server, Quantization, Benchmark Harness)
pip install "blinkthink[all]"
```

## 2. In-Process Python Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from blinkthink import BlinkThink

model_id = "google/gemma-4-12B-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Initialize Dual-Process Router
bt = BlinkThink(model=model, tokenizer=tokenizer, threshold=0.35, n_blinks=2)

# Multi-class Categorical Decision
res = bt.choice(
    prompt="Which planet has the highest surface temperature in the solar system?",
    options=["Mercury", "Venus", "Mars", "Jupiter"],
)
print(f"Decision:   {res.decision}")      # Venus
print(f"System:     {res.system_used}")  # system_1 (0 decoded tokens)
print(f"Confidence: {res.confidence:.3f}")
print(f"Latency:    {res.latency_ms:.1f}ms")
```

## 3. Serving via REST API

Start the server using the built-in CLI:

```bash
export BLINKTHINK_MODEL_ID="google/gemma-4-12B-it"
export BLINKTHINK_API_KEY="your-secret-key"
blinkthink-serve
```

Or run via Uvicorn:
```bash
uvicorn blinkthink.server:app --host 0.0.0.0 --port 8000
```
