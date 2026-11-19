#!/usr/bin/env python3
"""
Example 2: Deploying the BlinkThink REST Service.
Demonstrates starting the FastAPI service programmatically or via standard CLI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from blinkthink import BlinkThink
from blinkthink.server import create_app

# Set environment authentication key (optional)
os.environ["BLINKTHINK_API_KEY"] = "sk_live_decision_demo_token"


def start_custom_server():
    model_id = os.getenv("BLINKTHINK_MODEL_ID", "google/gemma-4-12B-it")
    print(f"Initializing model {model_id} for decision service...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    engine = BlinkThink(model=model, tokenizer=tokenizer, threshold=0.35)

    # create_app injects the preloaded engine into the FastAPI endpoints
    app = create_app(engine=engine)

    import uvicorn
    print("Serving decision API on http://0.0.0.0:8000 (OpenAPI docs at /docs)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    start_custom_server()
