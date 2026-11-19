#!/usr/bin/env python3
"""
Example 1: In-Process SDK Usage (Local GPU / PyTorch).
Demonstrates zero-copy local inference with choice and binary decision routing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from blinkthink import BlinkThink


def main():
    model_id = "google/gemma-4-12B-it"
    print(f"Loading {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Initialize BlinkThink client
    client = BlinkThink(
        model=model,
        tokenizer=tokenizer,
        threshold=0.35,  # tau: routing cutoff
        n_blinks=2,      # 2 batched cyclic permutations for System 1
    )

    print("\n--- 1. Categorical Choice Primitive ---")
    res1 = client.choice(
        prompt="Which organelle is responsible for cellular ATP synthesis?",
        options=["Golgi Apparatus", "Mitochondria", "Ribosome", "Endoplasmic Reticulum"],
    )
    print(f"Decision:     {res1.decision}")
    print(f"Confidence:   {res1.confidence:.4f}")
    print(f"System:       {res1.system_used}")
    print(f"Latency:      {res1.latency_ms:.1f} ms")
    if res1.telemetry:
        print(f"S1 Consensus: {res1.telemetry.consensus_rate * 100:.0f}% (Drift: {res1.telemetry.drift:.4f})")

    print("\n--- 2. Binary Verification (2-Option Choice) ---")
    res2 = client.binary(
        prompt="Context: The speed of light in vacuum is approx 300,000 km/s.\nIs light faster in glass than in vacuum?",
    )
    print(f"Decision:     {res2.decision}")
    print(f"Confidence:   {res2.confidence:.4f}")
    print(f"System:       {res2.system_used}")
    print(f"Probabilities:{res2.probabilities}")

    print("\n--- 3. Forcing System 2 Deliberate Reasoning ---")
    res3 = client.choice(
        prompt="If all Bloops are Razzies and some Razzies are Lurgs, are all Bloops definitely Lurgs?",
        options=["Yes, conclusively", "No, not necessarily"],
        force_mode="deliberate",
    )
    print(f"Decision:     {res3.decision}")
    print(f"System:       {res3.system_used}")
    print(f"Latency:      {res3.latency_ms:.1f} ms")
    print(f"Trace:\n{res3.reasoning_trace}")


if __name__ == "__main__":
    main()
