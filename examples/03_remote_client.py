#!/usr/bin/env python3
"""
Example 3: Remote HTTP Client Usage (with health check and timeout configuration).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import httpx
from blinkthink import AsyncBlinkThinkRemoteClient, BlinkThinkRemoteClient

BASE_URL = "http://localhost:8000"
API_KEY = "sk_live_decision_demo_token"
TIMEOUT = 180.0  # Allow up to 3 minutes for System 2 CoT generation


def check_server_readiness():
    try:
        resp = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        data = resp.json()
        if not data.get("engine_ready"):
            print("Server is still initializing model weights. Please wait...")
            sys.exit(1)
        print(" Server is healthy and engine is ready.")
    except Exception as e:
        print(f" Cannot reach server at {BASE_URL}: {e}")
        print("Ensure examples/02_fastapi_server.py is running.")
        sys.exit(1)


def run_sync_example():
    print("\n=== 1. Synchronous Remote Client ===")
    with BlinkThinkRemoteClient(base_url=BASE_URL, api_key=API_KEY, timeout=TIMEOUT) as client:
        # A. Fast System 1 path (guaranteed sub-second)
        print("Running Fast-Path Decision (force_mode='blink_only')...")
        fast_res = client.choice(
            prompt="What is the chemical symbol for Gold?",
            options=["Ag", "Fe", "Au", "Cu"],
            force_mode="blink_only",
        )
        print(f"Decision:   {fast_res.decision}")
        print(f"Confidence: {fast_res.confidence:.3f}")
        print(f"System:     {fast_res.system_used} ({fast_res.latency_ms:.1f} ms)")

        # B. Dual-process decision (may escalate to System 2)
        print("\nRunning Dual-Process Decision (threshold=0.35)...")
        result = client.choice(
            prompt="Classify the incident ticket: 'All payment webhooks returning HTTP 500.'",
            options=["P1 - Outage", "P2 - High", "P3 - Moderate", "P4 - Informational"],
            threshold=0.35,
        )
        print(f"Decision:   {result.decision}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"System:     {result.system_used} ({result.latency_ms:.1f} ms)")
        if result.reasoning_trace:
            print(f"Reasoning:\n{result.reasoning_trace[:200]}...")


async def run_async_example():
    print("\n=== 2. Asynchronous Remote Client ===")
    async with AsyncBlinkThinkRemoteClient(base_url=BASE_URL, api_key=API_KEY, timeout=TIMEOUT) as async_client:
        queries = [
            "Is Python dynamically typed?",
            "Can sound travel in a vacuum?",
            "Is the Atlantic Ocean larger than the Pacific Ocean?",
        ]

        # Use force_mode='blink_only' for concurrent queries so they don't block each other
        print("Dispatching concurrent binary verifications...")
        tasks = [async_client.binary(q, force_mode="blink_only") for q in queries]
        results = await asyncio.gather(*tasks)

        for q, res in zip(queries, results):
            print(f"Query:  '{q}' -> {res.decision} ({res.system_used}, {res.latency_ms:.1f}ms)")


def main():
    check_server_readiness()
    run_sync_example()
    asyncio.run(run_async_example())


if __name__ == "__main__":
    main()
