# BlinkThink

[![PyPI version](https://img.shields.io/pypi/v/blinkthink.svg)](https://pypi.org/project/blinkthink/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24406872.svg)](https://doi.org/10.5281/zenodo.24406872)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-green)](https://pydantic.dev/)

**Training-Free Calibrated Dual-Process Routing for Open-Weight Language Models.**

BlinkThink implements Daniel Kahneman's **System 1 / System 2** cognitive paradigm for discrete LLM decisions (classification, triage, assertion verification, and multiple-choice QA).

- **System 1 (`BlinkEngine`):** Single batched prefill pass with $N$ cyclic option permutations (`0` autoregressively decoded tokens, ~130–240ms on 4B–12B NF4 models).
- **System 2 (`ThinkEngine`):** Deliberative Chain-of-Thought (CoT) with XML tag delimitation and early exit, triggered selectively only when System 1 confidence falls below a validation-tuned routing threshold $\tau^*$.

---

## Architecture Overview

```
                      ┌────────────────────────────┐
                      │    Incoming Decision       │
                      │    (Choice / Binary)       │
                      └─────────────┬──────────────┘
                                    │
                                    ▼
                      ┌────────────────────────────┐
                      │    System 1: BlinkEngine   │
                      │  • N cyclic permutations   │
                      │  • Single batched prefill  │
                      │  • Temperature scaling T*  │
                      └─────────────┬──────────────┘
                                    │
                         Confidence >= tau* ?
                                    │
                     ┌──────────────┴──────────────┐
              Yes    │                             │   No
         (Fast Path) ▼                             ▼ (Deliberate Path)
        ┌─────────────────────────┐   ┌───────────────────────────┐
        │ Instant Return          │   │  System 2: ThinkEngine    │
        │ • 0 decoded tokens      │   │  • Autoregressive CoT     │
        │ • Sub-second prefill    │   │  • StopStringCriteria     │
        │ • Calibrated probs      │   │  • Calibrated token probs │
        └─────────────────────────┘   └─────────────┬─────────────┘
                                                    │
                                                    ▼
                                      ┌───────────────────────────┐
                                      │ Final Decision & Trace    │
                                      └───────────────────────────┘
```

---

## Installation

Install from PyPI:

```bash
# Core package (local inference & remote HTTP client)
pip install blinkthink

# With 4-bit/8-bit quantization support
pip install "blinkthink[quantization]"

# With FastAPI server, benchmarks, and quantization
pip install "blinkthink[all]"
```

Or install from source for development:

```bash
git clone https://github.com/RebuildAI-UK/blinkthink.git
cd blinkthink
pip install -e ".[all,dev]"
```

---

## Quick Example

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

client = BlinkThink(model=model, tokenizer=tokenizer, threshold=0.35, n_blinks=2)

# 1. Multi-Choice Classification (with optional per-call threshold / blink overrides)
res = client.choice(
    prompt="Which organelle is responsible for cellular ATP synthesis?",
    options=["Golgi Apparatus", "Mitochondria", "Ribosome", "Endoplasmic Reticulum"],
    n_blinks=2,        # Override blink count per-decision
    threshold=0.35,    # Override escalation cutoff per-decision
)
print(f"Decision:   {res.decision}")
print(f"Confidence: {res.confidence:.3f}")
print(f"System:     {res.system_used} ({res.latency_ms:.1f}ms)")

# 2. Binary Assertion Verification
guard = client.binary(
    prompt="Is light faster in glass than in vacuum?",
    threshold=0.40,
)
print(f"Decision:   {guard.decision}")  # "No"
```

---

## Benchmark Highlights

Evaluated under 4-bit NF4 quantization on MMLU ($K=4$, $M_{\text{test}}=1000$) and BoolQ ($K=2$, $M_{\text{test}}=800$):

| Model | Dataset | Mode | Acc (%) | Latency | Sys 2 Esc (%) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `google/gemma-4-12B-it` | MMLU | CoT Baseline | 79.4 | 57.80 s | 100.0 |
| `google/gemma-4-12B-it` | MMLU | **BlinkThink ($N=2$, Balanced)** | **78.9** | **19.16 s** (-66.8%) | **28.9** |
| `google/gemma-4-12B-it` | BoolQ | CoT Baseline | 89.9 | 27.45 s | 100.0 |
| `google/gemma-4-12B-it` | BoolQ | **BlinkThink ($N=2$, Balanced)** | **89.5** | **3.94 s** (-85.6%) | **8.8** |
| `Qwen/Qwen3.5-4B` | MMLU | CoT Baseline | 43.7 | 46.15 s | 100.0 |
| `Qwen/Qwen3.5-4B` | MMLU | **BlinkThink ($N=4$, Cyclic)** | **68.4** | **190.4 ms** | **0.0** |
| `Qwen/Qwen3.5-4B` | BoolQ | CoT Baseline | 86.4 | 44.48 s | 100.0 |
| `Qwen/Qwen3.5-4B` | BoolQ | **BlinkThink ($N=2$, Pure S1)** | **86.1** | **149.1 ms** | **0.0** |

---

## Guides & Documentation

- **[Quickstart Guide](docs/QUICKSTART.md):** Setup, local usage, and production REST API deployment.
- **[Decision API Guide](docs/DECISION_API_GUIDE.md):** Mathematical formulas, permutation invariance, temperature calibration, and optimal $\tau^*$ routing theory.
- **[Examples Directory](examples/):**
  - `01_in_process_sdk.py`: In-memory GPU SDK usage.
  - `02_fastapi_server.py`: FastAPI OpenAPI REST deployment.
  - `03_remote_client.py`: Sync & Async HTTP client usage.

---

## Running Benchmarks

Run the built-in study harness across curated or standard Hugging Face datasets:

```bash
# 1. Single study run (fast curated sanity test)
python benchmarks/run_study.py --dataset curated

# 2. Single study run on MMLU
python benchmarks/run_study.py --dataset mmlu --val_size 50 --test_size 200 --tune_objective balanced

# 3. Run complete matrix across default models and all objectives
python benchmarks/run_study_matrix.py --run --dataset curated

# 4. Run matrix across custom models on MMLU
python benchmarks/run_study_matrix.py --run --models Qwen/Qwen3.5-4B google/gemma-4-12B-it --dataset mmlu

# 5. Inspect and compare existing results (auto-detects datasets)
python benchmarks/run_study_matrix.py
```

---

## Citation

If you use BlinkThink in your research, please cite our preprint:

```bibtex
@misc{petnehazi2026blinkthink,
  title={BlinkThink: Training-Free Calibrated Dual-Process Routing for Discrete LLM Decisions},
  author={Petnehazi, Gabor},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.24406872},
  url={https://doi.org/10.5281/zenodo.24406872}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
