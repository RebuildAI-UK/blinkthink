"""
Comprehensive offline test suite for BlinkThink core primitives,
calibration math, CoT parsing, and FastAPI endpoints.
"""

import pytest
import numpy as np
import torch
from fastapi.testclient import TestClient

from blinkthink.schemas import DecisionRequest, DecisionResult, DecisionType
from blinkthink.calibrator import TemperatureCalibrator
from blinkthink.benchmark import compute_ece
from blinkthink.system2 import ThinkEngine
from blinkthink.server import create_app


# ---------------------------------------------------------------------------
# 1. Pydantic Request Validation Tests
# ---------------------------------------------------------------------------
def test_decision_request_validation():
    # Valid choice request
    req = DecisionRequest(
        prompt="Select a color",
        decision_type=DecisionType.CHOICE,
        options=["Red", "Green", "Blue"],
    )
    assert req.decision_type == DecisionType.CHOICE
    assert len(req.options) == 3

    # Empty prompt fails
    with pytest.raises(ValueError):
        DecisionRequest(prompt="", options=["A", "B"])

    # Fewer than 2 options fails
    with pytest.raises(ValueError):
        DecisionRequest(prompt="Valid prompt", options=["OnlyOne"])


def test_binary_request_auto_options():
    # Legacy binary string should automatically configure ["Yes", "No"]
    req = DecisionRequest(
        prompt="Is this a test?",
        decision_type="binary",
    )
    assert req.decision_type == DecisionType.CHOICE
    assert req.options == ["Yes", "No"]


# ---------------------------------------------------------------------------
# 2. Calibration & ECE Metric Tests
# ---------------------------------------------------------------------------
def test_temperature_calibrator():
    calibrator = TemperatureCalibrator(temperature=2.0)
    assert calibrator.temperature == 2.0
    assert calibrator.to_dict() == {"choice": 2.0}

    # Verify softmax scaling
    logits = torch.tensor([[2.0, 4.0]])
    probs = calibrator.calibrate(logits)
    assert probs.shape == (1, 2)
    assert torch.isclose(probs.sum(), torch.tensor(1.0))

    # Synthetic calibration optimization
    synthetic_logits = torch.tensor([
        [1.0, 4.0],
        [0.5, 3.5],
        [0.2, 5.0],
        [4.0, 0.5],
    ])
    synthetic_targets = [1, 1, 1, 0]

    optimal_t = calibrator.fit(synthetic_logits, synthetic_targets)
    assert 0.1 <= optimal_t <= 10.0


def test_compute_ece():
    # 1. Perfect calibration: confidence exactly matches accuracy
    probs_perfect = np.array([0.9, 0.9, 0.1, 0.1])
    labels_perfect = np.array([1, 1, 0, 0])
    ece_good = compute_ece(probs_perfect, labels_perfect, n_bins=5)
    assert ece_good < 0.15

    # 2. Pathological calibration: 100% confidence, 0% accuracy
    probs_bad = np.array([1.0, 1.0, 1.0, 1.0])
    labels_bad = np.array([0, 0, 0, 0])
    ece_bad = compute_ece(probs_bad, labels_bad, n_bins=5)
    assert np.isclose(ece_bad, 1.0)


# ---------------------------------------------------------------------------
# 3. System 2 Reasoning Parse Tests (Edge Cases & Fallbacks)
# ---------------------------------------------------------------------------
def test_think_engine_parse_answer():
    # Dummy engine instance without loading heavy models
    engine = ThinkEngine(model=None, tokenizer=None)
    req = DecisionRequest(
        prompt="Identify the powerhouse of the cell.",
        options=["Golgi Apparatus", "Mitochondria", "Ribosome"],
    )

    # 1. Clean XML answer tag with letter
    trace1 = "<reasoning>ATP is synthesized in mitochondria.</reasoning>\n<answer>B</answer>"
    assert engine._parse_answer(trace1, req) == "Mitochondria"

    # 2. Tag with label prefix and formatting: e.g. <answer>B: Mitochondria</answer>
    trace2 = "<reasoning>Let's check choices...</reasoning>\n<answer>B: Mitochondria</answer>"
    assert engine._parse_answer(trace2, req) == "Mitochondria"

    # 3. Tag with option text inside: <answer>Mitochondria</answer>
    trace3 = "<reasoning>Analysis...</reasoning>\n<answer>Mitochondria</answer>"
    assert engine._parse_answer(trace3, req) == "Mitochondria"

    # 4. Tail deduction without explicit tags
    trace4 = "Examining the biology question carefully. Therefore, the correct option is B."
    assert engine._parse_answer(trace4, req) == "Mitochondria"

    # 5. Non-matching or ambiguous response returns None (triggering System 1 fallback)
    trace5 = "I cannot determine the answer from the available text."
    assert engine._parse_answer(trace5, req) is None


# ---------------------------------------------------------------------------
# 4. FastAPI Decision Service Tests
# ---------------------------------------------------------------------------
def test_fastapi_endpoints():
    class MockEngine:
        def decide(self, request, n_blinks=None, threshold=None, force_mode=None):
            return DecisionResult(
                decision=request.options[0],
                confidence=0.95,
                system_used="system_1",
                latency_ms=12.5,
                probabilities={opt: (0.95 if i == 0 else 0.05) for i, opt in enumerate(request.options)},
            )

        async def adecide(self, *args, **kwargs):
            return self.decide(*args, **kwargs)

        async def achoice(self, prompt, options, metadata=None, n_blinks=None, threshold=None, force_mode=None):
            req = DecisionRequest(prompt=prompt, options=options, metadata=metadata)
            return self.decide(req, n_blinks=n_blinks, threshold=threshold, force_mode=force_mode)

        async def abinary(self, prompt, options=None, metadata=None, n_blinks=None, threshold=None, force_mode=None):
            return await self.achoice(prompt, options or ["Yes", "No"], metadata, n_blinks, threshold, force_mode)

    app = create_app(engine=MockEngine())
    client = TestClient(app)

    # Health check
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["engine_ready"] is True

    # Multi-choice decide endpoint
    choice_resp = client.post(
        "/v1/decide/choice",
        json={"prompt": "Pick one", "options": ["Option A", "Option B"], "threshold": 0.4},
    )
    assert choice_resp.status_code == 200
    assert choice_resp.json()["decision"] == "Option A"
    assert choice_resp.json()["confidence"] == 0.95

    # Binary decide endpoint
    binary_resp = client.post(
        "/v1/decide/binary",
        json={"prompt": "Is the sky blue?"},
    )
    assert binary_resp.status_code == 200
    assert binary_resp.json()["decision"] == "Yes"
