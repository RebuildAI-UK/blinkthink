from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .router import BlinkThinkRouter
from .schemas import DecisionRequest


@dataclass
class BenchmarkReport:
    config_name: str
    n_blinks: int
    threshold: float
    accuracy: float
    avg_latency_ms: float
    system2_escalation_pct: float
    expected_calibration_error: float
    avg_drift: float
    avg_consensus: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE).
    """
    if len(probs) == 0:
        return 0.0
    probs = np.clip(probs, 0.0, 1.0)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (
            (probs >= bin_lower) & (probs <= bin_upper)
            if i == 0
            else (probs > bin_lower) & (probs <= bin_upper)
        )
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return float(ece)


class BlinkStudyEvaluator:
    def __init__(self, router: BlinkThinkRouter):
        self.router = router

    def evaluate_dataset(
        self,
        dataset: List[Tuple[DecisionRequest, Any]],
        config_name: str = "BlinkThink",
        n_blinks: int = 2,
        threshold: float = 0.35,
        force_mode: Optional[str] = None,
    ) -> BenchmarkReport:
        correct: List[int] = []
        latencies: List[float] = []
        escalated: List[int] = []
        confidences: List[float] = []
        drifts: List[float] = []
        consensuses: List[float] = []

        for req, y_true in dataset:
            res = self.router.decide(
                req,
                n_blinks=n_blinks,
                threshold=threshold,
                force_mode=force_mode,
            )
            is_correct = (res.decision == y_true)

            correct.append(1 if is_correct else 0)
            latencies.append(res.latency_ms)
            escalated.append(1 if res.system_used in ("system_2", "system_1_fallback") else 0)

            top_prob = (
                res.probabilities.get(res.decision, res.confidence)
                if res.probabilities
                else res.confidence
            )
            confidences.append(top_prob)

            if res.telemetry:
                drifts.append(res.telemetry.drift)
                consensuses.append(res.telemetry.consensus_rate)

        ece = compute_ece(np.array(confidences), np.array(correct))

        return BenchmarkReport(
            config_name=config_name,
            n_blinks=n_blinks,
            threshold=threshold,
            accuracy=float(np.mean(correct)) if correct else 0.0,
            avg_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
            system2_escalation_pct=float(np.mean(escalated) * 100.0) if escalated else 0.0,
            expected_calibration_error=ece,
            avg_drift=float(np.mean(drifts)) if drifts else 0.0,
            avg_consensus=float(np.mean(consensuses)) if consensuses else 1.0,
        )
