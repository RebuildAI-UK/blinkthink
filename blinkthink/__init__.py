"""
BlinkThink: Calibrated Dual-Process Routing for Open-Weight Language Models.
"""

from .benchmark import BenchmarkReport, BlinkStudyEvaluator, compute_ece
from .calibrator import TemperatureCalibrator
from .client import BlinkThink
from .remote import AsyncBlinkThinkRemoteClient, BlinkThinkRemoteClient
from .router import BlinkThinkRouter
from .system1 import BlinkEngine
from .system2 import ThinkEngine
from .tuner import BlinkThinkTuner, TuningReport
from .schemas import (
    BlinkTelemetry,
    DecisionRequest,
    DecisionResult,
    DecisionType,
)

__version__ = "1.1.1"

__all__ = [
    "BlinkThink",
    "BlinkThinkRemoteClient",
    "AsyncBlinkThinkRemoteClient",
    "BlinkThinkRouter",
    "BlinkEngine",
    "ThinkEngine",
    "BlinkThinkTuner",
    "TemperatureCalibrator",
    "BlinkStudyEvaluator",
    "DecisionRequest",
    "DecisionResult",
    "DecisionType",
    "BlinkTelemetry",
    "BenchmarkReport",
    "TuningReport",
    "compute_ece",
]
