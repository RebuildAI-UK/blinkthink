"""
BlinkThink Core Data Structures and Pydantic Decision Primitives.
"""

from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from pydantic import BaseModel, Field, model_validator


class DecisionType(str, Enum):
    CHOICE = "choice"

    @classmethod
    def from_str(cls, value: str) -> "DecisionType":
        val = value.lower().strip()
        if val in ("choice", "multichoice", "mcq", "binary", "bool", "boolean", "predicate"):
            return cls.CHOICE
        raise ValueError(f"Unsupported decision type: '{value}'. Choose from {list(cls)}")

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            try:
                return cls.from_str(value)
            except ValueError:
                pass
        return super()._missing_(value)


class DecisionRequest(BaseModel):
    prompt: str = Field(..., description="The query or context requiring a decision.")
    decision_type: DecisionType = Field(
        default=DecisionType.CHOICE,
        description="Decision type (standardized to 'choice')."
    )
    options: List[str] = Field(..., description="Candidate list for choice decisions.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Arbitrary metadata attached to the request.")
    n_blinks: Optional[int] = Field(default=None, description="Optional override for number of blinks.")
    threshold: Optional[float] = Field(default=None, description="Optional override for routing confidence threshold.")
    force_mode: Optional[str] = Field(default=None, description="Optional override: 'blink_only', 'deliberate', etc.")
    confidence_metric: Optional[str] = Field(default=None, description="Confidence metric formulation: 'composite', 'margin', or 'msp'.")

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        prompt: str = ...,
        decision_type: Union[DecisionType, str] = DecisionType.CHOICE,
        options: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        n_blinks: Optional[int] = None,
        threshold: Optional[float] = None,
        force_mode: Optional[str] = None,
        confidence_metric: Optional[str] = None,
        **extra_data: Any,
    ):
        data: Dict[str, Any] = {**extra_data}
        if prompt is not ...:
            data["prompt"] = prompt
        if decision_type is not ...:
            data["decision_type"] = decision_type
        if options is not None:
            data["options"] = options
        if metadata is not None:
            data["metadata"] = metadata
        if n_blinks is not None:
            data["n_blinks"] = n_blinks
        if threshold is not None:
            data["threshold"] = threshold
        if force_mode is not None:
            data["force_mode"] = force_mode
        if confidence_metric is not None:
            data["confidence_metric"] = confidence_metric
        super().__init__(**data)

    @model_validator(mode="before")
    @classmethod
    def validate_request_inputs(cls, data: Any) -> Any:
        if isinstance(data, dict):
            prompt = data.get("prompt", "")
            d_type = data.get("decision_type", DecisionType.CHOICE)
            options = data.get("options")

            if not prompt or not str(prompt).strip():
                raise ValueError("DecisionRequest prompt cannot be empty.")

            raw_type_str = str(d_type).lower().strip()
            if isinstance(d_type, str):
                d_type = DecisionType.from_str(d_type)
                data["decision_type"] = d_type

            # Allow legacy binary declarations to automatically adopt default Yes/No options
            if not options and raw_type_str in ("binary", "bool", "boolean", "predicate"):
                options = ["Yes", "No"]
                data["options"] = options

            if not options or len(options) < 2:
                raise ValueError("DecisionType.CHOICE requires at least 2 candidate options.")
        return data


class BlinkTelemetry(BaseModel):
    n_blinks: int
    blink_decisions: List[str]
    consensus_rate: float
    drift: float
    raw_margins: List[float]
    canonical_distributions: List[Dict[str, float]]


T = TypeVar("T", bound=str)


class DecisionResult(BaseModel, Generic[T]):
    decision: T = Field(..., description="The final selected option string.")
    confidence: float = Field(..., description="Calibrated dual-process confidence metric [0.0, 1.0].")
    system_used: str = Field(..., description="'system_1' (fast blink) or 'system_2' (deliberate CoT).")
    probabilities: Dict[str, float] = Field(default_factory=dict, description="Calibrated class probabilities.")
    latency_ms: float = Field(..., description="End-to-end processing latency in milliseconds.")
    telemetry: Optional[BlinkTelemetry] = Field(default=None, description="System 1 permutation and consensus telemetry.")
    reasoning_trace: Optional[str] = Field(default=None, description="System 2 Chain-of-Thought reasoning (if escalated).")
    raw_logits: Optional[Dict[str, float]] = Field(default=None, description="Unscaled token logits.")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
