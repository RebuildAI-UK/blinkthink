import json
from typing import Any, Dict, Literal, Optional, Union
import torch
from .calibrator import TemperatureCalibrator
from .system1 import BlinkEngine
from .system2 import ThinkEngine
from .schemas import DecisionRequest, DecisionResult


class BlinkThinkRouter:
    """
    Main API interface for BlinkThink dual-process routing.
    """
    def __init__(
        self,
        model,
        tokenizer,
        n_blinks: int = 2,
        threshold: float = 0.35,
        strict_consensus: bool = False,
        calibrator: Optional[TemperatureCalibrator] = None,
        max_think_tokens: int = 512,
        device: Optional[Union[str, torch.device]] = None,
        prompt_mode: Literal["auto", "chat", "completion"] = "auto",
        confidence_metric: Literal["composite", "margin", "msp"] = "composite",
    ):
        if getattr(tokenizer, "pad_token", None) is None:
            if getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token

        self.calibrator = calibrator or TemperatureCalibrator()
        self.confidence_metric = confidence_metric
        self.blink_engine = BlinkEngine(
            model,
            tokenizer,
            self.calibrator,
            device=device,
            prompt_mode=prompt_mode,
            confidence_metric=confidence_metric,
        )
        self.think_engine = ThinkEngine(
            model, tokenizer, calibrator=self.calibrator, max_new_tokens=max_think_tokens, use_calibrator_temperature=False, device=device
        )

        self.n_blinks = n_blinks
        self.threshold = threshold
        self.strict_consensus = strict_consensus

    def decide(
        self,
        request: Union[DecisionRequest, Dict[str, Any]],
        n_blinks: Optional[int] = None,
        threshold: Optional[float] = None,
        force_mode: Optional[Literal["blink_only", "deliberate", "force_deliberate", "system_1", "system_2", "think_only"]] = None,
    ) -> DecisionResult[str]:
        if isinstance(request, dict):
            request = DecisionRequest(**request)

        active_mode = force_mode if force_mode is not None else getattr(request, "force_mode", None)
        active_blinks = n_blinks if n_blinks is not None else (getattr(request, "n_blinks", None) or self.n_blinks)
        active_threshold = threshold if threshold is not None else (
            getattr(request, "threshold", None) if getattr(request, "threshold", None) is not None else self.threshold
        )

        active_metric = getattr(request, "confidence_metric", None) or self.confidence_metric

        if active_mode in ("blink_only", "system_1"):
            return self.blink_engine.decide(
                request,
                n_blinks=active_blinks,
                strict_consensus=self.strict_consensus,
                confidence_metric=active_metric,
            )

        blink_res = self.blink_engine.decide(
            request,
            n_blinks=active_blinks,
            strict_consensus=self.strict_consensus,
            confidence_metric=active_metric,
        )

        is_forced_deliberation = active_mode in ("deliberate", "force_deliberate", "system_2", "think_only")
        if not is_forced_deliberation and blink_res.confidence >= active_threshold:
            return blink_res

        think_res = self.think_engine.decide(request, fallback=blink_res)
        think_res.latency_ms += blink_res.latency_ms
        if think_res.telemetry is None:
            think_res.telemetry = blink_res.telemetry

        return think_res

    def save_config(self, filepath: str):
        config = {
            "threshold": self.threshold,
            "n_blinks": self.n_blinks,
            "strict_consensus": self.strict_consensus,
            "temperatures": self.calibrator.to_dict(),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def load_config(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            config = json.load(f)
        self.threshold = config.get("threshold", self.threshold)
        self.n_blinks = config.get("n_blinks", self.n_blinks)
        self.strict_consensus = config.get("strict_consensus", self.strict_consensus)
        if "temperatures" in config:
            self.calibrator.load_dict(config["temperatures"])
