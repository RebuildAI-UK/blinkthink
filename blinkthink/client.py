"""
BlinkThink Developer Client: Pythonic ergonomic API for local and embedded execution.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from .calibrator import TemperatureCalibrator
from .router import BlinkThinkRouter
from .schemas import DecisionRequest, DecisionResult, DecisionType


class BlinkThink:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        threshold: float = 0.35,
        n_blinks: int = 2,
        strict_consensus: bool = False,
        calibrator: Optional[TemperatureCalibrator] = None,
        max_think_tokens: int = 512,
        device: Optional[Union[str, torch.device]] = None,
        config_path: Optional[str] = None,
        prompt_mode: Literal["auto", "chat", "completion"] = "auto",
        confidence_metric: Literal["composite", "margin", "msp"] = "composite",
    ):
        self.router = BlinkThinkRouter(
            model=model,
            tokenizer=tokenizer,
            n_blinks=n_blinks,
            threshold=threshold,
            strict_consensus=strict_consensus,
            calibrator=calibrator,
            max_think_tokens=max_think_tokens,
            device=device,
            prompt_mode=prompt_mode,
            confidence_metric=confidence_metric,
        )
        if config_path:
            self.router.load_config(config_path)

        self._executor = ThreadPoolExecutor(max_workers=1)

    def choice(
        self,
        prompt: str,
        options: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        n_blinks: Optional[int] = None,
        threshold: Optional[float] = None,
        force_mode: Optional[Literal["blink_only", "deliberate", "force_deliberate", "think_only"]] = None,
    ) -> DecisionResult[str]:
        req = DecisionRequest(
            prompt=prompt,
            decision_type=DecisionType.CHOICE,
            options=options,
            metadata=metadata,
            n_blinks=n_blinks,
            threshold=threshold,
            force_mode=force_mode,
        )
        return self.decide(req, n_blinks=n_blinks, threshold=threshold, force_mode=force_mode)

    def binary(
        self,
        prompt: str,
        options: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        n_blinks: Optional[int] = None,
        threshold: Optional[float] = None,
        force_mode: Optional[Literal["blink_only", "deliberate", "force_deliberate", "think_only"]] = None,
    ) -> DecisionResult[str]:
        return self.choice(
            prompt=prompt,
            options=options or ["Yes", "No"],
            metadata=metadata,
            n_blinks=n_blinks,
            threshold=threshold,
            force_mode=force_mode,
        )

    def decide(
        self,
        request: Union[DecisionRequest, Dict[str, Any]],
        n_blinks: Optional[int] = None,
        threshold: Optional[float] = None,
        force_mode: Optional[Literal["blink_only", "deliberate", "force_deliberate", "system_1", "system_2", "think_only"]] = None,
    ) -> DecisionResult[str]:
        if isinstance(request, dict):
            request = DecisionRequest(**request)
        return self.router.decide(
            request=request,
            n_blinks=n_blinks,
            threshold=threshold,
            force_mode=force_mode,
        )

    async def adecide(self, *args, **kwargs) -> DecisionResult[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: self.decide(*args, **kwargs))

    async def achoice(self, *args, **kwargs) -> DecisionResult[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: self.choice(*args, **kwargs))

    async def abinary(self, *args, **kwargs) -> DecisionResult[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: self.binary(*args, **kwargs))

    def save_config(self, filepath: str):
        self.router.save_config(filepath)

    def load_config(self, filepath: str):
        self.router.load_config(filepath)

    def close(self):
        self._executor.shutdown(wait=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
