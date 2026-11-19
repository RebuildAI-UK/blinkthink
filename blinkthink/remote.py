"""
BlinkThink Remote Client: Light-weight HTTP client for remote service interaction.
"""

from typing import Any, Dict, List, Literal, Optional, Union
import httpx
from .schemas import DecisionRequest, DecisionResult, DecisionType


class BlinkThinkRemoteClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.client.close()

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
        force_mode: Optional[str] = None,
    ) -> DecisionResult[str]:
        payload = request.model_dump(mode="json") if isinstance(request, DecisionRequest) else request
        params = {}
        if n_blinks is not None:
            params["n_blinks"] = n_blinks
        if threshold is not None:
            params["threshold"] = threshold
        if force_mode is not None:
            params["force_mode"] = force_mode

        resp = self.client.post("/v1/decide", json=payload, params=params)
        resp.raise_for_status()
        return DecisionResult.model_validate(resp.json())

    def decide_batch(self, requests: List[Union[DecisionRequest, Dict[str, Any]]]) -> List[DecisionResult[str]]:
        payload = {
            "requests": [
                r.model_dump(mode="json") if isinstance(r, DecisionRequest) else r
                for r in requests
            ]
        }
        resp = self.client.post("/v1/decide/batch", json=payload)
        resp.raise_for_status()
        return [DecisionResult.model_validate(item) for item in resp.json()["results"]]


class AsyncBlinkThinkRemoteClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=timeout)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def aclose(self):
        await self.client.aclose()

    async def choice(
        self,
        prompt: str,
        options: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        n_blinks: Optional[int] = None,
        threshold: Optional[float] = None,
        force_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> DecisionResult[str]:
        req = DecisionRequest(
            prompt=prompt,
            decision_type=DecisionType.CHOICE,
            options=options,
            metadata=metadata,
            n_blinks=n_blinks,
            threshold=threshold,
            force_mode=force_mode,
            **kwargs,
        )
        return await self.decide(req, n_blinks=n_blinks, threshold=threshold, force_mode=force_mode)

    async def binary(
        self,
        prompt: str,
        options: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        n_blinks: Optional[int] = None,
        threshold: Optional[float] = None,
        force_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> DecisionResult[str]:
        return await self.choice(
            prompt=prompt,
            options=options or ["Yes", "No"],
            metadata=metadata,
            n_blinks=n_blinks,
            threshold=threshold,
            force_mode=force_mode,
            **kwargs,
        )

    async def decide(self, request: Union[DecisionRequest, Dict[str, Any]], **params) -> DecisionResult[str]:
        payload = request.model_dump(mode="json") if isinstance(request, DecisionRequest) else request
        resp = await self.client.post("/v1/decide", json=payload, params={k: v for k, v in params.items() if v is not None})
        resp.raise_for_status()
        return DecisionResult.model_validate(resp.json())

    async def decide_batch(self, requests: List[Union[DecisionRequest, Dict[str, Any]]]) -> List[DecisionResult[str]]:
        payload = {
            "requests": [
                r.model_dump(mode="json") if isinstance(r, DecisionRequest) else r
                for r in requests
            ]
        }
        resp = await self.client.post("/v1/decide/batch", json=payload)
        resp.raise_for_status()
        return [DecisionResult.model_validate(item) for item in resp.json()["results"]]
