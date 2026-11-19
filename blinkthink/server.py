"""
BlinkThink Decision Service: Production OpenAPI REST Service.
"""

from contextlib import asynccontextmanager
import os
from typing import Any, Dict, List, Literal, Optional
from fastapi import FastAPI, HTTPException, Query, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from .client import BlinkThink
from .schemas import DecisionRequest, DecisionResult

_engine: Optional[BlinkThink] = None
security = HTTPBearer(auto_error=False)


def set_engine(engine: BlinkThink):
    global _engine
    _engine = engine


def get_engine() -> BlinkThink:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Decision engine not initialized.")
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    if _engine is None:
        model_id = os.getenv("BLINKTHINK_MODEL_ID", "google/gemma-4-12B-it")
        config_path = os.getenv("BLINKTHINK_CONFIG_PATH", None)

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        use_cuda = torch.cuda.is_available()
        bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            if use_cuda
            else None
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto" if use_cuda else None,
            torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
            attn_implementation="sdpa" if use_cuda else "eager",
        )

        prompt_mode = os.getenv("BLINKTHINK_PROMPT_MODE", "auto")
        confidence_metric = os.getenv("BLINKTHINK_CONFIDENCE_METRIC", "composite")
        threshold = float(os.getenv("BLINKTHINK_THRESHOLD", "0.35"))
        n_blinks = int(os.getenv("BLINKTHINK_N_BLINKS", "2"))
        _engine = BlinkThink(
            model=model,
            tokenizer=tokenizer,
            threshold=threshold,
            n_blinks=n_blinks,
            config_path=config_path,
            prompt_mode=prompt_mode,
            confidence_metric=confidence_metric,
        )
    yield
    _engine = None


def create_app(engine: Optional[BlinkThink] = None) -> FastAPI:
    if engine is not None:
        set_engine(engine)

    app = FastAPI(
        title="BlinkThink Decision API",
        description="Calibrated Dual-Process Routing API for Fast, High-Reliability LLM Decisions",
        version="1.1.0",
        lifespan=lifespan if engine is None else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)):
        required_key = os.getenv("BLINKTHINK_API_KEY")
        if required_key:
            if not credentials or credentials.credentials != required_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or missing API Bearer token.",
                )
        return True

    @app.get("/health", tags=["System"])
    def health_check():
        return {
            "status": "healthy" if _engine is not None else "initializing",
            "engine_ready": _engine is not None,
        }

    @app.post("/v1/decide", response_model=DecisionResult[str], tags=["Decisions"])
    async def decide_endpoint(
        request: DecisionRequest,
        n_blinks: Optional[int] = Query(default=None),
        threshold: Optional[float] = Query(default=None),
        force_mode: Optional[str] = Query(default=None),
        auth: bool = Security(verify_api_key),
    ):
        engine = get_engine()
        active_blinks = n_blinks if n_blinks is not None else request.n_blinks
        active_threshold = threshold if threshold is not None else request.threshold
        active_mode = force_mode if force_mode is not None else request.force_mode

        return await engine.adecide(
            request=request,
            n_blinks=active_blinks,
            threshold=active_threshold,
            force_mode=active_mode,
        )

    class ChoiceRequest(BaseModel):
        prompt: str
        options: List[str]
        metadata: Optional[Dict[str, Any]] = None
        n_blinks: Optional[int] = None
        threshold: Optional[float] = None
        force_mode: Optional[Literal["blink_only", "deliberate", "force_deliberate", "think_only"]] = None

    class BinaryRequest(BaseModel):
        prompt: str
        options: Optional[List[str]] = None
        metadata: Optional[Dict[str, Any]] = None
        n_blinks: Optional[int] = None
        threshold: Optional[float] = None
        force_mode: Optional[Literal["blink_only", "deliberate", "force_deliberate", "think_only"]] = None

    @app.post("/v1/decide/choice", response_model=DecisionResult[str], tags=["Decisions"])
    async def decide_choice(req: ChoiceRequest, auth: bool = Security(verify_api_key)):
        return await get_engine().achoice(
            req.prompt,
            options=req.options,
            metadata=req.metadata,
            n_blinks=req.n_blinks,
            threshold=req.threshold,
            force_mode=req.force_mode,
        )

    @app.post("/v1/decide/binary", response_model=DecisionResult[str], tags=["Decisions"])
    async def decide_binary(req: BinaryRequest, auth: bool = Security(verify_api_key)):
        return await get_engine().abinary(
            req.prompt,
            options=req.options,
            metadata=req.metadata,
            n_blinks=req.n_blinks,
            threshold=req.threshold,
            force_mode=req.force_mode,
        )

    class BatchDecisionRequest(BaseModel):
        requests: List[DecisionRequest]

    class BatchDecisionResponse(BaseModel):
        results: List[DecisionResult[str]]

    @app.post("/v1/decide/batch", response_model=BatchDecisionResponse, tags=["Decisions"])
    async def decide_batch(payload: BatchDecisionRequest, auth: bool = Security(verify_api_key)):
        engine = get_engine()
        results = [await engine.adecide(req) for req in payload.requests]
        return BatchDecisionResponse(results=results)

    return app


app = create_app()

def run_server():
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("blinkthink.server:app", host=host, port=port, reload=False)
