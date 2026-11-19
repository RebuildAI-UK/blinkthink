import logging
from typing import Any, Dict, Optional, Tuple
from scipy.optimize import minimize_scalar
import torch
import torch.nn.functional as F
from .schemas import DecisionType

logger = logging.getLogger(__name__)


class TemperatureCalibrator:
    """
    Manages and optimizes temperature scaling for choice decisions.
    """
    def __init__(
        self,
        temperature: float = 3.4,
        temperature_choice: Optional[float] = None,
        enable_prior_correction: bool = False,
        **kwargs: Any,
    ):
        temp = temperature_choice if temperature_choice is not None else temperature
        self.temperature = float(temp)
        self.enable_prior_correction = enable_prior_correction

    @property
    def temperatures(self) -> Dict[DecisionType, float]:
        return {DecisionType.CHOICE: self.temperature}

    def calibrate(self, logits: torch.Tensor, decision_type: Optional[DecisionType] = None) -> torch.Tensor:
        temp = max(self.temperature, 1e-4)
        scaled = logits / temp
        return F.softmax(scaled, dim=-1)

    def fit(
        self,
        candidate_logits: Any,
        target_indices: Any,
        decision_type: Optional[DecisionType] = None,
        temp_bounds: Tuple[float, float] = (0.1, 10.0),
    ) -> float:
        """
        Finds optimal temperature T* by minimizing Negative Log-Likelihood (NLL).
        """
        if isinstance(candidate_logits, torch.Tensor) and candidate_logits.ndim == 2:
            logits_list = [row for row in candidate_logits.detach().cpu().float()]
        else:
            logits_list = [
                row.detach().cpu().float() if isinstance(row, torch.Tensor)
                else torch.tensor(row, dtype=torch.float32)
                for row in candidate_logits
            ]

        targets_list = (
            target_indices.detach().cpu().long().tolist()
            if isinstance(target_indices, torch.Tensor)
            else [int(t) for t in target_indices]
        )

        if len(logits_list) < 3:
            logger.warning(
                "Only %d samples provided for calibration. Retaining temperature T=%.2f.",
                len(logits_list),
                self.temperature,
            )
            return self.temperature

        def nll_objective(t: float) -> float:
            t_clamped = max(t, 1e-4)
            total_nll = 0.0
            for l_row, target in zip(logits_list, targets_list):
                scaled = l_row / t_clamped
                log_probs = F.log_softmax(scaled, dim=-1)
                total_nll += -log_probs[target].item()
            return total_nll / len(logits_list)

        res = minimize_scalar(nll_objective, bounds=temp_bounds, method="bounded")
        self.temperature = float(res.x)
        logger.info("Temperature calibrated: T* = %.3f", self.temperature)
        return self.temperature

    def to_dict(self) -> Dict[str, float]:
        return {"choice": self.temperature}

    def load_dict(self, data: Dict[str, float]):
        if "choice" in data:
            self.temperature = float(data["choice"])
        elif "temperature" in data:
            self.temperature = float(data["temperature"])
