"""
Automated Tuning Harness for BlinkThink.
Optimizes calibration temperature and parameter-free dual-process routing thresholds.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Literal, Tuple
import numpy as np
import torch
from .router import BlinkThinkRouter
from .schemas import DecisionRequest, DecisionType

logger = logging.getLogger(__name__)


@dataclass
class TuningReport:
    tuned_temperatures: Dict[str, float]
    optimal_threshold: float
    validation_accuracy: float
    validation_escalation_pct: float
    system1_only_accuracy: float
    system2_only_accuracy: float


class BlinkThinkTuner:
    def __init__(self, router: BlinkThinkRouter):
        self.router = router

    def fit(
        self,
        val_dataset: List[Tuple[DecisionRequest, Any]],
        n_blinks: int = 2,
        objective: Literal["balanced", "peak_synergy", "knee_point", "target_accuracy", "max_escalation"] = "balanced",
        target_value: float = 0.70,
        tune_temperatures: bool = True,
        verbose: bool = True,
    ) -> TuningReport:
        if not val_dataset:
            raise ValueError("Validation dataset cannot be empty.")

        n_samples = len(val_dataset)
        logger.info("Fitting BlinkThink parameters on %d validation samples (blinks=%d)...", n_samples, n_blinks)

        # STEP 1: Fit Calibration Temperature on Prefill Logits
        if tune_temperatures:
            logits_list: List[List[float]] = []
            targets_list: List[int] = []

            for req, y_true in val_dataset:
                s1_res = self.router.blink_engine.decide(req, n_blinks=1)
                labels = list(s1_res.probabilities.keys())

                if y_true in labels:
                    target_idx = labels.index(y_true)
                else:
                    matched = [i for i, lbl in enumerate(labels) if lbl.strip().lower() == str(y_true).strip().lower()]
                    if matched:
                        target_idx = matched[0]
                    else:
                        continue

                if s1_res.raw_logits:
                    cand_logits = [s1_res.raw_logits[lbl] for lbl in labels]
                    logits_list.append(cand_logits)
                    targets_list.append(target_idx)

            if len(logits_list) >= 3:
                t_logits = [torch.tensor(l, dtype=torch.float32) for l in logits_list]
                self.router.calibrator.fit(t_logits, targets_list, DecisionType.CHOICE)

        # STEP 2: Profile System 1 vs System 2 Across the Validation Set
        confidences: List[float] = []
        s1_corrects: List[int] = []
        s2_corrects: List[int] = []

        iterator = val_dataset
        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(val_dataset, desc="Profiling S1 vs S2", unit="sample", leave=False)
            except ImportError:
                pass

        for req, y_true in iterator:
            s1_res = self.router.blink_engine.decide(
                req, n_blinks=n_blinks, strict_consensus=self.router.strict_consensus
            )
            s2_res = self.router.think_engine.decide(req, fallback=s1_res)

            confidences.append(float(s1_res.confidence))
            s1_corrects.append(1 if s1_res.decision == y_true else 0)
            s2_corrects.append(1 if s2_res.decision == y_true else 0)

        conf_arr = np.array(confidences)
        s1_arr = np.array(s1_corrects)
        s2_arr = np.array(s2_corrects)

        # STEP 3: Vectorized Threshold Optimization
        candidate_taus = np.linspace(0.0, 1.0, 101)
        escalated_matrix = conf_arr[None, :] < candidate_taus[:, None]
        acc_curve = np.mean(np.where(escalated_matrix, s2_arr[None, :], s1_arr[None, :]), axis=1)
        esc_curve = np.mean(escalated_matrix, axis=1)

        s1_base_acc = float(np.mean(s1_arr))
        s2_base_acc = float(np.mean(s2_arr))
        max_acc = float(np.max(acc_curve))
        tie_breaker = 1.0 / (2.0 * n_samples)

        best_tau = 0.35

        if objective == "balanced":
            max_gain = max_acc - s1_base_acc
            if max_gain <= 1e-6:
                best_tau = 0.0
            else:
                gain_recall = np.maximum(0.0, (acc_curve - s1_base_acc) / max_gain)
                compute_saved = 1.0 - esc_curve
                f1_scores = np.where(
                    (gain_recall + compute_saved) > 0,
                    2.0 * (gain_recall * compute_saved) / (gain_recall + compute_saved + 1e-8),
                    0.0
                )
                f1_scores = f1_scores - (tie_breaker * esc_curve)
                best_tau = float(candidate_taus[np.argmax(f1_scores)])

        elif objective == "peak_synergy":
            if max_acc <= s1_base_acc:
                best_tau = 0.0
            else:
                regularized_acc = acc_curve - (tie_breaker * esc_curve)
                best_tau = float(candidate_taus[np.argmax(regularized_acc)])

        elif objective == "knee_point":
            if max_acc <= s1_base_acc:
                best_tau = 0.0
            else:
                peak_idx = int(np.argmax(acc_curve - (tie_breaker * esc_curve)))
                e_peak = max(1e-4, esc_curve[peak_idx])
                a_gain_peak = max(1e-4, acc_curve[peak_idx] - s1_base_acc)

                norm_x = esc_curve / e_peak
                norm_y = (acc_curve - s1_base_acc) / a_gain_peak
                chord_dist = norm_y - norm_x
                chord_dist[peak_idx + 1:] = -np.inf
                best_tau = float(candidate_taus[np.argmax(chord_dist)])

        elif objective == "target_accuracy":
            valid_indices = np.where(acc_curve >= target_value)[0]
            if len(valid_indices) > 0:
                best_idx = valid_indices[np.argmin(esc_curve[valid_indices])]
                best_tau = float(candidate_taus[best_idx])
            else:
                regularized_acc = acc_curve - (tie_breaker * esc_curve)
                best_tau = float(candidate_taus[np.argmax(regularized_acc)])

        elif objective == "max_escalation":
            valid_indices = np.where(esc_curve <= target_value)[0]
            if len(valid_indices) > 0:
                best_idx = valid_indices[np.argmax(acc_curve[valid_indices])]
                best_tau = float(candidate_taus[best_idx])
            else:
                best_tau = 0.0

        self.router.threshold = best_tau
        self.router.n_blinks = n_blinks

        final_esc = (conf_arr < best_tau)
        final_acc = float(np.mean(np.where(final_esc, s2_arr, s1_arr)))

        report = TuningReport(
            tuned_temperatures=self.router.calibrator.to_dict(),
            optimal_threshold=best_tau,
            validation_accuracy=final_acc,
            validation_escalation_pct=float(np.mean(final_esc) * 100.0),
            system1_only_accuracy=s1_base_acc,
            system2_only_accuracy=s2_base_acc,
        )

        logger.info(
            "Optimization complete [%s]: tau* = %.2f (Val Acc: %.1f%%, Escalation: %.1f%%)",
            objective.upper(), report.optimal_threshold, report.validation_accuracy * 100.0, report.validation_escalation_pct
        )
        return report
