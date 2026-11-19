"""
System 1: The Blink Engine.
Executes N fast blinks in a single batched prefill pass (0 autoregressively decoded tokens).
"""

import string
import time
from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import torch
from .calibrator import TemperatureCalibrator
from .schemas import BlinkTelemetry, DecisionRequest, DecisionResult


class BlinkEngine:
    def __init__(
        self,
        model,
        tokenizer,
        calibrator: Optional[TemperatureCalibrator] = None,
        device: Optional[Union[str, torch.device]] = None,
        prompt_mode: Literal["auto", "chat", "completion"] = "auto",
        confidence_metric: Literal["composite", "margin", "msp"] = "composite",
    ):
        self.model = model
        self.tokenizer = tokenizer
        if getattr(self.tokenizer, "padding_side", None) != "left":
            self.tokenizer.padding_side = "left"
        if getattr(self.tokenizer, "pad_token", None) is None:
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        self.calibrator = calibrator or TemperatureCalibrator()
        self.confidence_metric = confidence_metric

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if prompt_mode == "auto":
            has_template = bool(
                hasattr(self.tokenizer, "chat_template")
                and self.tokenizer.chat_template is not None
            )
            model_name = str(
                getattr(model, "name_or_path", "")
                or getattr(getattr(model, "config", None), "_name_or_path", "")
            ).lower()
            is_instruct_named = any(tag in model_name for tag in ["-it", "_it", "instruct", "chat"])
            is_base_named = any(tag in model_name for tag in ["-base", "_base"])

            if is_instruct_named:
                self.use_chat_prompt = has_template
            elif is_base_named:
                self.use_chat_prompt = False
            else:
                self.use_chat_prompt = has_template
        else:
            self.use_chat_prompt = (prompt_mode == "chat")

    def _generate_permutations(
        self, request: DecisionRequest, n_blinks: int
    ) -> Tuple[List[str], List[List[str]], List[List[int]], List[str]]:
        options = request.options
        k = len(options)
        prompts, permuted_labels, index_mappings = [], [], []

        for i in range(n_blinks):
            shift = i % k
            perm_opts = options[shift:] + options[:shift]
            perm_mapping = [(shift + idx) % k for idx in range(k)]
            letters = list(string.ascii_uppercase[:k])
            body = "\n".join(f"{ltr}: {opt}" for ltr, opt in zip(letters, perm_opts))

            if self.use_chat_prompt:
                user_msg = (
                    f"{request.prompt.strip()}\n\n"
                    f"Options:\n{body}\n\n"
                    f"Select the correct option. Respond with ONLY the single option letter "
                    f"({', '.join(letters)})."
                )
                try:
                    prompt_text = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": user_msg}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    prompt_text = f"{user_msg}\n\nAnswer:"
            else:
                prompt_text = f"{request.prompt.strip()}\n\nOptions:\n{body}\n\nAnswer:"

            prompts.append(prompt_text)
            permuted_labels.append(letters)
            index_mappings.append(perm_mapping)

        return prompts, permuted_labels, index_mappings, options

    def _resolve_candidate_tokens(
        self, prompts: List[str], label_lists: List[List[str]], target_device: torch.device
    ) -> torch.Tensor:
        all_ids = []
        for prompt, labels in zip(prompts, label_lists):
            ids = []
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
            len_p = len(prompt_tokens)
            has_space_ending = bool(prompt and prompt[-1].isspace())

            for lbl in labels:
                s_lbl = lbl.strip()
                cand = f"{prompt}{s_lbl}" if has_space_ending else f"{prompt} {s_lbl}"
                full_tokens = self.tokenizer.encode(cand, add_special_tokens=True)

                if len(full_tokens) > len_p and full_tokens[:len_p] == prompt_tokens:
                    token_id = full_tokens[len_p]
                else:
                    first_try = s_lbl if has_space_ending else f" {s_lbl}"
                    second_try = f" {s_lbl}" if has_space_ending else s_lbl
                    enc = self.tokenizer.encode(first_try, add_special_tokens=False)
                    if not enc:
                        enc = self.tokenizer.encode(second_try, add_special_tokens=False)
                    token_id = enc[0] if enc else (self.tokenizer.unk_token_id or 0)
                ids.append(token_id)
            all_ids.append(ids)
        return torch.tensor(all_ids, device=target_device)

    @torch.inference_mode()
    def decide(
        self,
        request: DecisionRequest,
        n_blinks: int = 1,
        strict_consensus: bool = False,
        confidence_metric: Optional[Literal["composite", "margin", "msp"]] = None,
    ) -> DecisionResult:
        start_time = time.perf_counter()
        prompts, perm_labels, mappings, canonical_labels = self._generate_permutations(request, n_blinks)
        actual_blinks = len(prompts)

        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)

        try:
            outputs = self.model(**inputs, logits_to_keep=1, use_cache=False)
            batch_logits = outputs.logits[:, -1, :]
        except TypeError:
            outputs = self.model(**inputs, use_cache=False)
            batch_logits = outputs.logits[:, -1, :]

        target_device = batch_logits.device
        candidate_ids = self._resolve_candidate_tokens(prompts, perm_labels, target_device=target_device)

        cand_logits = torch.gather(batch_logits, 1, candidate_ids)
        calibrated_probs = self.calibrator.calibrate(cand_logits, request.decision_type)

        canonical_probs = torch.zeros_like(calibrated_probs)
        for b in range(actual_blinks):
            for slot, orig_idx in enumerate(mappings[b]):
                canonical_probs[b, orig_idx] = calibrated_probs[b, slot]

        mean_prob = torch.mean(canonical_probs, dim=0)
        prob_dict = {canonical_labels[i]: float(mean_prob[i].item()) for i in range(len(canonical_labels))}

        blink_choices_idx = torch.argmax(canonical_probs, dim=1).tolist()
        chosen_idx = int(torch.argmax(mean_prob).item())

        consensus_rate = sum(1 for idx in blink_choices_idx if idx == chosen_idx) / actual_blinks

        margins = []
        for b in range(actual_blinks):
            top_vals = torch.topk(canonical_probs[b], min(2, len(canonical_labels))).values
            m = (top_vals[0] - top_vals[1]).item() if len(top_vals) > 1 else top_vals[0].item()
            margins.append(float(m))
        agg_margin = float(sum(margins) / len(margins))

        drift = 0.0
        if actual_blinks > 1:
            total_pairs = 0
            for i in range(actual_blinks):
                for j in range(i + 1, actual_blinks):
                    drift += 0.5 * torch.sum(torch.abs(canonical_probs[i] - canonical_probs[j])).item()
                    total_pairs += 1
            drift = drift / total_pairs if total_pairs > 0 else 0.0

        active_metric = confidence_metric or getattr(request, "confidence_metric", None) or self.confidence_metric
        base_confidence = agg_margin

        if active_metric == "msp":
            confidence = float(torch.max(mean_prob).item())
        elif actual_blinks == 1 or active_metric == "margin":
            confidence = base_confidence
        else:  # "composite"
            if strict_consensus and consensus_rate < 1.0:
                confidence = 0.0
            else:
                stability = max(0.0, 1.0 - drift)
                confidence = base_confidence * float(np.sqrt(consensus_rate * stability))

        latency = (time.perf_counter() - start_time) * 1000.0

        final_decision = canonical_labels[chosen_idx]
        blink_decisions = [canonical_labels[idx] for idx in blink_choices_idx]

        raw_logits_dict = {
            canonical_labels[mappings[0][slot]]: float(cand_logits[0, slot].item())
            for slot in range(len(canonical_labels))
        }

        return DecisionResult(
            decision=final_decision,
            confidence=float(confidence),
            system_used="system_1",
            probabilities=prob_dict,
            latency_ms=latency,
            raw_logits=raw_logits_dict,
            telemetry=BlinkTelemetry(
                n_blinks=actual_blinks,
                blink_decisions=blink_decisions,
                consensus_rate=consensus_rate,
                drift=drift,
                raw_margins=margins,
                canonical_distributions=[
                    {canonical_labels[i]: float(canonical_probs[b, i].item()) for i in range(len(canonical_labels))}
                    for b in range(actual_blinks)
                ],
            ),
        )
