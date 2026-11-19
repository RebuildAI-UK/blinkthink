"""
System 2: Think Engine.
Executes deliberative Chain-of-Thought (CoT) reasoning with tag delimitation,
resilient multi-template chat formatting, and strict fallback triggering.
"""

from collections import OrderedDict
import re
import string
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import StoppingCriteria, StoppingCriteriaList

try:
    from transformers import StopStringCriteria
except ImportError:
    try:
        from transformers.generation.stopping_criteria import StopStringCriteria
    except ImportError:
        class StopStringCriteria(StoppingCriteria):  # type: ignore
            def __init__(self, tokenizer, stop_strings: List[str]):
                super().__init__()
                self.tokenizer = tokenizer
                self.stop_strings = [s.lower() for s in stop_strings]

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                tail = self.tokenizer.decode(input_ids[0, -15:], skip_special_tokens=False).lower()
                return any(stop in tail for stop in self.stop_strings)

from .calibrator import TemperatureCalibrator
from .schemas import DecisionRequest, DecisionResult


class ThinkEngine:
    def __init__(
        self,
        model,
        tokenizer,
        calibrator: Optional[TemperatureCalibrator] = None,
        max_new_tokens: int = 512,
        use_calibrator_temperature: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        max_cache_size: int = 10000,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.calibrator = calibrator
        self.max_new_tokens = max_new_tokens
        self.use_calibrator_temperature = use_calibrator_temperature
        self.max_cache_size = max_cache_size
        self.device = device or getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self._cache: OrderedDict[Tuple[str, Optional[Tuple[str, ...]], str], DecisionResult] = OrderedDict()

    def _get_cache_key(self, request: DecisionRequest) -> Tuple[str, Optional[Tuple[str, ...]], str]:
        opts = tuple(request.options) if request.options else None
        return (request.prompt.strip(), opts, request.decision_type.value)

    def warm_cache(self, request: DecisionRequest, result: DecisionResult):
        key = self._get_cache_key(request)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = result.model_copy(deep=True)
        if len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)

    def clear_cache(self):
        self._cache.clear()

    def _format_chat_prompt(self, request: DecisionRequest) -> Tuple[str, List[str]]:
        system_instruction = (
            "You are a rigorous, precise evaluator. Analyze the problem step-by-step inside "
            "<reasoning>...</reasoning> tags. Then conclude with your exact final choice strictly inside "
            "<answer>...</answer> tags.\n"
            "Format:\n"
            "<reasoning>\n[Concise step-by-step analysis]\n</reasoning>\n"
            "<answer>[Final Option Letter]</answer>"
        )

        k = len(request.options)
        letters = list(string.ascii_uppercase[:k])
        valid_labels = letters
        opts_str = "\n".join(f"{ltr}: {opt}" for ltr, opt in zip(letters, request.options))
        user_content = (
            f"{request.prompt.strip()}\n\n"
            f"Options:\n{opts_str}\n\n"
            f"Provide your reasoning, then place ONLY the single option letter ({', '.join(letters)}) inside the <answer> tags."
        )

        messages_with_system = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content},
        ]
        messages_user_only = [
            {"role": "user", "content": f"{system_instruction}\n\n{user_content}"}
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            # 1. Try standard system + user roles
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages_with_system, tokenize=False, add_generation_prompt=True
                )
                return formatted, valid_labels
            except Exception:
                pass

            # 2. Try single user role
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages_user_only, tokenize=False, add_generation_prompt=True
                )
                return formatted, valid_labels
            except Exception:
                pass

        # 3. Base model raw text completion fallback
        raw = (
            f"{system_instruction}\n\n"
            f"Question:\n{request.prompt.strip()}\n\n"
            f"Options:\n{opts_str}\n\n"
            f"<reasoning>\n"
        )
        return raw, valid_labels

    def _parse_answer(self, text: str, request: DecisionRequest) -> Optional[str]:
        k = len(request.options)
        letters = list(string.ascii_uppercase[:k])

        # If a thinking block closed, only parse the post-</think> response
        search_text = text.split("</think>")[-1] if "</think>" in text else text

        # 1. Explicit closed <answer>...</answer> tag matching (last match wins;
        # avoids swallowing unclosed prompt echoes like "in <answer> tags")
        closed_matches = list(
            re.finditer(r"<answer>\s*([^<]*?)\s*</answer>", search_text, re.DOTALL | re.IGNORECASE)
        )
        if not closed_matches:
            tail_tag = re.search(r"<answer>\s*([A-Z])\.?\s*$", search_text, re.IGNORECASE)
            if tail_tag:
                closed_matches = [tail_tag]

        for match in reversed(closed_matches):
            raw_val = match.group(1).strip()
            clean_val = re.sub(r"[\*\_`\[\]\(\)]", "", raw_val).strip()
            if not clean_val or clean_val in ("...", "Final Option Letter"):
                continue

            # Exact letter match: "B"
            if clean_val.upper() in letters:
                return request.options[letters.index(clean_val.upper())]

            # Letter prefix match: "B: Mitochondria" or "B."
            letter_prefix = re.match(r"^([A-Z])(?:\s*[:.\-)]|$)", clean_val, re.IGNORECASE)
            if letter_prefix and letter_prefix.group(1).upper() in letters:
                return request.options[letters.index(letter_prefix.group(1).upper())]

            # Explicit sentence inside tag: "Answer is B" or "Option B"
            explicit_in_tag = re.search(
                r"\b(?:answer|choice|option)\s*(?:is|:|=)?\s*\b([A-Z])\b",
                clean_val,
                re.IGNORECASE,
            )
            if explicit_in_tag and explicit_in_tag.group(1).upper() in letters:
                return request.options[letters.index(explicit_in_tag.group(1).upper())]

            # Option string content matching: e.g. <answer>Mitochondria</answer>
            for idx, opt in enumerate(request.options):
                opt_str = opt.strip()
                if len(opt_str) > 1 and opt_str.lower() == clean_val.lower():
                    return request.options[idx]
            for idx, opt in enumerate(request.options):
                opt_str = opt.strip()
                if len(opt_str) > 3 and opt_str.lower() in clean_val.lower():
                    return request.options[idx]

        # 2. Strict tail search when tags are omitted or generation hit max_new_tokens
        tail = search_text[-250:]
        tail_explicit = re.search(
            r"(?i:(?:final\s+answer|correct\s+(?:choice|option|answer)\s*(?:is|:|=)?|(?:therefore|thus|hence),?\s*(?:the\s+)?(?:answer|choice|option)\s*(?:is|:|=)|(?:answer|choice|option)\s*(?:is|:|=)))\s*[:=\-]?\s*\b\(?([A-Z])\)?\b",
            tail,
        )
        if tail_explicit and tail_explicit.group(1).upper() in letters:
            return request.options[letters.index(tail_explicit.group(1).upper())]

        # 3. DO NOT search arbitrary standalone uppercase letters in tail (prevents matching English 'a')
        # Returning None guarantees System 1 fallback executes correctly!
        return None

    def _extract_decision_logits(
        self,
        outputs: Any,
        prompt_len: int,
        gen_tokens: torch.Tensor,
        valid_labels: List[str],
        request: DecisionRequest,
        parsed_decision: Optional[str],
        reasoning_trace: str,
        prompt: str,
    ) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]], Optional[float]]:
        if not valid_labels:
            return None, None, None

        keys = list(request.options)

        cand_id_pairs = []
        all_candidate_ids = set()
        for lbl in valid_labels:
            s_lbl = str(lbl).strip()
            enc_sp = self.tokenizer.encode(f" {s_lbl}", add_special_tokens=False)
            enc_nsp = self.tokenizer.encode(s_lbl, add_special_tokens=False)
            id_sp = enc_sp[0] if enc_sp else None
            id_nsp = enc_nsp[0] if enc_nsp else None
            cand_id_pairs.append((id_sp, id_nsp))
            if id_sp is not None:
                all_candidate_ids.add(id_sp)
            if id_nsp is not None:
                all_candidate_ids.add(id_nsp)

        step_logits: Optional[torch.Tensor] = None
        scores = getattr(outputs, "scores", None)

        if scores and len(scores) > 0:
            num_scores = len(scores)
            T = len(gen_tokens)
            target_step: Optional[int] = None

            max_lookback = min(num_scores, 40)
            for offset in range(max_lookback):
                step_idx = num_scores - 1 - offset
                tok_idx = T - 1 - offset if T > offset else step_idx
                if tok_idx < len(gen_tokens):
                    tok_id = int(gen_tokens[tok_idx].item())
                    if tok_id in all_candidate_ids:
                        target_step = step_idx
                        break

            if target_step is None:
                m_ans = re.search(r"<answer>\s*", reasoning_trace, re.IGNORECASE)
                if m_ans:
                    char_target = m_ans.end()
                    accum_chars = 0
                    for s_idx in range(min(num_scores, T)):
                        t_str = self.tokenizer.decode([gen_tokens[s_idx]], skip_special_tokens=True)
                        accum_chars += len(t_str)
                        if accum_chars >= char_target:
                            target_step = s_idx
                            break

            if target_step is not None and 0 <= target_step < num_scores:
                s_log = scores[target_step]
                step_logits = s_log if s_log.ndim == 2 else s_log.unsqueeze(0)

        if step_logits is None and self.model is not None and hasattr(self.model, "forward"):
            try:
                m_ans = re.search(r"(<answer>.*)", reasoning_trace, re.IGNORECASE | re.DOTALL)
                prefix_trace = reasoning_trace[:m_ans.start()].rstrip() if m_ans else reasoning_trace.rstrip()
                eval_prompt = f"{prompt.rstrip()}\n{prefix_trace}\n<answer>"
                eval_inputs = self.tokenizer(eval_prompt, return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    try:
                        eval_out = self.model(**eval_inputs, logits_to_keep=1, use_cache=False)
                        last_logits = eval_out.logits[:, -1, :]
                    except TypeError:
                        eval_out = self.model(**eval_inputs, use_cache=False)
                        last_logits = eval_out.logits[:, -1, :]
                    step_logits = last_logits if last_logits.ndim == 2 else last_logits.unsqueeze(0)
            except Exception:
                step_logits = None

        if step_logits is None:
            return None, None, None

        target_device = step_logits.device
        chosen_cand_ids: List[int] = []
        for id_sp, id_nsp in cand_id_pairs:
            if id_sp is not None and id_nsp is not None:
                l_sp = step_logits[0, id_sp].item()
                l_nsp = step_logits[0, id_nsp].item()
                chosen_cand_ids.append(id_sp if l_sp >= l_nsp else id_nsp)
            elif id_sp is not None:
                chosen_cand_ids.append(id_sp)
            elif id_nsp is not None:
                chosen_cand_ids.append(id_nsp)
            else:
                chosen_cand_ids.append(self.tokenizer.unk_token_id or 0)

        cand_tensor = torch.tensor([chosen_cand_ids], device=target_device)
        cand_logits = torch.gather(step_logits, 1, cand_tensor)

        if self.use_calibrator_temperature and self.calibrator is not None:
            calibrated_probs = self.calibrator.calibrate(cand_logits, request.decision_type)
        else:
            calibrated_probs = F.softmax(cand_logits, dim=-1)

        prob_dict = {
            keys[i]: float(calibrated_probs[0, i].item())
            for i in range(min(len(keys), calibrated_probs.shape[1]))
        }
        raw_logits_dict = {
            keys[i]: float(cand_logits[0, i].item())
            for i in range(min(len(keys), cand_logits.shape[1]))
        }

        sorted_probs = sorted(prob_dict.values(), reverse=True)
        confidence = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else float(sorted_probs[0])
        confidence = float(np.clip(confidence, 0.0, 1.0))
        return prob_dict, raw_logits_dict, confidence

    @torch.inference_mode()
    def decide(
        self,
        request: DecisionRequest,
        fallback: Optional[DecisionResult] = None,
    ) -> DecisionResult:
        cache_key = self._get_cache_key(request)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key].model_copy(deep=True)

        start_time = time.perf_counter()
        prompt, valid_labels = self._format_chat_prompt(request)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        stop_criteria = StopStringCriteria(
            tokenizer=self.tokenizer,
            stop_strings=["</answer>", "</Answer>", "</ANSWER>"],
        )
        stopping_criteria = StoppingCriteriaList([stop_criteria])

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        seq = outputs.sequences[0] if hasattr(outputs, "sequences") else outputs[0]
        gen_tokens = seq[prompt_len:]
        reasoning_trace = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        latency = (time.perf_counter() - start_time) * 1000.0

        decision = self._parse_answer(reasoning_trace, request)

        # Clean fallback activation when System 2 generation fails to produce a valid option
        if decision is None and fallback is not None:
            res = DecisionResult(
                decision=fallback.decision,
                confidence=fallback.confidence,
                system_used="system_1_fallback",
                probabilities=fallback.probabilities,
                raw_logits=fallback.raw_logits,
                latency_ms=latency,
                reasoning_trace=f"[S2 Parse Failed -> Defaulted to S1 Fallback: {fallback.decision}]\n{reasoning_trace}",
                telemetry=fallback.telemetry,
            )
        else:
            default_val = fallback.decision if fallback is not None else request.options[0]
            final_decision = decision if decision is not None else default_val

            prob_dict, raw_logits_dict, conf = self._extract_decision_logits(
                outputs=outputs,
                prompt_len=prompt_len,
                gen_tokens=gen_tokens,
                valid_labels=valid_labels,
                request=request,
                parsed_decision=final_decision,
                reasoning_trace=reasoning_trace,
                prompt=prompt,
            )

            if prob_dict is None:
                if fallback is not None and fallback.probabilities:
                    prob_dict = dict(fallback.probabilities)
                    raw_logits_dict = fallback.raw_logits
                    conf = fallback.confidence
                else:
                    raw_logits_dict = None
                    if decision is not None:
                        conf = 0.85
                        keys = list(request.options)
                        rem = (1.0 - conf) / max(1, len(keys) - 1)
                        prob_dict = {k: (conf if k == final_decision else rem) for k in keys}
                    else:
                        conf = 0.25
                        prob_dict = {}

            res = DecisionResult(
                decision=final_decision,
                confidence=float(conf),
                system_used="system_2",
                probabilities=prob_dict,
                raw_logits=raw_logits_dict,
                latency_ms=latency,
                reasoning_trace=reasoning_trace,
                telemetry=fallback.telemetry if fallback else None,
            )

        self.warm_cache(request, res)
        return res
