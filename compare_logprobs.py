#!/usr/bin/env python3
"""Compare top-token log probabilities from OpenRouter and Hugging Face models."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

EXAMPLE_QUERIES = (
    "Explain why the daytime sky is blue but sunsets often appear red. Connect the explanation to scattering, wavelength, and the distance sunlight travels through the atmosphere.",
    "A city wants to reduce downtown traffic without making commuting harder for low-income workers. Compare congestion pricing, improved public transit, and parking restrictions, then recommend a phased policy with safeguards and measurable success criteria.",
    "Maya manages a project originally due Friday. The client moves it to Wednesday, an engineer reports a two-day blocker, and a required reviewer is unavailable Tuesday. Develop a realistic recovery plan, identify assumptions, and explain what Maya should communicate to each stakeholder.",
    "A company claims that productivity increased after employees returned to the office, so remote work must reduce productivity. Critique this inference, propose plausible confounders, and design a stronger evaluation that could support a causal conclusion.",
    "Design a fair procedure for allocating five emergency shelter beds among twelve eligible people when needs differ and information is incomplete. Explain the values behind your procedure and how appeals or new evidence should be handled.",
    "Fred took his fishing pole to the bank of a river. Later, a friend texted that she would meet him at the bank to discuss a loan. Analyze the ambiguity, explain which interpretation each person may hold, and propose a message that prevents a costly misunderstanding.",
    "All roses are flowers, some flowers fade quickly, and no quickly fading plant survives a frost. Explain exactly what can and cannot be inferred about roses, then give two additional premises that would support different conclusions.",
    "A small software team must choose between shipping a fragile feature this week or delaying it for testing while a competitor is launching a similar product. Build a decision framework, evaluate the main risks, and recommend a course of action under clearly stated assumptions.",
    "Plan how to prepare tea for six guests when there is one kettle, four clean mugs, two dirty mugs, and one guest avoids caffeine. Include ordering, resource constraints, and a contingency if the kettle stops working.",
    "A store is considering a 25% discount followed by a loyalty reward, but margins are thin and customers respond differently to promotions. Explain how the store should evaluate profitability, customer behavior, and long-term effects before choosing a promotion design.",
    "A coastal town must decide whether to rebuild a storm-damaged seawall, restore wetlands, or relocate the most exposed homes. Compare the options across cost, resilience, fairness, and uncertainty, then propose a decision process that can adapt as conditions change.",
    "A hospital has fewer intensive-care beds than patients likely to need them during an outbreak. Design a transparent allocation policy, explain how it handles changing prognoses and ties, and identify safeguards against bias and avoidable harm.",
    "Two departments report conflicting results from the same customer survey: one says satisfaction improved, while the other says complaints became more severe. Explain how both claims could be true and outline an analysis that would reconcile the evidence.",
    "A teacher discovers that students are using generative AI for homework, but the school has no clear policy. Develop a response that supports learning, treats students fairly, and distinguishes acceptable assistance from work that misrepresents understanding.",
    "An old bridge is still considered safe but requires increasingly frequent repairs. Compare continued maintenance, major rehabilitation, and replacement while accounting for disruption, uncertain future demand, public safety, and budget constraints.",
    "A neighborhood wants more housing but disagrees about building height, affordability requirements, parking, and preservation of local businesses. Propose a negotiation framework and a compromise plan, including who bears each cost and how outcomes should be measured.",
    "A research team finds a statistically significant effect that is much smaller than expected and disappears under one reasonable analysis choice. Interpret the result, identify what should be reported, and recommend the next study without reducing the decision to a single p-value.",
    "A family must choose between caring for an aging relative at home, hiring in-home support, or moving them to assisted living. Build a respectful decision process that considers autonomy, safety, finances, caregiver capacity, and how the plan should be revisited over time.",
    "A news platform wants to reduce misinformation without suppressing legitimate disagreement or breaking-news updates that later change. Design a moderation approach that combines labels, distribution rules, appeals, and evidence standards, then explain its likely failure modes.",
    "A manufacturer can lower emissions by replacing equipment now, purchasing cleaner electricity, or waiting for a promising technology still under development. Recommend a staged strategy using plausible assumptions about cost, risk, and regulation, and specify signals that would trigger a change in course.",
)


@dataclass(frozen=True)
class TokenLogprob:
    token: str
    logprob: float

    @property
    def probability(self) -> float:
        return math.exp(self.logprob)


@dataclass(frozen=True)
class GenerationStep:
    generated_token: str
    top_tokens: list[TokenLogprob]


@dataclass(frozen=True)
class ModelResult:
    provider: str
    model: str
    generated_text: str
    steps: list[GenerationStep]
    teacher_forced: bool = False
    reasoning_text: str = ""
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class StepStats:
    top1_match: bool
    overlap_count: int
    overlap_ratio: float
    openrouter_top1_prob: float
    hf_top1_prob: float
    openrouter_top1_logprob: float
    hf_top1_logprob: float
    prob_diff: float
    hf_reference_rank: int | None = None
    hf_reference_prob: float | None = None
    hf_reference_logprob: float | None = None


@dataclass(frozen=True)
class QueryComparison:
    prompt: str
    openrouter_result: ModelResult
    huggingface_result: ModelResult
    step_stats: list[StepStats]


@dataclass(frozen=True)
class SummaryStats:
    total_queries: int
    total_steps: int
    avg_reasoning_tokens: float | None
    top1_match_rate: float
    avg_overlap_count: float
    avg_overlap_ratio: float
    avg_openrouter_top1_prob: float
    avg_hf_top1_prob: float
    avg_openrouter_top1_logprob: float
    avg_hf_top1_logprob: float
    avg_prob_diff: float


_HF_MODEL_CACHE: dict[tuple[str, str | None], tuple[Any, Any]] = {}
_OPENROUTER_MAX_ATTEMPTS = 4
_RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504, 529}


def _retry_delay(error: urllib.error.HTTPError, attempt: int) -> float:
    retry_after = error.headers.get("Retry-After") if error.headers else None
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    return float(2 ** (attempt - 1))


def _provider_top_logprobs_limit(details: str) -> int | None:
    match = re.search(r"top_logprobs.*?\[0,\s*(\d+)\]", details, re.DOTALL)
    return int(match.group(1)) if match else None


def _openrouter_request(payload: dict[str, Any], api_key: str) -> urllib.request.Request:
    return urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/askerlee/mermed",
            "X-Title": "mermed logprob comparison",
        },
        method="POST",
    )


def _reasoning_text(message: dict[str, Any]) -> str:
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str):
        return reasoning
    details = message.get("reasoning_details") or []
    parts = []
    for detail in details:
        if not isinstance(detail, dict):
            continue
        text = detail.get("text") or detail.get("summary")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def _reasoning_prefill(reasoning_text: str, visible_prefix: str) -> str:
    reasoning_text = reasoning_text.strip()
    if not reasoning_text:
        return visible_prefix
    if reasoning_text.startswith("<think>"):
        return reasoning_text + visible_prefix
    return f"<think>\n{reasoning_text}\n</think>\n\n{visible_prefix}"


def _render_huggingface_prompt(
    tokenizer: Any,
    prompt: str,
    reasoning_text: str,
    visible_prefix: str,
) -> str:
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    if not reasoning_text and not visible_prefix:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    sentinel = "<|mermed_prefill_end|>"
    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": visible_prefix + sentinel,
    }
    if reasoning_text:
        assistant["reasoning_content"] = reasoning_text
    messages.append(assistant)
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        continue_final_message=True,
    )
    if reasoning_text and reasoning_text not in rendered:
        messages[-1] = {
            "role": "assistant",
            "content": _reasoning_prefill(reasoning_text, visible_prefix) + sentinel,
        }
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=True,
        )
    if not rendered.endswith(sentinel):
        raise RuntimeError("The Hugging Face chat template could not preserve prefill text")
    return rendered[: -len(sentinel)]


def load_huggingface_model(
    model_name_or_path: str,
    device: str | None = None,
) -> tuple[Any, Any]:
    cache_key = (model_name_or_path, device)
    if cache_key in _HF_MODEL_CACHE:
        return _HF_MODEL_CACHE[cache_key]

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "The local provider requires torch, transformers, and accelerate. "
            "Install them with: pip install torch transformers accelerate"
        ) from error

    placement_options, target_device = _huggingface_placement(torch, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        **placement_options,
    )
    if target_device is not None:
        model.to(target_device)
    model.eval()

    _HF_MODEL_CACHE[cache_key] = (tokenizer, model)
    return tokenizer, model


def _huggingface_placement(
    torch_module: Any, device: str | None
) -> tuple[dict[str, Any], str | None]:
    automatic = device is None or device == "auto"
    if automatic and torch_module.cuda.device_count() > 1:
        return {"device_map": "auto"}, None
    if automatic:
        if torch_module.cuda.is_available():
            return {}, "cuda"
        if (
            hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        ):
            return {}, "mps"
        return {}, "cpu"
    return {}, device


def query_openrouter(
    model: str,
    prompt: str,
    top_k: int,
    max_new_tokens: int,
    api_key: str,
    provider: str | None = None,
    max_openrouter_tokens: int | None = None,
    max_reasoning_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> ModelResult:
    initial_token_budget = max_new_tokens + (max_reasoning_tokens or 0)
    token_ceiling = max_openrouter_tokens or initial_token_budget
    if initial_token_budget > token_ceiling:
        raise RuntimeError(
            "The OpenRouter hard cap must cover --max-new-tokens plus "
            "--max-reasoning-tokens"
        )
    provider_preferences: dict[str, Any] = {"require_parameters": True}
    if provider:
        provider_preferences["only"] = [provider]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": initial_token_budget,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": top_k,
        "provider": provider_preferences,
    }
    if max_reasoning_tokens is not None:
        payload["reasoning"] = {"max_tokens": max_reasoning_tokens}
    elif reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    request = _openrouter_request(payload, api_key)

    transient_failures = 0
    adapted_top_k = False
    while True:
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.load(response)
            choices = body.get("choices") if isinstance(body, dict) else None
            choice = choices[0] if choices else None
            message = choice.get("message") if isinstance(choice, dict) else None
            reasoning_only = (
                isinstance(message, dict)
                and bool(_reasoning_text(message))
                and not message.get("content")
            )
            if (
                reasoning_only
                and choice.get("finish_reason") == "length"
                and max_reasoning_tokens is None
                and reasoning_effort is None
                and payload["max_tokens"] < token_ceiling
            ):
                next_budget = min(payload["max_tokens"] * 2, token_ceiling)
                print(
                    "OpenRouter exhausted "
                    f"{payload['max_tokens']} tokens on reasoning; retrying from "
                    f"scratch with {next_budget} (hard cap: {token_ceiling})",
                    file=sys.stderr,
                )
                payload["max_tokens"] = next_budget
                request = _openrouter_request(payload, api_key)
                transient_failures = 0
                continue
            break
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            provider_top_k = _provider_top_logprobs_limit(details)
            if (
                error.code == 400
                and not adapted_top_k
                and provider_top_k is not None
                and provider_top_k < payload["top_logprobs"]
            ):
                print(
                    f"OpenRouter provider limits top_logprobs to {provider_top_k}; "
                    "retrying with that limit",
                    file=sys.stderr,
                )
                payload["top_logprobs"] = provider_top_k
                request = _openrouter_request(payload, api_key)
                adapted_top_k = True
                continue
            if (
                error.code not in _RETRYABLE_HTTP_CODES
                or transient_failures == _OPENROUTER_MAX_ATTEMPTS - 1
            ):
                raise RuntimeError(
                    f"OpenRouter returned HTTP {error.code}: {details}"
                ) from error
            transient_failures += 1
            delay = _retry_delay(error, transient_failures)
            print(
                f"OpenRouter returned HTTP {error.code}; retrying in {delay:g}s "
                f"({transient_failures}/{_OPENROUTER_MAX_ATTEMPTS - 1})",
                file=sys.stderr,
            )
            time.sleep(delay)
        except urllib.error.URLError as error:
            raise RuntimeError(f"Could not reach OpenRouter: {error.reason}") from error

    try:
        choice = body["choices"][0]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            f"OpenRouter returned an invalid chat completion: {body}"
        ) from error
    if choice.get("logprobs") is None:
        provider = body.get("provider", "unknown provider")
        message = choice.get("message") or {}
        reasoning_only = message.get("reasoning") and not message.get("content")
        if reasoning_only:
            if max_reasoning_tokens is not None or reasoning_effort is not None:
                reasoning_control = (
                    f"the requested {max_reasoning_tokens}-token reasoning cap"
                    if max_reasoning_tokens is not None
                    else f"reasoning effort {reasoning_effort!r}"
                )
                detail = (
                    " The response contained reasoning but no visible output tokens. "
                    f"The provider did not finish reasoning with {reasoning_control}; "
                    f"the fixed "
                    f"{initial_token_budget}-token request was not retried."
                )
            else:
                detail = (
                    " The response contained reasoning but no visible output tokens; "
                    f"the {token_ceiling}-token OpenRouter hard cap was reached."
                )
        else:
            detail = " Choose a model with a logprob-capable provider."
        raise RuntimeError(
            f"OpenRouter provider {provider!r} did not return token logprobs.{detail}"
        )
    try:
        content_logprobs = choice["logprobs"]["content"][:max_new_tokens]
        message = choice["message"]
        generated_text = "".join(item["token"] for item in content_logprobs)
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"OpenRouter returned malformed token logprobs: {choice.get('logprobs')}"
        ) from error
    if not content_logprobs:
        raise RuntimeError("OpenRouter returned no generated-token logprobs")

    steps = []
    for item in content_logprobs:
        top_tokens = [
            TokenLogprob(token=entry["token"], logprob=float(entry["logprob"]))
            for entry in item.get("top_logprobs", [])
        ]
        if not top_tokens:
            raise RuntimeError(
                "OpenRouter returned no top-token logprobs for a generated token. "
                "The selected model or provider may not support them."
            )
        steps.append(
            GenerationStep(
                generated_token=item["token"],
                top_tokens=top_tokens,
            )
        )

    return ModelResult(
        provider="openrouter",
        model=model,
        generated_text=generated_text,
        steps=steps,
        reasoning_text=_reasoning_text(message),
        reasoning_tokens=(
            body.get("usage", {})
            .get("completion_tokens_details", {})
            .get("reasoning_tokens")
        ),
    )


def query_huggingface(
    model_name_or_path: str,
    prompt: str,
    top_k: int,
    max_new_tokens: int,
    device: str | None = None,
    reference_tokens: list[str] | None = None,
    tokenizer: Any = None,
    model: Any = None,
    reasoning_text: str = "",
) -> ModelResult:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "The local provider requires torch. Install it with: pip install torch"
        ) from error

    if tokenizer is None or model is None:
        cached_tokenizer, cached_model = load_huggingface_model(
            model_name_or_path, device
        )
        if tokenizer is None:
            tokenizer = cached_tokenizer
        if model is None:
            model = cached_model

    input_device = model.get_input_embeddings().weight.device

    generated_ids: list[int] = []
    steps: list[GenerationStep] = []
    reference_prefix = ""
    step_count = len(reference_tokens) if reference_tokens is not None else max_new_tokens
    with torch.inference_mode():
        for step_index in range(step_count):
            if tokenizer.chat_template:
                local_prompt = _render_huggingface_prompt(
                    tokenizer,
                    prompt,
                    reasoning_text,
                    reference_prefix,
                )
            else:
                local_prompt = prompt + _reasoning_prefill(
                    reasoning_text, reference_prefix
                )

            encoded = tokenizer(
                local_prompt, return_tensors="pt", add_special_tokens=True
            )
            input_ids = encoded["input_ids"].to(input_device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(input_device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logprobs = torch.log_softmax(outputs.logits[0, -1].float(), dim=-1)
            count = min(top_k, logprobs.shape[-1])
            values, token_ids = torch.topk(logprobs, k=count)
            next_token_id = int(token_ids[0].item())

            top_tokens = [
                TokenLogprob(
                    token=tokenizer.decode([int(token_id)]),
                    logprob=float(value),
                )
                for value, token_id in zip(values.tolist(), token_ids.tolist())
            ]
            generated_token = tokenizer.decode([next_token_id])
            steps.append(
                GenerationStep(
                    generated_token=generated_token,
                    top_tokens=top_tokens,
                )
            )
            generated_ids.append(next_token_id)

            if reference_tokens is not None:
                reference_prefix += reference_tokens[step_index]
                continue

            if next_token_id == tokenizer.eos_token_id:
                break
            reference_prefix += generated_token

    return ModelResult(
        provider="huggingface",
        model=model_name_or_path,
        generated_text=(
            reference_prefix
            if reference_tokens is not None
            else tokenizer.decode(generated_ids)
        ),
        steps=steps,
        teacher_forced=reference_tokens is not None,
        reasoning_text=reasoning_text,
    )


def _token_label(token: str) -> str:
    return repr(token)


def compute_step_stats(
    left_step: GenerationStep,
    right_step: GenerationStep,
    top_k: int,
) -> StepStats:
    if not left_step.top_tokens or not right_step.top_tokens:
        raise ValueError("Both generation steps must contain ranked tokens")

    top1_match = left_step.top_tokens[0].token == right_step.top_tokens[0].token
    left_tokens = {item.token for item in left_step.top_tokens}
    right_tokens = {item.token for item in right_step.top_tokens}
    overlap = left_tokens & right_tokens
    overlap_count = len(overlap)
    comparable_token_count = min(len(left_tokens), len(right_tokens), top_k)
    overlap_ratio = (
        overlap_count / comparable_token_count if comparable_token_count else 0.0
    )

    or_top1_prob = left_step.top_tokens[0].probability
    or_top1_logprob = left_step.top_tokens[0].logprob
    hf_top1_prob = right_step.top_tokens[0].probability
    hf_top1_logprob = right_step.top_tokens[0].logprob

    ref_token = left_step.generated_token
    hf_rank = None
    hf_ref_prob = None
    hf_ref_logprob = None
    for rank, item in enumerate(right_step.top_tokens, start=1):
        if item.token == ref_token:
            hf_rank = rank
            hf_ref_prob = item.probability
            hf_ref_logprob = item.logprob
            break

    return StepStats(
        top1_match=top1_match,
        overlap_count=overlap_count,
        overlap_ratio=overlap_ratio,
        openrouter_top1_prob=or_top1_prob,
        hf_top1_prob=hf_top1_prob,
        openrouter_top1_logprob=or_top1_logprob,
        hf_top1_logprob=hf_top1_logprob,
        prob_diff=abs(or_top1_prob - hf_top1_prob),
        hf_reference_rank=hf_rank,
        hf_reference_prob=hf_ref_prob,
        hf_reference_logprob=hf_ref_logprob,
    )


def compute_summary_stats(
    all_step_stats: list[StepStats],
    total_queries: int,
    reasoning_token_counts: list[int | None] | None = None,
) -> SummaryStats:
    reported_reasoning_counts = [
        count for count in (reasoning_token_counts or []) if count is not None
    ]
    avg_reasoning_tokens = (
        sum(reported_reasoning_counts) / len(reported_reasoning_counts)
        if reported_reasoning_counts
        else None
    )
    if not all_step_stats:
        return SummaryStats(
            total_queries=total_queries,
            total_steps=0,
            avg_reasoning_tokens=avg_reasoning_tokens,
            top1_match_rate=0.0,
            avg_overlap_count=0.0,
            avg_overlap_ratio=0.0,
            avg_openrouter_top1_prob=0.0,
            avg_hf_top1_prob=0.0,
            avg_openrouter_top1_logprob=0.0,
            avg_hf_top1_logprob=0.0,
            avg_prob_diff=0.0,
        )
    n = len(all_step_stats)
    return SummaryStats(
        total_queries=total_queries,
        total_steps=n,
        avg_reasoning_tokens=avg_reasoning_tokens,
        top1_match_rate=sum(1 for s in all_step_stats if s.top1_match) / n,
        avg_overlap_count=sum(s.overlap_count for s in all_step_stats) / n,
        avg_overlap_ratio=sum(s.overlap_ratio for s in all_step_stats) / n,
        avg_openrouter_top1_prob=sum(s.openrouter_top1_prob for s in all_step_stats) / n,
        avg_hf_top1_prob=sum(s.hf_top1_prob for s in all_step_stats) / n,
        avg_openrouter_top1_logprob=sum(s.openrouter_top1_logprob for s in all_step_stats) / n,
        avg_hf_top1_logprob=sum(s.hf_top1_logprob for s in all_step_stats) / n,
        avg_prob_diff=sum(s.prob_diff for s in all_step_stats) / n,
    )


def print_summary_stats(stats: SummaryStats, top_k: int) -> None:
    print(f"\n{'=' * 50}")
    print("AVERAGE STATS SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total queries evaluated:         {stats.total_queries}")
    print(f"Total generation steps:          {stats.total_steps}")
    print(f"Top-k used:                      {top_k}")
    reasoning_average = (
        f"{stats.avg_reasoning_tokens:.2f}"
        if stats.avg_reasoning_tokens is not None
        else "N/A"
    )
    print(f"Average reasoning tokens:        {reasoning_average}")
    match_count = int(round(stats.top1_match_rate * stats.total_steps))
    print(
        f"Top-1 match rate:                {stats.top1_match_rate:.2%} "
        f"({match_count}/{stats.total_steps})"
    )
    print(
        f"Average top-{top_k} token overlap:     {stats.avg_overlap_count:.2f} "
        f"({stats.avg_overlap_ratio:.2%} of available ranks)"
    )
    print(f"Average OpenRouter top-1 prob:   {stats.avg_openrouter_top1_prob:.4f}")
    print(f"Average HF top-1 prob:           {stats.avg_hf_top1_prob:.4f}")
    print(f"Average top-1 prob diff:         {stats.avg_prob_diff:.4f}")
    print(f"Average OpenRouter top-1 logprob:{stats.avg_openrouter_top1_logprob:>8.4f}")
    print(f"Average HF top-1 logprob:        {stats.avg_hf_top1_logprob:>8.4f}")


def print_comparison(left: ModelResult, right: ModelResult) -> None:
    print(f"OpenRouter:   {left.model}")
    print(f"Hugging Face: {right.model}")
    step_count = max(len(left.steps), len(right.steps))
    if step_count > 20:
        first_text = "".join(step.generated_token for step in left.steps[:10])
        last_text = "".join(step.generated_token for step in left.steps[-10:])
        print(
            "Reference continuation (OpenRouter): "
            f"{first_text!r} ... [{step_count - 20} tokens omitted] ... {last_text!r}"
        )
    else:
        print(f"Reference continuation (OpenRouter): {left.generated_text!r}")
    if right.teacher_forced:
        print("Hugging Face was teacher-forced along that continuation.")
    elif len(right.steps) > 20:
        first_text = "".join(step.generated_token for step in right.steps[:10])
        last_text = "".join(step.generated_token for step in right.steps[-10:])
        print(
            "Hugging Face generated: "
            f"{first_text!r} ... [{len(right.steps) - 20} tokens omitted] ... "
            f"{last_text!r}"
        )
    else:
        print(f"Hugging Face generated: {right.generated_text!r}")

    if step_count > 20:
        step_indexes = [*range(10), *range(step_count - 10, step_count)]
    else:
        step_indexes = range(step_count)
    for position, step_index in enumerate(step_indexes):
        if step_count > 20 and position == 10:
            print(f"\n... {step_count - 20} generation steps omitted ...")
        print(f"\n=== Generation step {step_index + 1} ===")
        left_step = left.steps[step_index] if step_index < len(left.steps) else None
        right_step = right.steps[step_index] if step_index < len(right.steps) else None

        if left_step and right_step:
            left_tokens = {item.token for item in left_step.top_tokens}
            right_tokens = {item.token for item in right_step.top_tokens}
            overlap = left_tokens & right_tokens
            print(
                f"reference token: {_token_label(left_step.generated_token)}; "
                f"HF top prediction: {_token_label(right_step.generated_token)}; "
                f"exact decoded-token overlap: {len(overlap)}"
            )

        left_top = left_step.top_tokens if left_step else []
        right_top = right_step.top_tokens if right_step else []
        print(
            f"{'rank':>4}  {'OpenRouter token':<24} {'logprob':>10} {'prob':>9} | "
            f"{'HF token':<24} {'logprob':>10} {'prob':>9}"
        )
        print("-" * 103)
        for rank in range(max(len(left_top), len(right_top))):
            left_item = left_top[rank] if rank < len(left_top) else None
            right_item = right_top[rank] if rank < len(right_top) else None
            left_columns = (
                f"{_token_label(left_item.token):<24.24} "
                f"{left_item.logprob:>10.4f} {left_item.probability:>9.5f}"
                if left_item
                else " " * 45
            )
            right_columns = (
                f"{_token_label(right_item.token):<24.24} "
                f"{right_item.logprob:>10.4f} {right_item.probability:>9.5f}"
                if right_item
                else ""
            )
            print(f"{rank + 1:>4}  {left_columns} | {right_columns}")


def save_json(
    path: Path,
    comparisons: list[QueryComparison],
    summary_stats: SummaryStats,
) -> None:
    if len(comparisons) == 1:
        comp = comparisons[0]
        output: dict[str, Any] = {
            "prompt": comp.prompt,
            "note": (
                "The OpenRouter output is the reference continuation. Hugging Face is "
                "teacher-forced along the same accumulated text. Token strings are "
                "decoded with different tokenizers, so exact token overlap is only a "
                "surface-form comparison."
            ),
            "results": [asdict(comp.openrouter_result), asdict(comp.huggingface_result)],
            "step_stats": [asdict(s) for s in comp.step_stats],
            "summary_stats": asdict(summary_stats),
        }
    else:
        output = {
            "note": (
                "The OpenRouter output is the reference continuation. Hugging Face is "
                "teacher-forced along the same accumulated text. Token strings are "
                "decoded with different tokenizers, so exact token overlap is only a "
                "surface-form comparison."
            ),
            "summary_stats": asdict(summary_stats),
            "queries": [
                {
                    "prompt": comp.prompt,
                    "results": [asdict(comp.openrouter_result), asdict(comp.huggingface_result)],
                    "step_stats": [asdict(s) for s in comp.step_stats],
                }
                for comp in comparisons
            ],
        }
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare top-token log probabilities from an OpenRouter model and a "
            "local Hugging Face causal language model."
        )
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Prompt supplied to both models (if omitted, loops over EXAMPLE_QUERIES)",
    )
    parser.add_argument("--openrouter-model", required=True)
    parser.add_argument(
        "--openrouter-provider",
        help="OpenRouter provider slug to use exclusively, such as fireworks",
    )
    parser.add_argument("--hf-model", required=True, help="Hub model ID or local path")
    parser.add_argument("-k", "--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    reasoning_group = parser.add_mutually_exclusive_group()
    reasoning_group.add_argument(
        "--max-reasoning-tokens",
        type=int,
        help=(
            "Cap OpenRouter reasoning tokens before generation; support varies "
            "by model and provider"
        ),
    )
    reasoning_group.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        help="OpenRouter reasoning effort (default: low)",
    )
    parser.add_argument(
        "--max-openrouter-tokens",
        type=int,
        default=16384,
        help=(
            "Hard cap for automatic OpenRouter budget growth when reasoning "
            "exhausts --max-new-tokens (default: 16384)"
        ),
    )
    parser.add_argument(
        "--device",
        help=(
            "PyTorch device such as cpu, cuda, or cuda:1; auto (the default) "
            "shards across multiple CUDA GPUs"
        ),
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    if args.max_reasoning_tokens is None and args.reasoning_effort is None:
        args.reasoning_effort = "low"

    if not 1 <= args.top_k <= 20:
        parser.error("--top-k must be between 1 and 20 (OpenRouter API limit)")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.max_reasoning_tokens is not None and args.max_reasoning_tokens < 1:
        parser.error("--max-reasoning-tokens must be at least 1")
    initial_openrouter_tokens = args.max_new_tokens + (
        args.max_reasoning_tokens or 0
    )
    if args.max_openrouter_tokens < initial_openrouter_tokens:
        parser.error(
            "--max-openrouter-tokens must be at least --max-new-tokens plus "
            "--max-reasoning-tokens"
        )
    return args


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("error: OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2

    prompts = [args.prompt] if args.prompt is not None else list(EXAMPLE_QUERIES)

    comparisons: list[QueryComparison] = []
    all_step_stats: list[StepStats] = []

    for idx, prompt in enumerate(prompts, start=1):
        if len(prompts) > 1:
            print(f"\n{'=' * 80}")
            print(f"[{idx}/{len(prompts)}] Prompt: {prompt}")
            print(f"{'=' * 80}")

        try:
            openrouter_result = query_openrouter(
                args.openrouter_model,
                prompt,
                args.top_k,
                args.max_new_tokens,
                api_key,
                args.openrouter_provider,
                args.max_openrouter_tokens,
                args.max_reasoning_tokens,
                args.reasoning_effort,
            )
            huggingface_result = query_huggingface(
                args.hf_model,
                prompt,
                args.top_k,
                args.max_new_tokens,
                args.device,
                reference_tokens=[
                    step.generated_token for step in openrouter_result.steps
                ],
                reasoning_text=openrouter_result.reasoning_text,
            )
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1

        print_comparison(openrouter_result, huggingface_result)

        query_steps: list[StepStats] = []
        step_count = min(len(openrouter_result.steps), len(huggingface_result.steps))
        for step_idx in range(step_count):
            s_stat = compute_step_stats(
                openrouter_result.steps[step_idx],
                huggingface_result.steps[step_idx],
                args.top_k,
            )
            query_steps.append(s_stat)
            all_step_stats.append(s_stat)

        comparisons.append(
            QueryComparison(
                prompt=prompt,
                openrouter_result=openrouter_result,
                huggingface_result=huggingface_result,
                step_stats=query_steps,
            )
        )

    summary_stats = compute_summary_stats(
        all_step_stats,
        total_queries=len(prompts),
        reasoning_token_counts=[
            comparison.openrouter_result.reasoning_tokens
            for comparison in comparisons
        ],
    )
    print_summary_stats(summary_stats, args.top_k)

    if args.json_output:
        save_json(
            args.json_output,
            comparisons,
            summary_stats,
        )
        print(f"\nWrote machine-readable results to {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())