#!/usr/bin/env python3
"""Compare top-token log probabilities from OpenRouter and Hugging Face models."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
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
    top1_match_rate: float
    avg_overlap_count: float
    avg_overlap_ratio: float
    avg_openrouter_top1_prob: float
    avg_hf_top1_prob: float
    avg_openrouter_top1_logprob: float
    avg_hf_top1_logprob: float
    avg_prob_diff: float


_HF_MODEL_CACHE: dict[tuple[str, str | None], tuple[Any, Any]] = {}


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
) -> ModelResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": top_k,
    }
    request = urllib.request.Request(
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

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter returned HTTP {error.code}: {details}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach OpenRouter: {error.reason}") from error

    try:
        choice = body["choices"][0]
        content_logprobs = choice["logprobs"]["content"]
        generated_text = choice["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            "OpenRouter response did not contain token logprobs. "
            f"The selected model may not support them: {body}"
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
                messages = [{"role": "user", "content": prompt}]
                if reference_prefix:
                    messages.append(
                        {"role": "assistant", "content": reference_prefix}
                    )
                    local_prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        continue_final_message=True,
                    )
                else:
                    local_prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            else:
                local_prompt = prompt + reference_prefix

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
    all_step_stats: list[StepStats], total_queries: int
) -> SummaryStats:
    if not all_step_stats:
        return SummaryStats(
            total_queries=total_queries,
            total_steps=0,
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
    print(f"Reference continuation (OpenRouter): {left.generated_text!r}")
    if right.teacher_forced:
        print("Hugging Face was teacher-forced along that continuation.")
    else:
        print(f"Hugging Face generated: {right.generated_text!r}")

    step_count = max(len(left.steps), len(right.steps))
    for step_index in range(step_count):
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
    parser.add_argument("--hf-model", required=True, help="Hub model ID or local path")
    parser.add_argument("-k", "--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument(
        "--device",
        help=(
            "PyTorch device such as cpu, cuda, or cuda:1; auto (the default) "
            "shards across multiple CUDA GPUs"
        ),
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    if not 1 <= args.top_k <= 20:
        parser.error("--top-k must be between 1 and 20 (OpenRouter API limit)")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
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

    summary_stats = compute_summary_stats(all_step_stats, total_queries=len(prompts))
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