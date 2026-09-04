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

    steps = []
    for item in content_logprobs:
        top_tokens = [
            TokenLogprob(token=entry["token"], logprob=float(entry["logprob"]))
            for entry in item.get("top_logprobs", [])
        ]
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
    device: str | None,
    reference_tokens: list[str] | None = None,
) -> ModelResult:
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
    input_device = (
        target_device
        if target_device is not None
        else model.get_input_embeddings().weight.device
    )

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


def save_json(path: Path, prompt: str, left: ModelResult, right: ModelResult) -> None:
    output: dict[str, Any] = {
        "prompt": prompt,
        "note": (
            "The OpenRouter output is the reference continuation. Hugging Face is "
            "teacher-forced along the same accumulated text. Token strings are "
            "decoded with different tokenizers, so exact token overlap is only a "
            "surface-form comparison."
        ),
        "results": [asdict(left), asdict(right)],
    }
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare top-token log probabilities from an OpenRouter model and a "
            "local Hugging Face causal language model."
        )
    )
    parser.add_argument("prompt", help="Prompt supplied to both models")
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

    try:
        openrouter_result = query_openrouter(
            args.openrouter_model,
            args.prompt,
            args.top_k,
            args.max_new_tokens,
            api_key,
        )
        huggingface_result = query_huggingface(
            args.hf_model,
            args.prompt,
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
    if args.json_output:
        save_json(
            args.json_output,
            args.prompt,
            openrouter_result,
            huggingface_result,
        )
        print(f"\nWrote machine-readable results to {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())