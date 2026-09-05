# Top-token logprob comparison

`compare_logprobs.py` compares the top next-token log probabilities from an
OpenRouter chat model and a local Hugging Face causal language model.

## Setup

Use Python 3.10 or newer, then install the local-model dependencies:

```bash
python -m pip install -r requirements.txt
export OPENROUTER_API_KEY="..."
```

The selected OpenRouter model and provider must support token logprobs. The
request requires parameter support so OpenRouter excludes providers that would
silently ignore `logprobs`; if no compatible route exists, choose another
model. Models on the Hugging Face Hub are downloaded on first use; `--hf-model`
can instead point to a local model directory.

OpenRouter reasoning is allowed when the model requires or chooses it. The
reasoning trace is retained but excluded from similarity statistics because
OpenRouter supplies `logprobs.content` only for visible completion tokens. The
local model is prefixed with the same reasoning trace before it is
teacher-forced along those visible tokens. Models whose chat templates support
`reasoning_content` receive it structurally; other templates use
`<think>...</think>` markers as a fallback.

If a response reaches `--max-new-tokens` with reasoning but no visible output,
the script retries from scratch with twice the OpenRouter token budget. It
continues until visible output appears or `--max-openrouter-tokens` is reached
(default: 4100). These are new billable requests, not continuations of the
same generation, so lower the hard cap when controlling cost is more important.

By default, the script requests `--max-reasoning-tokens 4000` and reserves
`--max-new-tokens` in addition to that cap for visible output. Thus the default
sends one fixed 4100-token OpenRouter request while comparing at most 100
visible tokens. Automatic budget growth remains disabled. Exact token caps are
supported by Anthropic, Gemini thinking, and some Qwen thinking models; other
reasoning models may map the value to an approximate effort level.

Alternatively, use `--reasoning-effort` with `none`, `minimal`, `low`,
`medium`, `high`, `xhigh`, or `max`. The two reasoning controls are mutually
exclusive, and an explicit effort replaces the default numeric cap. The
combined budget must not exceed
`--max-openrouter-tokens`. If OpenRouter returns more visible tokens because it
uses less than its reasoning allowance, only the first `--max-new-tokens` are
teacher-forced, compared, printed, and saved.

When either reasoning control is set, the request uses a fixed budget and
automatic budget growth is disabled. If the selected provider returns no
visible tokens, the script skips that query instead of retrying with a larger,
separately billed request. Skipped queries are excluded from all summary
statistics, and the summary reports how many were skipped for overly long
reasoning.

For teacher-forced comparison, compatible Hugging Face tokenizations are scored
in one model forward pass, so a long reasoning prefill is not recomputed for
every visible token. Per-query progress messages report OpenRouter and local
scoring times separately. After each query's selected token details, a compact
query summary reports its reasoning tokens and comparison metrics. The complete
aggregate summary is printed after all queries finish.
Transient OpenRouter responses such as HTTP 429 and provider-side 5xx errors
are retried up to three times, respecting `Retry-After` when supplied.
If a provider reports a smaller `top_logprobs` limit than requested, the
OpenRouter request is retried once using that provider limit. The local model
still returns the requested `--top-k`; statistics use the ranks available from
both models.

## Usage

```bash
# Compare a single prompt:
python compare_logprobs.py \
  "The capital of France is" \
  --openrouter-model openai/gpt-4.1-mini \
  --openrouter-provider fireworks \
  --hf-model Qwen/Qwen2.5-1.5B-Instruct \
  --top-k 20 \
  --max-new-tokens 100 \
  --max-reasoning-tokens 4000 \
  --max-openrouter-tokens 4100

# If prompt is omitted, it evaluates all EXAMPLE_QUERIES and computes average stats:
python compare_logprobs.py \
  --openrouter-model openai/gpt-4.1-mini \
  --hf-model Qwen/Qwen2.5-1.5B-Instruct \
  --top-k 20 \
  --max-new-tokens 3

# Evaluate the 20 built-in medical prompts instead:
python compare_logprobs.py \
  --medical-queries \
  --openrouter-model openai/gpt-4.1-mini \
  --hf-model Qwen/Qwen2.5-1.5B-Instruct \
  --top-k 20 \
  --max-new-tokens 3
```

Results are written by default to a filename formed from the Hugging Face and
OpenRouter model slugs. The examples above write to
`qwen-qwen2-5-1-5b-instruct-openai-gpt-4-1-mini.json`. Use `--json-output` to
choose a different path.

Use `--openrouter-provider` to restrict routing to one provider, for example
`fireworks`, `morph`, or `digitalocean`. The provider must offer the selected
model and support every requested parameter. Omit the option to let OpenRouter
choose among compatible providers.

By default, the local model is sharded across all visible CUDA GPUs when more
than one is available. This uses Hugging Face Accelerate's `device_map="auto"`.
`--device auto` selects the same behavior explicitly. Use `--device cpu`,
`--device cuda`, or a specific device such as `--device cuda:1` to force the
entire model onto one device.

OpenRouter generates the reference continuation. At every OpenRouter token
boundary, the local model is evaluated after the same accumulated response
text. Its top prediction is recorded but not appended; the next OpenRouter
reference token is appended instead. This teacher forcing keeps the response
text shared even when the models disagree. Each provider still applies its own
model-specific chat template.

Token IDs are not comparable across different tokenizers. The displayed
overlap count compares exact decoded token strings only. Comparison checkpoints
use OpenRouter's token boundaries, which may not be token boundaries for the
Hugging Face model; the ranked logprobs and probabilities are the primary
output.
