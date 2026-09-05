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
(default: 16384). These are new billable requests, not continuations of the
same generation, so lower the hard cap when controlling cost is more important.
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
  --max-new-tokens 3 \
  --max-openrouter-tokens 16384 \
  --json-output comparison.json

# If prompt is omitted, it evaluates all EXAMPLE_QUERIES and computes average stats:
python compare_logprobs.py \
  --openrouter-model openai/gpt-4.1-mini \
  --hf-model Qwen/Qwen2.5-1.5B-Instruct \
  --top-k 20 \
  --max-new-tokens 3 \
  --json-output comparison.json
```

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
