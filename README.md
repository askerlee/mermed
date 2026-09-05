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

The OpenRouter request disables reasoning so `--max-new-tokens` is spent on
visible completion tokens. OpenRouter does not return content logprobs for
reasoning tokens, so those tokens cannot be used in this comparison.
Transient OpenRouter responses such as HTTP 429 and provider-side 5xx errors
are retried up to three times, respecting `Retry-After` when supplied.

## Usage

```bash
# Compare a single prompt:
python compare_logprobs.py \
  "The capital of France is" \
  --openrouter-model openai/gpt-4.1-mini \
  --hf-model Qwen/Qwen2.5-1.5B-Instruct \
  --top-k 20 \
  --max-new-tokens 3 \
  --json-output comparison.json

# If prompt is omitted, it evaluates all EXAMPLE_QUERIES and computes average stats:
python compare_logprobs.py \
  --openrouter-model openai/gpt-4.1-mini \
  --hf-model Qwen/Qwen2.5-1.5B-Instruct \
  --top-k 20 \
  --max-new-tokens 3 \
  --json-output comparison.json
```

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
