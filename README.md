# Top-token logprob comparison

`compare_logprobs.py` compares the top next-token log probabilities from an
OpenRouter chat model and a local Hugging Face causal language model.

## Setup

Use Python 3.10 or newer, then install the local-model dependencies:

```bash
python -m pip install -r requirements.txt
export OPENROUTER_API_KEY="..."
```

The selected OpenRouter model and provider must support token logprobs. Models
on the Hugging Face Hub are downloaded on first use; `--hf-model` can instead
point to a local model directory.

## Usage

```bash
python compare_logprobs.py \
  "The capital of France is" \
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

Both models start with the same user prompt, but generate independently after
that. Their contexts can therefore diverge after the first generated token.
The local model's chat template is applied automatically when it has one.

Token IDs are not comparable across different tokenizers. The displayed
overlap count compares exact decoded token strings only; the ranked logprobs
and probabilities are the primary output.
