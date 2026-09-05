import io
import json
import os
import threading
import unittest
import urllib.error
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from compare_logprobs import (
    EXAMPLE_QUERIES,
    MEDICAL_QUERIES,
    GenerationStep,
    ModelResult,
    ReasoningBudgetExceeded,
    StepStats,
    TokenLogprob,
    _huggingface_placement,
    _render_huggingface_prompt,
    compute_step_stats,
    compute_summary_stats,
    main,
    parse_args,
    print_comparison,
    print_query_summary,
    print_summary_stats,
    query_huggingface,
    query_openrouter,
)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def fake_torch(cuda_devices, cuda_available=True, mps_available=False):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            device_count=lambda: cuda_devices,
            is_available=lambda: cuda_available,
        ),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps_available)
        ),
    )


class HuggingFacePlacementTest(unittest.TestCase):
    def test_automatic_mode_shards_across_multiple_cuda_devices(self):
        options, target_device = _huggingface_placement(fake_torch(2), None)

        self.assertEqual(options, {"device_map": "auto"})
        self.assertIsNone(target_device)

    def test_explicit_device_disables_sharding(self):
        options, target_device = _huggingface_placement(fake_torch(2), "cuda:1")

        self.assertEqual(options, {})
        self.assertEqual(target_device, "cuda:1")

    def test_single_cuda_device_is_selected_automatically(self):
        options, target_device = _huggingface_placement(fake_torch(1), "auto")

        self.assertEqual(options, {})
        self.assertEqual(target_device, "cuda")


class OpenRouterTest(unittest.TestCase):
    def completion_response(self):
        return FakeResponse(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": " Paris"},
                            "logprobs": {
                                "content": [
                                    {
                                        "token": " Paris",
                                        "logprob": -0.1,
                                        "top_logprobs": [
                                            {"token": " Paris", "logprob": -0.1}
                                        ],
                                    }
                                ]
                            },
                        }
                    ]
                }
            ).encode()
        )

    def test_normalizes_chat_completion_logprobs(self):
        response = {
            "choices": [
                {
                    "message": {"content": " Paris"},
                    "logprobs": {
                        "content": [
                            {
                                "token": " Paris",
                                "logprob": -0.1,
                                "top_logprobs": [
                                    {"token": " Paris", "logprob": -0.1},
                                    {"token": " Lyon", "logprob": -2.3},
                                ],
                            }
                        ]
                    },
                }
            ]
        }
        fake_response = FakeResponse(json.dumps(response).encode())

        with patch("urllib.request.urlopen", return_value=fake_response) as urlopen:
            result = query_openrouter(
                "vendor/model", "The capital is", 2, 1, "test-key", "fireworks"
            )

        request = urlopen.call_args.args[0]
        request_body = json.loads(request.data)
        self.assertTrue(request_body["logprobs"])
        self.assertEqual(request_body["top_logprobs"], 2)
        self.assertNotIn("reasoning", request_body)
        self.assertEqual(
            request_body["provider"],
            {"require_parameters": True, "only": ["fireworks"]},
        )
        self.assertEqual(result.generated_text, " Paris")
        self.assertEqual(result.steps[0].generated_token, " Paris")
        self.assertEqual(result.steps[0].top_tokens[1].token, " Lyon")
        self.assertAlmostEqual(
            result.steps[0].top_tokens[0].probability, 0.904837, places=6
        )

    def test_preserves_reasoning_while_normalizing_content_logprobs(self):
        response = {
            "usage": {
                "completion_tokens_details": {"reasoning_tokens": 37}
            },
            "choices": [
                {
                    "message": {"content": " Answer", "reasoning": "Work it out"},
                    "logprobs": {
                        "content": [
                            {
                                "token": " Answer",
                                "logprob": -0.1,
                                "top_logprobs": [
                                    {"token": " Answer", "logprob": -0.1}
                                ],
                            }
                        ]
                    },
                }
            ]
        }

        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(json.dumps(response).encode()),
        ):
            result = query_openrouter("vendor/model", "Question", 1, 10, "test-key")

        self.assertEqual(result.reasoning_text, "Work it out")
        self.assertEqual(result.reasoning_tokens, 37)
        self.assertEqual([step.generated_token for step in result.steps], [" Answer"])

    def test_caps_reasoning_and_reserves_visible_token_budget(self):
        with patch(
            "urllib.request.urlopen",
            return_value=self.completion_response(),
        ) as urlopen:
            query_openrouter(
                "vendor/model",
                "Question",
                1,
                100,
                "test-key",
                max_openrouter_tokens=2000,
                max_reasoning_tokens=1000,
            )

        request_body = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(request_body["max_tokens"], 1100)
        self.assertEqual(request_body["reasoning"], {"max_tokens": 1000})

    def test_uses_low_reasoning_effort_with_fixed_reasoning_allowance(self):
        with patch(
            "urllib.request.urlopen",
            return_value=self.completion_response(),
        ) as urlopen:
            query_openrouter(
                "vendor/model",
                "Question",
                1,
                100,
                "test-key",
                max_openrouter_tokens=16384,
                reasoning_effort="low",
            )

        request_body = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(request_body["max_tokens"], 1100)
        self.assertEqual(request_body["reasoning"], {"effort": "low"})

    def test_none_reasoning_effort_does_not_reserve_reasoning_tokens(self):
        with patch(
            "urllib.request.urlopen",
            return_value=self.completion_response(),
        ) as urlopen:
            query_openrouter(
                "vendor/model",
                "Question",
                1,
                100,
                "test-key",
                max_openrouter_tokens=100,
                reasoning_effort="none",
            )

        request_body = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(request_body["max_tokens"], 100)
        self.assertEqual(request_body["reasoning"], {"effort": "none"})

    def test_does_not_grow_budget_with_reasoning_effort(self):
        reasoning_only = {
            "provider": "Example Provider",
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": None, "reasoning": "Still thinking"},
                    "logprobs": None,
                }
            ],
        }

        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(json.dumps(reasoning_only).encode()),
        ) as urlopen:
            with self.assertRaisesRegex(
                RuntimeError,
                "reasoning effort 'low'.*fixed 1100-token request was not retried",
            ):
                query_openrouter(
                    "vendor/model",
                    "Question",
                    1,
                    100,
                    "test-key",
                    max_openrouter_tokens=16384,
                    reasoning_effort="low",
                )

        urlopen.assert_called_once()

    def test_caps_visible_tokens_used_for_comparison(self):
        content = [
            {
                "token": token,
                "logprob": -0.1,
                "top_logprobs": [{"token": token, "logprob": -0.1}],
            }
            for token in [" one", " two", " three"]
        ]
        response = {
            "choices": [
                {
                    "message": {"content": " one two three"},
                    "logprobs": {"content": content},
                }
            ]
        }

        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(json.dumps(response).encode()),
        ):
            result = query_openrouter("vendor/model", "Question", 1, 2, "test-key")

        self.assertEqual(result.generated_text, " one two")
        self.assertEqual(
            [step.generated_token for step in result.steps],
            [" one", " two"],
        )

    def test_rejects_response_without_ranked_tokens(self):
        response = {
            "choices": [
                {
                    "message": {"content": " Paris"},
                    "logprobs": {
                        "content": [
                            {
                                "token": " Paris",
                                "logprob": -0.1,
                                "top_logprobs": [],
                            }
                        ]
                    },
                }
            ]
        }

        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(json.dumps(response).encode()),
        ):
            with self.assertRaisesRegex(RuntimeError, "no top-token logprobs"):
                query_openrouter("vendor/model", "The capital is", 2, 1, "test-key")

    def test_rejects_response_without_generation_steps(self):
        response = {
            "choices": [
                {
                    "message": {"content": ""},
                    "logprobs": {"content": []},
                }
            ]
        }

        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(json.dumps(response).encode()),
        ):
            with self.assertRaisesRegex(RuntimeError, "no generated-token logprobs"):
                query_openrouter("vendor/model", "The capital is", 2, 1, "test-key")

    def test_explains_reasoning_only_response_without_logprobs(self):
        response = {
            "provider": "Example Provider",
            "choices": [
                {
                    "message": {"content": None, "reasoning": "The"},
                    "logprobs": None,
                }
            ],
        }

        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(json.dumps(response).encode()),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Example Provider.*reasoning but no visible output tokens.*hard cap",
            ):
                query_openrouter("vendor/model", "The capital is", 2, 1, "test-key")

    def test_grows_budget_until_reasoning_produces_visible_tokens(self):
        reasoning_only = {
            "provider": "Example Provider",
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": None, "reasoning": "Still thinking"},
                    "logprobs": None,
                }
            ],
        }

        with patch(
            "urllib.request.urlopen",
            side_effect=[
                FakeResponse(json.dumps(reasoning_only).encode()),
                self.completion_response(),
            ],
        ) as urlopen:
            result = query_openrouter(
                "vendor/model",
                "Question",
                1,
                100,
                "test-key",
                max_openrouter_tokens=1000,
            )

        budgets = [
            json.loads(call.args[0].data)["max_tokens"]
            for call in urlopen.call_args_list
        ]
        self.assertEqual(budgets, [100, 200])
        self.assertEqual(result.generated_text, " Paris")

    def test_does_not_grow_budget_with_explicit_reasoning_cap(self):
        reasoning_only = {
            "provider": "Example Provider",
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": None, "reasoning": "Still thinking"},
                    "logprobs": None,
                }
            ],
        }

        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(json.dumps(reasoning_only).encode()),
        ) as urlopen:
            with self.assertRaisesRegex(
                RuntimeError,
                "did not finish reasoning.*fixed 1100-token request was not retried",
            ):
                query_openrouter(
                    "vendor/model",
                    "Question",
                    1,
                    100,
                    "test-key",
                    max_openrouter_tokens=16384,
                    max_reasoning_tokens=1000,
                )

        urlopen.assert_called_once()
        request_body = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(request_body["max_tokens"], 1100)

    def test_retries_rate_limit_using_retry_after(self):
        rate_limit = urllib.error.HTTPError(
            "https://openrouter.ai/api/v1/chat/completions",
            429,
            "Too Many Requests",
            {"Retry-After": "0"},
            FakeResponse(b'{"error":{"message":"rate limited"}}'),
        )

        with (
            patch(
                "urllib.request.urlopen",
                side_effect=[rate_limit, self.completion_response()],
            ) as urlopen,
            patch("compare_logprobs.time.sleep") as sleep,
            redirect_stdout(io.StringIO()),
        ):
            result = query_openrouter(
                "vendor/model", "The capital is", 1, 1, "test-key"
            )

        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(0.0)
        self.assertEqual(result.generated_text, " Paris")

    def test_does_not_retry_permanent_http_error(self):
        bad_request = urllib.error.HTTPError(
            "https://openrouter.ai/api/v1/chat/completions",
            400,
            "Bad Request",
            {},
            FakeResponse(b'{"error":{"message":"invalid request"}}'),
        )

        with (
            patch("urllib.request.urlopen", side_effect=bad_request) as urlopen,
            patch("compare_logprobs.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
                query_openrouter("vendor/model", "The capital is", 1, 1, "test-key")

        urlopen.assert_called_once()
        sleep.assert_not_called()

    def test_adapts_to_provider_top_logprobs_limit(self):
        invalid_top_k = urllib.error.HTTPError(
            "https://openrouter.ai/api/v1/chat/completions",
            400,
            "Bad Request",
            {},
            FakeResponse(
                b'{"error":{"metadata":{"raw":"Range of top_logprobs '
                b'should be [0, 5]"}}}'
            ),
        )

        with (
            patch(
                "urllib.request.urlopen",
                side_effect=[invalid_top_k, self.completion_response()],
            ) as urlopen,
            patch("compare_logprobs.time.sleep") as sleep,
        ):
            result = query_openrouter(
                "vendor/model", "The capital is", 20, 1, "test-key"
            )

        first_body = json.loads(urlopen.call_args_list[0].args[0].data)
        second_body = json.loads(urlopen.call_args_list[1].args[0].data)
        self.assertEqual(first_body["top_logprobs"], 20)
        self.assertEqual(second_body["top_logprobs"], 5)
        sleep.assert_not_called()
        self.assertEqual(result.generated_text, " Paris")


class MainWorkflowTest(unittest.TestCase):
    def test_huggingface_is_teacher_forced_with_openrouter_tokens(self):
        reference_steps = [
            GenerationStep(" first", [TokenLogprob(" first", -0.1)]),
            GenerationStep(" second", [TokenLogprob(" second", -0.2)]),
        ]
        openrouter_result = ModelResult(
            "openrouter",
            "remote/model",
            " first second",
            reference_steps,
            reasoning_text="Think first",
        )
        huggingface_result = ModelResult(
            "huggingface",
            "local/model",
            " first second",
            reference_steps,
            teacher_forced=True,
        )
        args = Namespace(
            prompt="prompt",
            openrouter_model="remote/model",
            openrouter_provider="fireworks",
            hf_model="local/model",
            top_k=20,
            max_new_tokens=2,
            max_openrouter_tokens=1000,
            max_reasoning_tokens=100,
            reasoning_effort=None,
            device="cpu",
            json_output=None,
        )
        stdout = io.StringIO()

        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
            patch("compare_logprobs.parse_args", return_value=args),
            patch(
                "compare_logprobs.query_openrouter",
                return_value=openrouter_result,
            ) as query_openrouter,
            patch(
                "compare_logprobs.query_huggingface",
                return_value=huggingface_result,
            ) as query_huggingface,
            redirect_stdout(stdout),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(query_openrouter.call_args.args[5], "fireworks")
        self.assertEqual(query_openrouter.call_args.args[6], 1000)
        self.assertEqual(query_openrouter.call_args.args[7], 100)
        self.assertIsNone(query_openrouter.call_args.args[8])
        self.assertEqual(
            query_huggingface.call_args.kwargs["reference_tokens"],
            [" first", " second"],
        )
        self.assertEqual(
            query_huggingface.call_args.kwargs["reasoning_text"], "Think first"
        )
        rendered = stdout.getvalue()
        self.assertLess(
            rendered.index("=== Generation step 1 ==="),
            rendered.index("--- Query summary ---"),
        )
        self.assertLess(
            rendered.index("--- Query summary ---"),
            rendered.index("AVERAGE STATS SUMMARY"),
        )

    def test_runs_openrouter_requests_concurrently_and_scores_in_prompt_order(self):
        prompts = ("first prompt", "second prompt")
        barrier = threading.Barrier(len(prompts))
        completed_prompts = []

        def concurrent_openrouter(model, prompt, *args):
            barrier.wait(timeout=1)
            completed_prompts.append(prompt)
            step = GenerationStep(f" {prompt}", [TokenLogprob(f" {prompt}", -0.1)])
            return ModelResult("openrouter", model, f" {prompt}", [step])

        scored_prompts = []

        def serial_huggingface(model, prompt, *args, **kwargs):
            scored_prompts.append(prompt)
            step = GenerationStep(f" {prompt}", [TokenLogprob(f" {prompt}", -0.1)])
            return ModelResult("huggingface", model, f" {prompt}", [step], True)

        args = Namespace(
            prompt=None,
            medical_queries=False,
            openrouter_model="remote/model",
            openrouter_provider=None,
            openrouter_concurrency=2,
            hf_model="local/model",
            top_k=1,
            max_new_tokens=1,
            max_openrouter_tokens=4100,
            max_reasoning_tokens=4000,
            reasoning_effort=None,
            device="cpu",
            json_output=None,
        )

        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
            patch("compare_logprobs.EXAMPLE_QUERIES", prompts),
            patch("compare_logprobs.parse_args", return_value=args),
            patch(
                "compare_logprobs.query_openrouter",
                side_effect=concurrent_openrouter,
            ),
            patch(
                "compare_logprobs.query_huggingface",
                side_effect=serial_huggingface,
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertCountEqual(completed_prompts, prompts)
        self.assertEqual(scored_prompts, list(prompts))


class HuggingFacePromptTest(unittest.TestCase):
    def test_teacher_forcing_uses_one_forward_pass_for_compatible_prefixes(self):
        import torch

        class FakeTokenizer:
            chat_template = None
            eos_token_id = 99

            def __call__(self, text, **kwargs):
                token_ids = [(ord(character) % 7) + 1 for character in text]
                return {
                    "input_ids": torch.tensor([token_ids]),
                    "attention_mask": torch.ones(1, len(token_ids), dtype=torch.long),
                }

            def decode(self, token_ids):
                return f" token-{token_ids[0]}"

        class FakeModel:
            def __init__(self):
                self.call_count = 0

            def get_input_embeddings(self):
                return SimpleNamespace(weight=torch.zeros(1))

            def __call__(self, input_ids, attention_mask):
                self.call_count += 1
                logits = torch.zeros(1, input_ids.shape[1], 10)
                logits[:, :, 3] = 1
                return SimpleNamespace(logits=logits)

        model = FakeModel()
        result = query_huggingface(
            "local/model",
            "Question",
            top_k=2,
            max_new_tokens=3,
            reference_tokens=[" one", " two", " three"],
            tokenizer=FakeTokenizer(),
            model=model,
            reasoning_text="Long reasoning trace",
        )

        self.assertEqual(model.call_count, 1)
        self.assertEqual(len(result.steps), 3)
        self.assertEqual(result.generated_text, " one two three")

    def test_uses_structured_reasoning_content_when_template_supports_it(self):
        tokenizer = SimpleNamespace(
            apply_chat_template=lambda messages, **kwargs: (
                f"USER:{messages[0]['content']}\n"
                f"THINK:{messages[1]['reasoning_content']}\n"
                f"ANSWER:{messages[1]['content']}"
            )
        )

        rendered = _render_huggingface_prompt(
            tokenizer, "Question", "Reasoning trace", " visible"
        )

        self.assertEqual(
            rendered,
            "USER:Question\nTHINK:Reasoning trace\nANSWER: visible",
        )

    def test_falls_back_to_think_tags_when_template_ignores_reasoning(self):
        tokenizer = SimpleNamespace(
            apply_chat_template=lambda messages, **kwargs: (
                f"USER:{messages[0]['content']}\nANSWER:{messages[-1]['content']}"
            )
        )

        rendered = _render_huggingface_prompt(
            tokenizer, "Question", "Reasoning trace", " visible"
        )

        self.assertEqual(
            rendered,
            "USER:Question\nANSWER:<think>\nReasoning trace\n</think>\n\n visible",
        )


class ComparisonOutputTest(unittest.TestCase):
    def test_prints_only_first_and_last_three_steps(self):
        steps = [
            GenerationStep(
                f" token-{index}",
                [TokenLogprob(f" token-{index}", -0.1)],
            )
            for index in range(1, 31)
        ]
        result = ModelResult(
            "openrouter",
            "model",
            "".join(step.generated_token for step in steps),
            steps,
        )
        output = io.StringIO()

        with redirect_stdout(output):
            print_comparison(result, result)

        rendered = output.getvalue()
        self.assertIn("=== Generation step 3 ===", rendered)
        self.assertNotIn("=== Generation step 4 ===", rendered)
        self.assertNotIn("=== Generation step 27 ===", rendered)
        self.assertIn("=== Generation step 28 ===", rendered)
        self.assertIn("=== Generation step 30 ===", rendered)
        self.assertIn("24 generation steps omitted", rendered)
        self.assertNotIn("token-15", rendered)

    def test_loops_over_example_queries_when_prompt_is_none(self):
        reference_steps = [
            GenerationStep(" first", [TokenLogprob(" first", -0.1)]),
        ]
        openrouter_result = ModelResult(
            "openrouter", "remote/model", " first", reference_steps
        )
        huggingface_result = ModelResult(
            "huggingface",
            "local/model",
            " first",
            reference_steps,
            teacher_forced=True,
        )
        args = Namespace(
            prompt=None,
            openrouter_model="remote/model",
            openrouter_provider=None,
            hf_model="local/model",
            top_k=20,
            max_new_tokens=1,
            max_openrouter_tokens=1000,
            max_reasoning_tokens=None,
            reasoning_effort="low",
            device="cpu",
            json_output=None,
        )

        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
            patch("compare_logprobs.parse_args", return_value=args),
            patch(
                "compare_logprobs.query_openrouter",
                return_value=openrouter_result,
            ) as mock_openrouter,
            patch(
                "compare_logprobs.query_huggingface",
                return_value=huggingface_result,
            ) as mock_hf,
            redirect_stdout(io.StringIO()),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(mock_openrouter.call_count, len(EXAMPLE_QUERIES))
        self.assertEqual(mock_hf.call_count, len(EXAMPLE_QUERIES))
        # Ensure all example queries were passed as prompts in order
        called_prompts = [call.args[1] for call in mock_openrouter.call_args_list]
        self.assertEqual(called_prompts, list(EXAMPLE_QUERIES))

    def test_loops_over_medical_queries_when_selected(self):
        reference_steps = [
            GenerationStep(" first", [TokenLogprob(" first", -0.1)]),
        ]
        openrouter_result = ModelResult(
            "openrouter", "remote/model", " first", reference_steps
        )
        huggingface_result = ModelResult(
            "huggingface",
            "local/model",
            " first",
            reference_steps,
            teacher_forced=True,
        )
        args = Namespace(
            prompt=None,
            medical_queries=True,
            openrouter_model="remote/model",
            openrouter_provider=None,
            hf_model="local/model",
            top_k=20,
            max_new_tokens=1,
            max_openrouter_tokens=1000,
            max_reasoning_tokens=None,
            reasoning_effort="low",
            device="cpu",
            json_output=None,
        )

        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
            patch("compare_logprobs.parse_args", return_value=args),
            patch(
                "compare_logprobs.query_openrouter",
                return_value=openrouter_result,
            ) as mock_openrouter,
            patch(
                "compare_logprobs.query_huggingface",
                return_value=huggingface_result,
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(MEDICAL_QUERIES), 20)
        called_prompts = [call.args[1] for call in mock_openrouter.call_args_list]
        self.assertEqual(called_prompts, list(MEDICAL_QUERIES))

    def test_skips_failed_query_and_excludes_it_from_summary(self):
        reference_steps = [
            GenerationStep(" first", [TokenLogprob(" first", -0.1)]),
        ]
        openrouter_result = ModelResult(
            "openrouter", "remote/model", " first", reference_steps
        )
        huggingface_result = ModelResult(
            "huggingface",
            "local/model",
            " first",
            reference_steps,
            teacher_forced=True,
        )
        args = Namespace(
            prompt=None,
            openrouter_model="remote/model",
            openrouter_provider="digitalocean",
            hf_model="local/model",
            top_k=1,
            max_new_tokens=1,
            max_openrouter_tokens=4100,
            max_reasoning_tokens=4000,
            reasoning_effort=None,
            device="cpu",
            json_output=None,
        )

        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
            patch("compare_logprobs.parse_args", return_value=args),
            patch(
                "compare_logprobs.query_openrouter",
                side_effect=[ReasoningBudgetExceeded("reasoning budget exhausted")]
                + [openrouter_result] * (len(EXAMPLE_QUERIES) - 1),
            ) as mock_openrouter,
            patch(
                "compare_logprobs.query_huggingface",
                return_value=huggingface_result,
            ) as mock_hf,
            patch("compare_logprobs.print_summary_stats") as print_summary,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()) as stderr,
        ):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(mock_openrouter.call_count, len(EXAMPLE_QUERIES))
        self.assertEqual(mock_hf.call_count, len(EXAMPLE_QUERIES) - 1)
        self.assertEqual(print_summary.call_args.args[0].total_queries, 19)
        self.assertEqual(
            print_summary.call_args.args[0].skipped_reasoning_queries,
            1,
        )
        self.assertIn("Skipping query 1: reasoning budget exhausted", stderr.getvalue())


class StatsComputationTest(unittest.TestCase):
    def test_compute_step_stats_match_and_overlap(self):
        left_step = GenerationStep(
            " Paris",
            [
                TokenLogprob(" Paris", -0.1),
                TokenLogprob(" Lyon", -2.0),
                TokenLogprob(" Nice", -3.0),
            ],
        )
        right_step = GenerationStep(
            " Paris",
            [
                TokenLogprob(" Paris", -0.2),
                TokenLogprob(" Marseille", -2.1),
                TokenLogprob(" Lyon", -2.5),
            ],
        )
        step_stat = compute_step_stats(left_step, right_step, top_k=3)
        self.assertTrue(step_stat.top1_match)
        self.assertEqual(step_stat.overlap_count, 2)
        self.assertAlmostEqual(step_stat.overlap_ratio, 2 / 3)
        self.assertEqual(step_stat.hf_reference_rank, 1)
        self.assertAlmostEqual(
            step_stat.prob_diff,
            abs(left_step.top_tokens[0].probability - right_step.top_tokens[0].probability),
        )

    def test_overlap_ratio_uses_available_ranked_tokens(self):
        step = GenerationStep(
            " Paris",
            [TokenLogprob(" Paris", -0.1), TokenLogprob(" Lyon", -2.0)],
        )

        step_stat = compute_step_stats(step, step, top_k=20)

        self.assertEqual(step_stat.overlap_count, 2)
        self.assertEqual(step_stat.overlap_ratio, 1.0)

    def test_top1_match_compares_ranked_tokens_not_emitted_token(self):
        left_step = GenerationStep(
            " sampled",
            [TokenLogprob(" ranked", -0.1), TokenLogprob(" sampled", -0.2)],
        )
        right_step = GenerationStep(
            " ranked",
            [TokenLogprob(" ranked", -0.1), TokenLogprob(" other", -1.0)],
        )

        step_stat = compute_step_stats(left_step, right_step, top_k=2)

        self.assertTrue(step_stat.top1_match)
        self.assertEqual(step_stat.hf_reference_rank, None)

    def test_compute_summary_stats(self):
        s1 = StepStats(
            top1_match=True,
            overlap_count=3,
            overlap_ratio=0.75,
            openrouter_top1_prob=0.8,
            hf_top1_prob=0.7,
            openrouter_top1_logprob=-0.223,
            hf_top1_logprob=-0.356,
            prob_diff=0.1,
        )
        s2 = StepStats(
            top1_match=False,
            overlap_count=1,
            overlap_ratio=0.25,
            openrouter_top1_prob=0.6,
            hf_top1_prob=0.5,
            openrouter_top1_logprob=-0.510,
            hf_top1_logprob=-0.693,
            prob_diff=0.1,
        )
        summary = compute_summary_stats(
            [s1, s2],
            total_queries=2,
            reasoning_token_counts=[100, 300],
            skipped_reasoning_queries=3,
        )
        self.assertEqual(summary.total_queries, 2)
        self.assertEqual(summary.skipped_reasoning_queries, 3)
        self.assertEqual(summary.total_steps, 2)
        self.assertEqual(summary.avg_reasoning_tokens, 200)
        self.assertAlmostEqual(summary.top1_match_rate, 0.5)
        self.assertAlmostEqual(summary.avg_overlap_count, 2.0)
        self.assertAlmostEqual(summary.avg_overlap_ratio, 0.5)
        self.assertAlmostEqual(summary.avg_openrouter_top1_prob, 0.7)
        self.assertAlmostEqual(summary.avg_hf_top1_prob, 0.6)
        self.assertAlmostEqual(summary.avg_prob_diff, 0.1)

        output = io.StringIO()
        with redirect_stdout(output):
            print_summary_stats(summary, top_k=5)
        self.assertIn("Top-k used:                      5", output.getvalue())
        self.assertIn("Average reasoning tokens:        200.00", output.getvalue())
        self.assertIn("Queries skipped (long reasoning):       3", output.getvalue())

        query_output = io.StringIO()
        with redirect_stdout(query_output):
            print_query_summary(summary, top_k=5)
        self.assertIn("--- Query summary ---", query_output.getvalue())
        self.assertIn("Reasoning tokens:                200", query_output.getvalue())
        self.assertIn("Visible generation steps:        2", query_output.getvalue())
        self.assertIn("Top-1 matches:                   1/2 (50.00%)", query_output.getvalue())


class ArgParseTest(unittest.TestCase):
    def test_prompt_optional(self):
        with patch(
            "sys.argv",
            [
                "compare_logprobs.py",
                "--openrouter-model",
                "remote/model",
                "--openrouter-provider",
                "fireworks",
                "--hf-model",
                "local/model",
            ],
        ):
            args = parse_args()
            self.assertIsNone(args.prompt)
            self.assertEqual(args.openrouter_model, "remote/model")
            self.assertEqual(args.openrouter_provider, "fireworks")
            self.assertEqual(args.max_openrouter_tokens, 4100)
            self.assertEqual(args.max_reasoning_tokens, 4000)
            self.assertIsNone(args.reasoning_effort)
            self.assertEqual(
                args.json_output,
                Path("local-model-remote-model.json"),
            )

    def test_explicit_json_output_overrides_default(self):
        with patch(
            "sys.argv",
            [
                "compare_logprobs.py",
                "--openrouter-model",
                "remote/model",
                "--hf-model",
                "local/model",
                "--json-output",
                "custom.json",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.json_output, Path("custom.json"))

    def test_medical_queries_flag(self):
        with patch(
            "sys.argv",
            [
                "compare_logprobs.py",
                "--medical-queries",
                "--openrouter-model",
                "remote/model",
                "--hf-model",
                "local/model",
            ],
        ):
            args = parse_args()

        self.assertTrue(args.medical_queries)
        self.assertIsNone(args.prompt)

    def test_medical_queries_rejects_positional_prompt(self):
        with patch(
            "sys.argv",
            [
                "compare_logprobs.py",
                "custom prompt",
                "--medical-queries",
                "--openrouter-model",
                "remote/model",
                "--hf-model",
                "local/model",
            ],
        ):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_numeric_reasoning_cap_disables_default_effort(self):
        with patch(
            "sys.argv",
            [
                "compare_logprobs.py",
                "--openrouter-model",
                "remote/model",
                "--hf-model",
                "local/model",
                "--max-reasoning-tokens",
                "1000",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.max_reasoning_tokens, 1000)
        self.assertIsNone(args.reasoning_effort)

    def test_reasoning_effort_disables_default_numeric_cap(self):
        with patch(
            "sys.argv",
            [
                "compare_logprobs.py",
                "--openrouter-model",
                "remote/model",
                "--hf-model",
                "local/model",
                "--reasoning-effort",
                "low",
            ],
        ):
            args = parse_args()

        self.assertIsNone(args.max_reasoning_tokens)
        self.assertEqual(args.reasoning_effort, "low")

    def test_reasoning_cap_must_fit_with_visible_budget(self):
        with patch(
            "sys.argv",
            [
                "compare_logprobs.py",
                "--openrouter-model",
                "remote/model",
                "--hf-model",
                "local/model",
                "--max-new-tokens",
                "100",
                "--max-reasoning-tokens",
                "1000",
                "--max-openrouter-tokens",
                "1050",
            ],
        ):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_prompt_provided(self):
        with patch(
            "sys.argv",
            [
                "compare_logprobs.py",
                "custom prompt",
                "--openrouter-model",
                "remote/model",
                "--hf-model",
                "local/model",
            ],
        ):
            args = parse_args()
            self.assertEqual(args.prompt, "custom prompt")


if __name__ == "__main__":
    unittest.main()