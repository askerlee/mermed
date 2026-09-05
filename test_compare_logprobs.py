import io
import json
import os
import unittest
import urllib.error
from argparse import Namespace
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from compare_logprobs import (
    EXAMPLE_QUERIES,
    GenerationStep,
    ModelResult,
    StepStats,
    TokenLogprob,
    _huggingface_placement,
    _render_huggingface_prompt,
    compute_step_stats,
    compute_summary_stats,
    main,
    parse_args,
    print_comparison,
    print_summary_stats,
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
            device="cpu",
            json_output=None,
        )

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
            redirect_stdout(io.StringIO()),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(query_openrouter.call_args.args[5], "fireworks")
        self.assertEqual(query_openrouter.call_args.args[6], 1000)
        self.assertEqual(query_openrouter.call_args.args[7], 100)
        self.assertEqual(
            query_huggingface.call_args.kwargs["reference_tokens"],
            [" first", " second"],
        )
        self.assertEqual(
            query_huggingface.call_args.kwargs["reasoning_text"], "Think first"
        )


class HuggingFacePromptTest(unittest.TestCase):
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
    def test_prints_only_first_and_last_ten_steps(self):
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
        self.assertIn("=== Generation step 10 ===", rendered)
        self.assertNotIn("=== Generation step 11 ===", rendered)
        self.assertNotIn("=== Generation step 20 ===", rendered)
        self.assertIn("=== Generation step 21 ===", rendered)
        self.assertIn("=== Generation step 30 ===", rendered)
        self.assertIn("10 generation steps omitted", rendered)
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
        )
        self.assertEqual(summary.total_queries, 2)
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
            self.assertEqual(args.max_openrouter_tokens, 16384)
            self.assertEqual(args.max_reasoning_tokens, 1000)

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