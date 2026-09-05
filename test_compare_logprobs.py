import io
import json
import os
import unittest
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
    compute_step_stats,
    compute_summary_stats,
    main,
    parse_args,
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
                "vendor/model", "The capital is", 2, 1, "test-key"
            )

        request = urlopen.call_args.args[0]
        request_body = json.loads(request.data)
        self.assertTrue(request_body["logprobs"])
        self.assertEqual(request_body["top_logprobs"], 2)
        self.assertEqual(result.generated_text, " Paris")
        self.assertEqual(result.steps[0].generated_token, " Paris")
        self.assertEqual(result.steps[0].top_tokens[1].token, " Lyon")
        self.assertAlmostEqual(
            result.steps[0].top_tokens[0].probability, 0.904837, places=6
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


class MainWorkflowTest(unittest.TestCase):
    def test_huggingface_is_teacher_forced_with_openrouter_tokens(self):
        reference_steps = [
            GenerationStep(" first", [TokenLogprob(" first", -0.1)]),
            GenerationStep(" second", [TokenLogprob(" second", -0.2)]),
        ]
        openrouter_result = ModelResult(
            "openrouter", "remote/model", " first second", reference_steps
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
            hf_model="local/model",
            top_k=20,
            max_new_tokens=2,
            device="cpu",
            json_output=None,
        )

        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
            patch("compare_logprobs.parse_args", return_value=args),
            patch(
                "compare_logprobs.query_openrouter",
                return_value=openrouter_result,
            ),
            patch(
                "compare_logprobs.query_huggingface",
                return_value=huggingface_result,
            ) as query_huggingface,
            redirect_stdout(io.StringIO()),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            query_huggingface.call_args.kwargs["reference_tokens"],
            [" first", " second"],
        )

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
            hf_model="local/model",
            top_k=20,
            max_new_tokens=1,
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
        summary = compute_summary_stats([s1, s2], total_queries=2)
        self.assertEqual(summary.total_queries, 2)
        self.assertEqual(summary.total_steps, 2)
        self.assertAlmostEqual(summary.top1_match_rate, 0.5)
        self.assertAlmostEqual(summary.avg_overlap_count, 2.0)
        self.assertAlmostEqual(summary.avg_overlap_ratio, 0.5)
        self.assertAlmostEqual(summary.avg_openrouter_top1_prob, 0.7)
        self.assertAlmostEqual(summary.avg_hf_top1_prob, 0.6)
        self.assertAlmostEqual(summary.avg_prob_diff, 0.1)


class ArgParseTest(unittest.TestCase):
    def test_prompt_optional(self):
        with patch(
            "sys.argv",
            [
                "compare_logprobs.py",
                "--openrouter-model",
                "remote/model",
                "--hf-model",
                "local/model",
            ],
        ):
            args = parse_args()
            self.assertIsNone(args.prompt)
            self.assertEqual(args.openrouter_model, "remote/model")

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