import io
import json
import os
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from compare_logprobs import (
    GenerationStep,
    ModelResult,
    TokenLogprob,
    _huggingface_placement,
    main,
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


if __name__ == "__main__":
    unittest.main()