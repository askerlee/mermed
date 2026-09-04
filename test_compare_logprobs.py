import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from compare_logprobs import _huggingface_placement, query_openrouter


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


if __name__ == "__main__":
    unittest.main()