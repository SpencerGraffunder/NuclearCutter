"""Tests for the GUI benchmark (nuclearcutter/bench.py).

The benchmark must exercise the EXACT same request path as a real scan —
including thinking disabled — so its results are representative of a scan.
"""

from pathlib import Path
from unittest.mock import MagicMock

from nuclearcutter.bench import _query_with_usage


def test_attach_usage_hook_chains_to_existing_callback():
    """Benchmark requests must also feed the server's model-stats tracker."""
    from unittest.mock import MagicMock

    from nuclearcutter.bench import attach_usage_hook

    client = MagicMock()
    fired = []
    client.usage_callback = lambda data, elapsed: fired.append((data, elapsed))
    attach_usage_hook(client)
    client.usage_callback(
        {"usage": {"prompt_tokens": 1406, "completion_tokens": 30},
         "timings": {"prompt_n": 1406, "prompt_ms": 1000, "predicted_n": 30, "predicted_ms": 500}},
        1.5,
    )
    assert fired, "the existing (model-stats) callback must still fire"
    assert client._last_usage["prompt_tokens"] == 1406
    assert client._last_timings["pp_tok_s"] is not None


def test_query_with_usage_uses_scan_client_method():
    """The benchmark calls LLMClient.vision_query_json — the same method the
    scanner's sweep and confirm pass use — so thinking-off flags and prompt
    plumbing are identical to a real scan."""
    client = MagicMock()
    client.vision_query_json.return_value = {"contains_flagged_content": False}
    client._last_usage = {"prompt_tokens": 1406, "completion_tokens": 30}
    client._last_timings = {}

    out = _query_with_usage(client, "the prompt", [Path("/tmp/f.png")])

    client.vision_query_json.assert_called_once_with("the prompt", [Path("/tmp/f.png")])
    assert out["parsed"] == {"contains_flagged_content": False}
    assert out["usage"]["completion_tokens"] == 30
