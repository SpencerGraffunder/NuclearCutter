"""Tests for transcription progress reporting (nuclearcutter/detection/transcribe.py).

mlx-whisper has no progress API, so transcribe() proxies its internal tqdm
bar. These tests verify the proxy math and that the monkeypatch is always
restored.
"""

import sys

import mlx_whisper.transcribe  # noqa: F401 — ensure the module is in sys.modules

from nuclearcutter.detection.transcribe import _TqdmProxy, _transcribe_with_progress

# The package's __init__ rebinds `mlx_whisper.transcribe` to a function, so
# grab the real module via sys.modules to test the tqdm monkeypatch.
mlx_transcribe = sys.modules["mlx_whisper.transcribe"]


def test_tqdm_proxy_reports_fractions():
    seen = []
    with _TqdmProxy(total=100, disable=True, _callback=seen.append) as p:
        p.update(25)
        p.update(25)
        p.update(50)
    assert seen == [0.25, 0.5, 1.0]


def test_tqdm_proxy_clamps():
    seen = []
    with _TqdmProxy(total=10, disable=True, _callback=seen.append) as p:
        p.update(50)  # over total -> clamps to 1.0
    assert seen == [1.0]


def test_transcribe_with_progress_restores_tqdm(monkeypatch):
    original = mlx_transcribe.tqdm
    called = []

    def _fake_whisper_transcribe(*args, **kwargs):
        # Exactly like real mlx_whisper.transcribe: the module-level name
        # `tqdm` is looked up at call time and `tqdm.tqdm(...)` resolves
        # through whatever we swapped in (the module stand-in proxy).
        with mlx_transcribe.tqdm.tqdm(total=40, unit="frames", disable=False) as bar:
            bar.update(10)
            bar.update(30)
        return {"segments": []}

    # Production calls `mlx_whisper.transcribe` — the package-level function
    # (rebound by __init__), not the module attribute.
    monkeypatch.setattr("mlx_whisper.transcribe", _fake_whisper_transcribe)
    result = _transcribe_with_progress("audio.wav", "model", called.append)
    assert result == {"segments": []}
    assert called == [0.25, 1.0]  # real progress fractions
    # The real tqdm module attribute is restored after the call.
    assert mlx_transcribe.tqdm is original


def test_transcribe_with_progress_restores_on_error(monkeypatch):
    original = mlx_transcribe.tqdm

    def _boom(*args, **kwargs):
        raise RuntimeError("whisper died")

    monkeypatch.setattr("mlx_whisper.transcribe", _boom)
    try:
        _transcribe_with_progress("audio.wav", "model", lambda f: None)
    except RuntimeError:
        pass
    assert mlx_transcribe.tqdm is original
