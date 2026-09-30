"""Tests for the web GUI server (nuclearcutter/server.py) — settings
validation, state snapshot, model listing, prompt endpoints, and job guards.
No real model server or movie files are needed."""

import json

from fastapi.testclient import TestClient

from nuclearcutter.server import AppState, JobInfo, _LogCapture, create_app


def make_client(state=None):
    st = state if state is not None else AppState()
    return st, TestClient(create_app(st))


def test_index_serves_ui():
    _, c = make_client()
    r = c.get("/")
    assert r.status_code == 200
    assert "NUCLEARCUTTER" in r.text
    assert "matrix" in r.text.lower() or "--green" in r.text


def test_state_snapshot_shape():
    _, c = make_client()
    r = c.get("/api/state")
    assert r.status_code == 200
    d = r.json()
    assert set(d) == {"settings", "jobs", "timeline", "system", "model", "server", "all_logs"}
    assert set(d["jobs"]) == {"scan", "render", "benchmark"}
    assert d["jobs"]["scan"]["status"] == "idle"
    assert d["timeline"] is None  # no scan loaded yet


def test_settings_apply_and_validate():
    _, c = make_client()
    ok = c.post("/api/settings", json={"settings": {
        "backend": "standalone",
        "scale": "720p",
        "sweep_interval": 5,
        "levels": {"nudity": "high"},
        "audio_actions": {"foul_language": "mute_word"},
    }})
    assert ok.status_code == 200
    s = ok.json()["settings"]
    assert s["scale"] == "720p"
    assert s["sweep_interval"] == 5
    assert s["levels"]["nudity"] == "high"
    assert s["audio_actions"]["foul_language"] == "mute_word"

    bad = c.post("/api/settings", json={"settings": {"backend": "bogus"}})
    assert bad.status_code == 400
    bad2 = c.post("/api/settings", json={"settings": {"scale": "4800p"}})
    assert bad2.status_code == 400
    bad3 = c.post("/api/settings", json={"settings": {"audio_actions": {"gore": "mute_word"}}})
    assert bad3.status_code == 400  # visual categories can't mute per-word


def test_summary_settings_apply_and_validate():
    """The separate summary-model settings round-trip and validate."""
    _, c = make_client()
    ok = c.post("/api/settings", json={"settings": {
        "summary_model": "big-summary",
        "summary_frames": 12,
        "summary_max_context": 24000,
    }})
    assert ok.status_code == 200
    s = ok.json()["settings"]
    assert s["summary_model"] == "big-summary"
    assert s["summary_frames"] == 12
    assert s["summary_max_context"] == 24000

    # Empty summary model disables the render-time summary pass.
    ok2 = c.post("/api/settings", json={"settings": {"summary_model": ""}})
    assert ok2.status_code == 200
    assert ok2.json()["settings"]["summary_model"] == ""

    # Out-of-range frames are rejected (0..24).
    assert c.post("/api/settings", json={"settings": {"summary_frames": 25}}).status_code == 400
    assert c.post("/api/settings", json={"settings": {"summary_frames": -1}}).status_code == 400
    assert c.post("/api/settings", json={"settings": {"summary_max_context": 0}}).status_code == 400

    # A rejected value must NOT leak into the stored settings (validate-then-assign).
    st = c.get("/api/state").json()["settings"]
    assert st["summary_frames"] == 12
    assert st["summary_max_context"] == 24000


def test_scan_start_requires_video():
    _, c = make_client()
    r = c.post("/api/scan/start")
    assert r.status_code == 400
    assert "video file not found" in r.json()["detail"]


def test_render_start_requires_video_and_scan(tmp_path):
    _, c = make_client()
    # No video set.
    assert c.post("/api/render/start").status_code == 400
    # Video set but no scan file exists.
    movie = tmp_path / "M.mkv"
    movie.write_bytes(b"x")
    c.post("/api/settings", json={"settings": {"video_path": str(movie)}})
    r = c.post("/api/render/start")
    assert r.status_code == 400
    assert "no scan file found" in r.json()["detail"]


def test_scan_clear_without_video_is_safe():
    st, c = make_client()
    r = c.post("/api/scan/clear")
    assert r.status_code == 200
    assert r.json()["removed"] == []
    # State was reset to fresh jobs.
    assert st.scan_result is None


def _write_full_scan(tmp_path):
    """Write a movie with a transcript cache + status + completed result, and
    return the movie path. Mirrors the auto-load test setup."""
    from nuclearcutter.schema import (
        Category, FilmIdentity, ScanResult, SeverityLevel, VisualDetection,
    )

    movie = tmp_path / "M.mkv"
    movie.write_bytes(b"x")
    # Transcript cache exists -> transcription counts as done.
    (tmp_path / "M.nuclearcutter.transcript.json").write_text('{"utterances": []}')
    (tmp_path / "M.nuclearcutter.status.json").write_text('{"phase": "done"}')
    scan = ScanResult(
        schema_version=1,
        identity=FilmIdentity(title="M", year=None, duration_seconds=100.0, phash_samples=[]),
        visual_detections=[
            VisualDetection(category=Category.NUDITY, start=1.0, end=2.0,
                            description="x", confidence=0.9, level=SeverityLevel.HIGH),
        ],
        language_detections=[],
    )
    scan.save(tmp_path / "M.nuclearcutter.json")
    return movie


def test_clear_scan_verify_keeps_transcript(tmp_path):
    """Clearing scan+verify deletes the sweep/confirm files and resets the scan
    job, but intentionally KEEPS the transcript cache (separate section)."""
    movie = _write_full_scan(tmp_path)
    st = AppState()
    st.update_settings({"video_path": str(movie)})
    assert st.scan_result is not None

    d = st.clear_scan_progress()
    removed = d["removed"]
    assert any(p.endswith(".nuclearcutter.status.json") for p in removed)
    assert any(p.endswith(".nuclearcutter.json") for p in removed)
    # Transcript is a separate section — untouched.
    assert not any(".transcript" in p for p in removed)
    assert (tmp_path / "M.nuclearcutter.transcript.json").exists()
    # Scan state fully reset, but the transcribe bar still reflects the
    # preserved transcript (it's a separate section that wasn't cleared).
    assert st.scan_result is None
    assert st.scan_result_path == ""
    assert st.scan.status == "idle"
    assert st.scan.steps == {"transcribe": 1.0, "scan": 0.0, "verify": 0.0, "summary": 0.0, "render": 0.0}


def test_clear_transcription_keeps_scan(tmp_path):
    """Clearing transcription deletes only the transcript cache and drops the
    transcribe bar, leaving the loaded scan/verify results intact."""
    movie = _write_full_scan(tmp_path)
    st = AppState()
    st.update_settings({"video_path": str(movie)})
    assert st.scan.steps == {"transcribe": 1.0, "scan": 1.0, "verify": 1.0, "summary": 1.0, "render": 0.0}

    d = st.clear_transcription_progress()
    assert any(p.endswith(".nuclearcutter.transcript.json") for p in d["removed"])
    assert len(d["removed"]) == 1
    # Scan/verify files survive.
    assert (tmp_path / "M.nuclearcutter.status.json").exists()
    assert (tmp_path / "M.nuclearcutter.json").exists()
    assert st.scan_result is not None
    # Only the transcribe bar drops back to 0.
    assert st.scan.steps["transcribe"] == 0.0
    assert st.scan.steps["scan"] == 1.0
    assert st.scan.steps["verify"] == 1.0


def test_loaded_scan_reflects_missing_transcript(tmp_path):
    """If a loaded result exists but the transcript cache was cleared, the
    transcribe bar shows 0% while scan/verify stay at 100%."""
    movie = _write_full_scan(tmp_path)
    (tmp_path / "M.nuclearcutter.transcript.json").unlink()

    st = AppState()
    st.update_settings({"video_path": str(movie)})
    assert st.scan_result is not None
    assert st.scan.steps == {"transcribe": 0.0, "scan": 1.0, "verify": 1.0, "summary": 1.0, "render": 0.0}
    assert "transcription pending" in st.scan.message


def test_restart_after_scan_clear_reflects_kept_transcript(tmp_path):
    """After a scan+verify clear, a server restart (which reloads state) still
    shows the preserved transcript at 100% — transcription is a separate
    section and survives."""
    movie = _write_full_scan(tmp_path)
    # Simulate the scan+verify clear: result + status gone, transcript kept.
    (tmp_path / "M.nuclearcutter.json").unlink()
    (tmp_path / "M.nuclearcutter.status.json").unlink()
    assert (tmp_path / "M.nuclearcutter.transcript.json").exists()

    st = AppState()
    st.update_settings({"video_path": str(movie)})  # reload -> _refresh_loaded_scan
    assert st.scan_result is None
    assert st.scan.steps["transcribe"] == 1.0
    assert st.scan.steps["scan"] == 0.0
    assert st.scan.steps["verify"] == 0.0
    assert "Transcript cached" in st.scan.message


def test_scan_clear_section_dispatch(tmp_path):
    """/api/scan/clear routes on the section body; unknown sections 400."""
    movie = _write_full_scan(tmp_path)
    st, c = make_client()
    c.post("/api/settings", json={"settings": {"video_path": str(movie)}})
    assert st.scan_result is not None

    # scan_verify (default) keeps the transcript.
    r = c.post("/api/scan/clear", json={"section": "scan_verify"})
    assert r.status_code == 200
    assert (tmp_path / "M.nuclearcutter.transcript.json").exists()
    assert not (tmp_path / "M.nuclearcutter.json").exists()

    # Rebuild, then clear transcribe only.
    movie2 = _write_full_scan(tmp_path)
    st2, c2 = make_client()
    c2.post("/api/settings", json={"settings": {"video_path": str(movie2)}})
    r2 = c2.post("/api/scan/clear", json={"section": "transcribe"})
    assert r2.status_code == 200
    assert not (tmp_path / "M.nuclearcutter.transcript.json").exists()
    assert (tmp_path / "M.nuclearcutter.json").exists()
    assert st2.scan.steps["transcribe"] == 0.0

    bad = c2.post("/api/scan/clear", json={"section": "bogus"})
    assert bad.status_code == 400


def test_models_endpoint_lists_remote_models(monkeypatch):
    _, c = make_client()

    def fake_list_models(self):
        return ["qwen3-vl-8b", "qwen3-8b", "llama-3.1"]

    monkeypatch.setattr("nuclearcutter.server.LLMClient.list_models", fake_list_models)
    r = c.get("/api/models", params={"base_url": "http://192.168.1.5:1234/v1"})
    assert r.status_code == 200
    d = r.json()
    assert d["reachable"] is True
    assert d["models"] == ["qwen3-vl-8b", "qwen3-8b", "llama-3.1"]
    assert d["base_url"] == "http://192.168.1.5:1234/v1"


def test_definitions_endpoint():
    """The render hover tooltips come from here: all 4 levels for every category."""
    _, c = make_client()
    r = c.get("/api/definitions")
    assert r.status_code == 200
    d = r.json()
    assert set(d) == {"nudity", "gore", "violence", "foul_language"}
    for cat in ("nudity", "gore", "violence"):
        assert set(d[cat]) == {"low", "med", "high", "exhigh"}
        assert all(len(d[cat][lv]) > 20 for lv in d[cat])  # real definitions, not empty
    assert len(d["foul_language"]) > 20  # foul language ships as one full scale text


def test_benchmark_stop_endpoint():
    _, c = make_client()
    r = c.post("/api/benchmark/stop")
    assert r.status_code == 200
    assert r.json()["job"]["status"] in ("idle", "stopping")  # safe when nothing running


def test_job_log_capture():
    """Job threads capture their stdout into job.logs for the GUI terminal."""
    st = AppState()
    job = JobInfo(kind="scan")

    def _worker():
        print("scanning started")
        print("warning: something", file=__import__("sys").stderr)

    st._launch(job, _worker)
    job.thread.join(timeout=5)
    assert not job.thread.is_alive()
    assert "scanning started" in job.logs
    assert "warning: something" in job.logs  # stderr captured too
    assert job.to_dict()["logs"] == job.logs
    # The same lines also land in the merged terminal stream, tagged with the job.
    assert st.all_logs
    assert all(e["job"] == "scan" for e in st.all_logs)
    assert any("scanning started" in e["line"] for e in st.all_logs)
    assert all("t" in e for e in st.all_logs)


def test_log_capture_is_bounded():
    sink = []
    cap = _LogCapture(sink)
    for i in range(_LogCapture.MAX_LINES + 100):
        cap.write(f"line {i}\n")
    assert len(sink) == _LogCapture.MAX_LINES
    assert sink[0] == "line 100"  # oldest lines dropped
    assert sink[-1] == f"line {_LogCapture.MAX_LINES + 99}"


def test_browse_lists_dirs_and_media_files(tmp_path):
    """The GUI file picker lists directories + media files from the server FS."""
    movies = tmp_path / "Movies"
    movies.mkdir()
    (movies / "The.Martian.mkv").write_bytes(b"x")
    (movies / "poster.png").write_bytes(b"x")  # not media -> excluded
    (tmp_path / "Other").mkdir()
    _, c = make_client()
    r = c.get("/api/browse", params={"path": str(movies)})
    assert r.status_code == 200
    d = r.json()
    assert d["path"] == str(movies)
    assert d["files"] == ["The.Martian.mkv"]
    assert d["dirs"] == []
    # parent listing shows both dirs
    r2 = c.get("/api/browse", params={"path": str(tmp_path)})
    assert r2.status_code == 200
    assert "Movies" in r2.json()["dirs"]
    assert "Other" in r2.json()["dirs"]


def test_browse_file_path_resolves_to_parent(tmp_path):
    (tmp_path / "a.mkv").write_bytes(b"x")
    _, c = make_client()
    r = c.get("/api/browse", params={"path": str(tmp_path / "a.mkv")})
    assert r.status_code == 200
    assert r.json()["path"] == str(tmp_path)


def test_browse_exts_filter(tmp_path):
    """The model-path picker filters by extension or lists all files."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    (tmp_path / "model.safetensors").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    _, c = make_client()
    r = c.get("/api/browse", params={"path": str(tmp_path), "exts": ".gguf"})
    assert r.json()["files"] == ["model.gguf"]
    r = c.get("/api/browse", params={"path": str(tmp_path), "exts": "all"})
    assert set(r.json()["files"]) == {"model.gguf", "model.safetensors", "notes.txt"}


def test_browse_missing_dir_returns_400(tmp_path):
    _, c = make_client()
    r = c.get("/api/browse", params={"path": str(tmp_path / "nope")})
    assert r.status_code == 400


def test_timeline_dict_during_scan_uses_live_data():
    """Before a scan completes, the timeline shows live candidates/detections."""
    st = AppState()
    st.scan.duration = 100.0
    st.live_timeline["visual_candidates"].append(
        {"category": "nudity", "start": 10.0, "end": 12.0, "confidence": 0.9, "level": "low"})
    st.live_timeline["language_detections"].append(
        {"category": "foul_language", "start": 5.0, "end": 5.5, "word": "fuck",
         "utterance_start": 4.0, "utterance_end": 6.0, "level": "med", "llm_confirmed": True})
    tl = st.timeline_dict()
    assert tl is not None
    assert tl["duration"] == 100.0
    assert len(tl["visual_candidates"]) == 1
    assert len(tl["language_detections"]) == 1


def test_timeline_dict_completed_scan_has_no_candidates():
    """A completed scan's timeline comes from the result (no generic candidates)."""
    from nuclearcutter.schema import (
        Category, FilmIdentity, ScanResult, SeverityLevel, VisualDetection,
    )

    st = AppState()
    st.scan_result = ScanResult(
        schema_version=1,
        identity=FilmIdentity(title="T", year=2024, duration_seconds=90.0, phash_samples=[]),
        visual_detections=[
            VisualDetection(category=Category.NUDITY, start=1.0, end=2.0,
                            description="x", confidence=0.9, level=SeverityLevel.HIGH),
        ],
        language_detections=[],
    )
    tl = st.timeline_dict()
    assert tl["duration"] == 90.0
    assert tl["visual_candidates"] == []
    assert len(tl["visual_detections"]) == 1


def test_loaded_scan_is_auto_loaded_for_saved_movie(tmp_path):
    """Choosing a movie with existing scan data loads it and marks work done."""
    from nuclearcutter.schema import (
        Category, FilmIdentity, ScanResult, SeverityLevel, VisualDetection,
    )

    movie = tmp_path / "M.mkv"
    movie.write_bytes(b"x")
    # A fully-completed scan also has its transcript cache, so transcription
    # counts as done too.
    (tmp_path / "M.nuclearcutter.transcript.json").write_text('{"utterances": []}')
    scan = ScanResult(
        schema_version=1,
        identity=FilmIdentity(title="M", year=None, duration_seconds=100.0, phash_samples=[]),
        visual_detections=[
            VisualDetection(category=Category.NUDITY, start=1.0, end=2.0,
                            description="x", confidence=0.9, level=SeverityLevel.HIGH),
        ],
        language_detections=[],
    )
    scan.save(tmp_path / "M.nuclearcutter.json")

    st = AppState()
    st.update_settings({"video_path": str(movie)})
    assert st.scan_result is not None
    assert st.scan_result_path == str(tmp_path / "M.nuclearcutter.json")
    assert len(st.scan_result.visual_detections) == 1
    # Progress bars reflect the completed work.
    assert st.scan.status == "done"
    assert st.scan.steps == {"transcribe": 1.0, "scan": 1.0, "verify": 1.0, "summary": 1.0, "render": 0.0}
    assert st.scan.duration == 100.0
    # Timeline now comes from the loaded scan (no generic candidates).
    tl = st.timeline_dict()
    assert tl["visual_candidates"] == []
    assert len(tl["visual_detections"]) == 1


def test_no_scan_loaded_when_movie_has_no_scan_file(tmp_path):
    st = AppState()
    movie = tmp_path / "M.mkv"
    movie.write_bytes(b"x")
    st.update_settings({"video_path": str(movie)})
    assert st.scan_result is None
    assert st.timeline_dict() is None


def test_show_prompts_logs_requests_with_images(tmp_path):
    """With show_prompts on, model prompts/responses stream to the terminal
    with downscaled image thumbnails."""
    import base64
    import io

    from PIL import Image

    st = AppState()
    st.show_prompts = True
    buf = io.BytesIO()
    Image.new("RGB", (320, 180), (20, 40, 60)).save(buf, format="JPEG")
    img_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    payload = {
        "model": "qwen/qwen3-vl-8b",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Is this flagged?"},
            {"type": "image_url", "image_url": {"url": img_uri}},
        ]}],
    }
    data = {"choices": [{"message": {"content": '{"contains_flagged_content": false}'}}]}
    st._log_request("scan", payload, data)

    lines = [e for e in st.all_logs if e["job"] == "scan"]
    assert any("Is this flagged?" in e["line"] for e in lines)
    assert any("←" in e["line"] and "contains_flagged_content" in e["line"] for e in lines)
    img_entries = [e for e in lines if e.get("imgs")]
    assert img_entries, "image thumbnails must be logged"
    assert img_entries[0]["imgs"][0].startswith("data:image/jpeg;base64,")


def test_show_prompts_off_logs_nothing():
    st = AppState()
    st.show_prompts = False
    st._log_request("scan", {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                    {"choices": [{"message": {"content": "ok"}}]})
    assert st.all_logs == []


def test_thumbnail_scales_down():
    import base64
    import io

    from PIL import Image

    from nuclearcutter.server import _thumbnail_data_uri

    buf = io.BytesIO()
    Image.new("RGB", (640, 360), (10, 20, 30)).save(buf, format="JPEG")
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    thumb = _thumbnail_data_uri(uri, max_height=16)
    img = Image.open(io.BytesIO(base64.b64decode(thumb.partition(",")[2])))
    assert img.height <= 16
    assert img.width < 640


def test_backend_validation_local_model_path(tmp_path):
    st, c = make_client()
    # mlx-vlm with a nonexistent model path -> 400 at scan start.
    st.video_path = str(tmp_path / "M.mkv")
    (tmp_path / "M.mkv").write_bytes(b"x")
    st.model_path = "/nonexistent/model"
    r = c.post("/api/scan/start")
    assert r.status_code == 400
    assert "model_path not found" in r.json()["detail"]


def test_settings_persist_across_restart(tmp_path):
    """Settings are saved to the server's settings file and reloaded on startup."""
    path = tmp_path / "settings.json"
    st1 = AppState(settings_path=path)
    st1.update_settings({
        "video_path": "/tmp/Movie.mkv",
        "scale": "720p",
        "levels": {"nudity": "high"},
        "audio_actions": {"foul_language": "mute_word"},
    })
    assert path.exists()
    # Atomic write — no temp file left behind.
    assert not (tmp_path / "settings.json.tmp").exists()

    # A "restarted" server reads the same values back.
    st2 = AppState(settings_path=path)
    assert st2.video_path == "/tmp/Movie.mkv"
    assert st2.scale == "720p"
    assert st2.levels["nudity"] == "high"
    assert st2.audio_actions["foul_language"] == "mute_word"


def test_corrupt_settings_file_is_ignored(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{ not valid json")
    st = AppState(settings_path=path)  # must not raise
    assert st.video_path == ""
    assert st.backend == "mlx-vlm"


def test_settings_load_is_lenient_per_field(tmp_path):
    """One bad saved field must not reset all the others."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({
        "video_path": "/tmp/Movie.mkv",
        "scale": "not-a-scale",  # invalid -> skipped
        "blur_strength": 1.5,
    }))
    st = AppState(settings_path=path)
    assert st.video_path == "/tmp/Movie.mkv"      # valid field kept
    assert st.blur_strength == 1.5                 # valid field kept
    assert st.scale == "480p"                      # invalid field fell back to default


def test_settings_file_surfaces_in_state(tmp_path):
    path = tmp_path / "settings.json"
    st, c = make_client(AppState(settings_path=path))
    d = c.get("/api/state").json()
    assert d["server"]["settings_file"] == str(path)
