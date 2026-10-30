"""Tests for the web GUI server (nuclearcutter/server.py) — settings
validation, state snapshot, model listing, prompt endpoints, and job guards.
No real model server or movie files are needed."""

import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from nuclearcutter.schema import (
    SCHEMA_VERSION, FilmIdentity, ScanResult, VisualDetection,
)
from nuclearcutter.server import (
    AppState, JobInfo, _LogCapture, _scan_steps, create_app,
)
from nuclearcutter.taxonomy import Minors


def make_client(state=None):
    st = state if state is not None else AppState()
    return st, TestClient(create_app(st))


def test_index_serves_ui():
    _, c = make_client()
    r = c.get("/")
    assert r.status_code == 200
    assert "NUCLEARCUTTER" in r.text
    assert "matrix" in r.text.lower() or "--green" in r.text


def test_number_fields_can_be_cleared_mid_edit():
    """Clearing a number box (e.g. the scan interval) must not pop the old
    value back: applySettingsToUI skips whichever field is focused, and the
    number boxes resync from the server on blur."""
    _, c = make_client()
    page = c.get("/").text
    assert "document.activeElement !== el" in page  # focused field is left alone
    assert '.addEventListener("blur", settingsChanged)' in page  # blur resyncs


def test_favicon_served_and_linked():
    """Issue #14: the UI must ship a favicon — the SVG is served at
    /favicon.svg and the page links it in <head>."""
    _, c = make_client()
    r = c.get("/favicon.svg")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/svg+xml")
    assert "00ff41" in r.text and "ff5f5f" in r.text
    page = c.get("/")
    assert '<link rel="icon" type="image/svg+xml" href="/favicon.svg">' in page.text


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
        "server_address": "http://192.168.4.164",
        "vlm_model": "qwen3-vl-8b",
        "whisper_model": "ggml-small",
        "scale": "720p",
        "sweep_interval": 5,
        "selection": ["nudity.shirtless_male", "gore.blood_heavy"],
        "family_actions": {"language": {"visual": "none", "audio": "mute_word"}},
    }})
    assert ok.status_code == 200
    s = ok.json()["settings"]
    assert s["scale"] == "720p"
    assert s["sweep_interval"] == 5
    assert s["selection"] == ["nudity.shirtless_male", "gore.blood_heavy"]
    assert s["family_actions"]["language"]["audio"] == "mute_word"
    assert s["vlm_model"] == "qwen3-vl-8b"
    assert s["whisper_model"] == "ggml-small"
    assert s["server_address"] == "http://192.168.4.164"


def test_settings_legacy_base_url_alias():
    """Older GUI tabs save base_url (…:8080/v1) — it must fold into the
    single server_address instead of erroring. The port is kept (it becomes
    the LLM port; whisper is the next port), so the derived URLs are
    unchanged."""
    _, c = make_client()
    ok = c.post("/api/settings", json={"settings": {
        "base_url": "http://192.168.4.164:8080/v1",
    }})
    assert ok.status_code == 200
    s = ok.json()["settings"]
    assert s["server_address"] == "http://192.168.4.164:8080"
    # Derived URLs follow the one address.
    d = c.get("/api/state").json()
    assert d["server"]["base_url"] == "http://192.168.4.164:8080/v1"
    assert d["server"]["whisper_base_url"] == "http://192.168.4.164:8081"

    # Invalid values still 400.
    bad = c.post("/api/settings", json={"settings": {"sweep_interval": 0}})
    assert bad.status_code == 400
    bad2 = c.post("/api/settings", json={"settings": {"scale": "4800p"}})
    assert bad2.status_code == 400
    # A picture-only family cannot mute per word — there are no words in it.
    bad3 = c.post("/api/settings", json={"settings": {
        "family_actions": {"gore": {"audio": "mute_word"}}}})
    assert bad3.status_code == 400
    # An unknown tag id in the selection is a 400, not a silently ignored entry:
    # a renamed tag surviving in settings.json would censor nothing, forever.
    bad4 = c.post("/api/settings", json={"settings": {"selection": ["nudity.nude_lady"]}})
    assert bad4.status_code == 400
    # ...and selecting nothing is legal and means "censor nothing".
    assert c.post("/api/settings", json={"settings": {"selection": []}}).status_code == 200


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
        SCHEMA_VERSION, FilmIdentity, LanguageDetection, ScanResult, VisualDetection,
    )

    movie = tmp_path / "M.mkv"
    movie.write_bytes(b"x")
    # Transcript cache exists -> transcription counts as done.
    (tmp_path / "M.nuclearcutter.transcript.json").write_text('{"utterances": []}')
    from nuclearcutter.utils.scan_status import ScanStatus
    st_done = ScanStatus(video="M.mkv", phase="done", duration_seconds=100.0,
                         frames_done=50, frames_total=50, sweep_interval=2.0)
    st_done.write(tmp_path / "M.nuclearcutter.status.json")
    scan = ScanResult(
        schema_version=SCHEMA_VERSION,
        identity=FilmIdentity(title="M", year=None, duration_seconds=100.0, phash_samples=[]),
        visual_detections=[
            VisualDetection(start=1.0, end=2.0, tags=["nudity.shirtless_male"],
                            description="x", confidence=0.9),
        ],
        language_detections=[
            LanguageDetection(start=5.0, end=5.5, utterance_start=4.0,
                              utterance_end=6.0, text="damn",
                              tags=["language.profanity_common"]),
        ],
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
    assert st.scan.steps == _scan_steps(transcribe=1.0, language=0.0, scan=0.0, identify=0.0, render=0.0)


def test_clear_transcription_keeps_scan(tmp_path):
    """Clearing transcription deletes only the transcript cache and drops the
    transcribe bar, leaving the loaded scan/verify results intact."""
    movie = _write_full_scan(tmp_path)
    st = AppState()
    st.update_settings({"video_path": str(movie)})
    assert st.scan.steps == _scan_steps(transcribe=1.0, language=1.0, scan=1.0, identify=1.0, render=0.0)

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
    assert st.scan.steps["identify"] == 1.0


def test_loaded_scan_reflects_missing_transcript(tmp_path):
    """If a loaded result exists but the transcript cache was cleared, the
    transcribe bar shows 0% while scan/verify stay at 100%."""
    movie = _write_full_scan(tmp_path)
    (tmp_path / "M.nuclearcutter.transcript.json").unlink()

    st = AppState()
    st.update_settings({"video_path": str(movie)})
    assert st.scan_result is not None
    assert st.scan.steps == _scan_steps(transcribe=0.0, language=1.0, scan=1.0, identify=1.0, render=0.0)
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
    assert st.scan.steps["identify"] == 0.0
    assert "Transcript cached" in st.scan.message


def test_switching_to_movie_without_scan_resets_status(tmp_path):
    """Issue #13: selecting a file that has no existing scan must reset the
    displayed status to 0 progress instead of leaving the previous movie's
    status on the bars and timeline."""
    movie = _write_full_scan(tmp_path)
    st = AppState()
    st.update_settings({"video_path": str(movie)})
    assert st.scan.status == "done"
    assert st.scan.progress == 1.0

    # Switch to a DIFFERENT movie that has no scan file: everything zeroes.
    other = tmp_path / "Other.mkv"
    other.write_bytes(b"x")
    st.update_settings({"video_path": str(other)})
    assert st.scan_result is None
    assert st.scan_result_path == ""
    assert st.scan.status == "idle"
    assert st.scan.progress == 0.0
    assert st.scan.duration == 0.0
    assert st.scan.steps == _scan_steps(transcribe=0.0, language=0.0, scan=0.0, identify=0.0, render=0.0)
    assert st.timeline_dict() is None


def test_clearing_video_path_resets_status(tmp_path):
    """Selecting no file (empty path) also resets the displayed status."""
    st = AppState()
    st.scan.status = "done"
    st.scan.progress = 0.5
    st.scan.duration = 42.0
    st.live_timeline = {"visual_candidates": [{"start": 1.0, "end": 2.0}],
                        "visual_detections": [], "language_detections": []}
    st.update_settings({"video_path": ""})
    assert st.scan.status == "idle"
    assert st.scan.progress == 0.0
    assert st.scan.duration == 0.0
    assert st.live_timeline == {"visual_candidates": [], "visual_detections": [],
                                "language_detections": []}
    assert st.timeline_dict() is None


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


def test_api_key_setting_roundtrip():
    """The GUI's API key box persists in settings and comes back in state."""
    st, c = make_client()
    r = c.post("/api/settings", json={"settings": {"api_key": "sekrit"}})
    assert r.status_code == 200
    assert r.json()["settings"]["api_key"] == "sekrit"
    assert c.get("/api/state").json()["settings"]["api_key"] == "sekrit"
    assert st.api_key == "sekrit"
    # Blank clears it (a keyless server).
    c.post("/api/settings", json={"settings": {"api_key": "  "}})
    assert st.api_key == ""


def test_models_endpoint_uses_key_override_then_saved_key(monkeypatch):
    """/api/models probes with the query override (fresh paste in the GUI)
    and falls back to the saved setting when no override is given."""
    st, c = make_client()
    st.api_key = "saved-key"
    seen = {}

    def fake_list_models(self):
        seen["api_key"] = self.config.api_key
        return ["m"]

    monkeypatch.setattr("nuclearcutter.server.LLMClient.list_models", fake_list_models)
    c.get("/api/models", params={"api_key": "box-key"})
    assert seen["api_key"] == "box-key"
    c.get("/api/models")
    assert seen["api_key"] == "saved-key"
    st.api_key = ""
    c.get("/api/models")
    assert seen["api_key"] == "not-needed"  # keyless default keeps old behavior


def test_taxonomy_endpoint_is_the_single_source_of_the_chooser():
    """The GUI builds "what do you want to hide?" from this payload.

    Worth pinning: the whole point of serving the vocabulary is that adding a
    category in one Python file makes it appear in the browser. If the endpoint
    stops reporting a real hint, or drops a family's audio options, the UI
    silently shows a checkbox that censors nothing.
    """
    _, c = make_client()
    r = c.get("/api/taxonomy")
    assert r.status_code == 200
    d = r.json()
    assert d["taxonomy_version"] == 2
    assert d["schema_version"] == 2

    fam_ids = [f["id"] for f in d["families"]]
    assert "gore" in fam_ids and "language" in fam_ids
    for fam in d["families"]:
        assert fam["label"] and fam["blurb"]
        # Exactly one control per medium the family exists in. A family with no
        # picture (language) must not offer a blur, or the GUI ships a dropdown
        # that changes nothing.
        assert fam["visual"] or fam["audio"], "every family needs one action"
        assert fam["default_visual"] in (fam["visual"] or ["none"])
        assert fam["default_audio"] in fam["audio"]
    lang = next(f for f in d["families"] if f["id"] == "language")
    assert lang["visual"] == [], "spoken words cannot be blurred"
    assert "mute_word" in lang["audio"]
    gore = next(f for f in d["families"] if f["id"] == "gore")
    assert gore["visual"] and "mute_word" not in gore["audio"], \
        "a picture has no words to mute one at a time"

    tags = d["tags"]
    assert len(tags) > 80, "the chooser is useless if the vocabulary is truncated"
    by_family = {}
    for t in tags:
        by_family.setdefault(t["family"], []).append(t)
        assert t["id"].startswith(t["family"] + "."), "ids are grouped by prefix"
        assert len(t["hint"]) > 20, f"{t['id']} has no usable hint"
        assert t["label"]
    # Gender split where the user asked for it, and not where they didn't.
    assert any(t["gender"] == "female" for t in by_family["nudity"])
    assert any(t["gender"] == "male" for t in by_family["nudity"])
    assert all(t["gender"] == "any" for t in by_family["gore"])
    assert all(t["gender"] == "any" for t in by_family["violence"])

    assert d["default_selection"] == []
    # "everything" is every tag a human may choose; the child-protection rule is
    # not a preference and must not appear as a checkbox.
    assert set(d["presets"]["everything"]) == {
        t["id"] for t in tags if not t["always_censor"]}
    assert len(d["presets"]["everything"]) < len(tags), \
        "an always-on tag must not be offered as a choice"
    assert d["presets"]["none"] == []
    from nuclearcutter.taxonomy import Rendering
    assert {r["id"] for r in d["rendering"]} == {r.value for r in Rendering}
    # Live action must be selectable on its own: it is the single most common way
    # people want to narrow a filter ("censor the gore, not the cartoon").
    assert "live_action" in {r["id"] for r in d["rendering"]}
    assert all(r["label"] for r in d["rendering"])
    # The minor flag is a fact about the film, never a checkbox you can turn off.
    assert any(m["id"] == "yes" for m in d["minors"])


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
        {"tags": ["nudity.shirtless_male"], "start": 10.0, "end": 12.0,
         "confidence": 0.9})
    st.live_timeline["language_detections"].append(
        {"tags": ["language.profanity_common"], "text": "damn", "start": 5.0,
         "end": 5.5, "utterance_start": 4.0, "utterance_end": 6.0,
         "llm_confirmed": True})
    tl = st.timeline_dict()
    assert tl is not None
    assert tl["duration"] == 100.0
    assert len(tl["visual_candidates"]) == 1
    assert len(tl["language_detections"]) == 1


def test_timeline_dict_completed_scan_has_no_candidates():
    """A completed scan's timeline comes from the result (no generic candidates)."""
    from nuclearcutter.schema import (
        FilmIdentity, ScanResult, VisualDetection,
    )

    st = AppState()
    st.scan_result = ScanResult(
        schema_version=1,
        identity=FilmIdentity(title="T", year=2024, duration_seconds=90.0, phash_samples=[]),
        visual_detections=[
            VisualDetection(start=1.0, end=2.0, tags=["nudity.shirtless_male"],
                            description="x", confidence=0.9),
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
        SCHEMA_VERSION, FilmIdentity, LanguageDetection, ScanResult, VisualDetection,
    )

    movie = tmp_path / "M.mkv"
    movie.write_bytes(b"x")
    # A fully-completed scan also has its transcript cache, so transcription
    # counts as done too.
    (tmp_path / "M.nuclearcutter.transcript.json").write_text('{"utterances": []}')
    scan = ScanResult(
        schema_version=SCHEMA_VERSION,
        identity=FilmIdentity(title="M", year=None, duration_seconds=100.0, phash_samples=[]),
        visual_detections=[
            VisualDetection(start=1.0, end=2.0, tags=["nudity.shirtless_male"],
                            description="x", confidence=0.9),
        ],
        language_detections=[
            LanguageDetection(start=5.0, end=5.5, utterance_start=4.0,
                              utterance_end=6.0, text="damn",
                              tags=["language.profanity_common"]),
        ],
    )
    scan.save(tmp_path / "M.nuclearcutter.json")

    st = AppState()
    st.update_settings({"video_path": str(movie)})
    assert st.scan_result is not None
    assert st.scan_result_path == str(tmp_path / "M.nuclearcutter.json")
    assert len(st.scan_result.visual_detections) == 1
    # Progress bars reflect the completed work.
    assert st.scan.status == "done"
    assert st.scan.steps == _scan_steps(transcribe=1.0, language=1.0, scan=1.0, identify=1.0, render=0.0)
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


def test_terminal_thumbs_use_triple_res_source():
    """Hover popups render the same data URI at CSS-pinned sizes; the source
    got 3x headroom (48px, q72) for crisp popups without layout changes."""
    import base64
    import io

    from PIL import Image

    from nuclearcutter.server import _thumbnail_data_uri

    buf = io.BytesIO()
    Image.new("RGB", (640, 360), (10, 20, 30)).save(buf, format="JPEG")
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    thumb = _thumbnail_data_uri(uri, max_height=48, quality=72)
    img = Image.open(io.BytesIO(base64.b64decode(thumb.partition(",")[2])))
    assert img.height == 48 and img.width == 85  # aspect preserved at the new cap

    # the terminal's vision-log call site must actually request the bigger box
    import inspect

    import nuclearcutter.server as srv

    assert "max_height=48" in inspect.getsource(srv)


def test_scan_requires_vlm_model(tmp_path):
    st, c = make_client()
    # No VLM model configured -> 400 at scan start (before any server check).
    st.video_path = str(tmp_path / "M.mkv")
    (tmp_path / "M.mkv").write_bytes(b"x")
    st.vlm_model = ""
    r = c.post("/api/scan/start")
    assert r.status_code == 400
    assert "VLM model" in r.json()["detail"]


def test_settings_persist_across_restart(tmp_path):
    """Settings are saved to the server's settings file and reloaded on startup."""
    path = tmp_path / "settings.json"
    st1 = AppState(settings_path=path)
    st1.update_settings({
        "video_path": "/tmp/Movie.mkv",
        "scale": "720p",
        "selection": ["nudity.shirtless_male"],
        "family_actions": {"language": {"visual": "none", "audio": "mute_word"}},
        "audio_actions": {"foul_language": "mute_word"},
    })
    assert path.exists()
    # Atomic write — no temp file left behind.
    assert not (tmp_path / "settings.json.tmp").exists()

    # A "restarted" server reads the same values back.
    st2 = AppState(settings_path=path)
    assert st2.video_path == "/tmp/Movie.mkv"
    assert st2.scale == "720p"
    assert st2.selection == ["nudity.shirtless_male"]
    assert st2.family_actions["language"]["audio"] == "mute_word"


def test_corrupt_settings_file_is_ignored(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{ not valid json")
    st = AppState(settings_path=path)  # must not raise
    assert st.video_path == ""
    assert st.llm_base_url  # defaults stay in place


def test_legacy_settings_migrate_to_server_address(tmp_path):
    """A pre-unification settings.json (base_url + whisper_base_url +
    whisper_models_dir) must load as one server_address, with the models
    dir dropped (auto-detected now)."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({
        "video_path": "/tmp/Movie.mkv",
        "base_url": "http://192.168.4.164:8080/v1",
        "whisper_base_url": "http://192.168.4.164:8081",
        "whisper_models_dir": "/some/old/models",
        "whisper_model": "ggml-base.en",
        "vlm_model": "qwen",
        "text_model": "qwen",
    }))
    st = AppState(settings_path=path)
    assert st.server_address == "http://192.168.4.164:8080"  # port kept = LLM port
    assert st.llm_base_url == "http://192.168.4.164:8080/v1"
    assert st.whisper_base_url == "http://192.168.4.164:8081"
    assert st.whisper_model == "ggml-base.en"
    assert st.video_path == "/tmp/Movie.mkv"
    # The migration rewrote the file in the new shape.
    data = json.loads(path.read_text())
    assert data["server_address"] == "http://192.168.4.164:8080"
    assert "whisper_models_dir" not in data


def test_whisper_models_endpoint_shape():
    """/api/whisper-models returns {id,label,loaded} entries (the GUI's
    whisper dropdown expects objects, not bare strings)."""
    st, c = make_client()
    with patch(
        "nuclearcutter.server.whisper_list_models",
        return_value=[{"id": "ggml-small", "label": "ggml-small", "loaded": True}],
    ), patch("nuclearcutter.server.whisper_loaded_model_id", return_value="ggml-small"), \
         patch("nuclearcutter.server.detect_whisper_models_dir", return_value="/models"), \
         patch("nuclearcutter.server.whisper_is_up", return_value=True):
        r = c.get("/api/whisper-models")
    assert r.status_code == 200
    d = r.json()
    assert d["models"][0]["id"] == "ggml-small"
    assert d["models"][0]["loaded"] is True
    assert d["loaded"] == "ggml-small"
    assert d["models_dir"] == "/models"


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


# ---------------------------------------------------------------- frame preview

def test_frame_requires_video():
    _, c = make_client()
    r = c.get("/api/frame", params={"t": 1.0})
    assert r.status_code == 400
    assert "no movie loaded" in r.json()["detail"]


def test_frame_rejects_bad_inputs(tmp_path):
    movie = tmp_path / "M.mkv"
    movie.write_bytes(b"x")
    st, c = make_client()
    c.post("/api/settings", json={"settings": {"video_path": str(movie)}})
    # Negative timestamp is a semantic error -> our 400.
    assert c.get("/api/frame", params={"t": -1}).status_code == 400
    # Non-numeric values are type errors -> FastAPI's 422.
    assert c.get("/api/frame", params={"t": "abc"}).status_code == 422
    assert c.get("/api/frame", params={"t": 1.0, "h": "xyz"}).status_code == 422


def test_frame_returns_jpeg_and_caches(tmp_path):
    """A valid request returns a JPEG; the same (ts, height) request is a
    cache hit — ffmpeg is invoked exactly once."""
    movie = tmp_path / "M.mkv"
    movie.write_bytes(b"x")
    st, c = make_client()
    c.post("/api/settings", json={"settings": {"video_path": str(movie)}})
    calls = []

    def fake_extract(video, timestamp_seconds, scale_height=None):
        calls.append((video, timestamp_seconds, scale_height))
        return b"\xff\xd8fake-jpeg\xff\xd9"

    with patch("nuclearcutter.utils.ffmpeg.extract_frame_jpeg_bytes",
               side_effect=fake_extract) as m:
        r1 = c.get("/api/frame", params={"t": 12.34, "h": 320})
        r2 = c.get("/api/frame", params={"t": 12.3, "h": 320})
    assert r1.status_code == 200
    assert r1.headers["content-type"] == "image/jpeg"
    assert r1.content == b"\xff\xd8fake-jpeg\xff\xd9"
    # 12.34 rounds to 12.3 -> same cache key as r2.
    assert r2.content == r1.content
    assert len(calls) == 1
    # Height is passed through (clamped server-side to the stamp range).
    assert calls[0][1] == 12.34 and calls[0][2] == 320


def test_frame_height_clamped(tmp_path):
    movie = tmp_path / "M.mkv"
    movie.write_bytes(b"x")
    st, c = make_client()
    c.post("/api/settings", json={"settings": {"video_path": str(movie)}})
    with patch("nuclearcutter.utils.ffmpeg.extract_frame_jpeg_bytes",
               return_value=b"\xff\xd8x\xff\xd9") as m:
        c.get("/api/frame", params={"t": 1.0, "h": 5000})
    assert m.call_args.kwargs["scale_height"] == st.FRAME_MAX_HEIGHT
    m.reset_mock()
    with patch("nuclearcutter.utils.ffmpeg.extract_frame_jpeg_bytes",
               return_value=b"\xff\xd8x\xff\xd9") as m:
        c.get("/api/frame", params={"t": 1.0, "h": 2})
    assert m.call_args.kwargs["scale_height"] == st.FRAME_MIN_HEIGHT


def test_frame_extraction_failure_is_500(tmp_path):
    movie = tmp_path / "M.mkv"
    movie.write_bytes(b"x")
    st, c = make_client()
    c.post("/api/settings", json={"settings": {"video_path": str(movie)}})
    with patch("nuclearcutter.utils.ffmpeg.extract_frame_jpeg_bytes",
               side_effect=RuntimeError("boom")):
        r = c.get("/api/frame", params={"t": 1.0})
    assert r.status_code == 500
    assert "frame extraction failed" in r.json()["detail"]


def test_switching_movie_resets_render_display(tmp_path):
    """Switching to a different movie also clears the RENDER job's display —
    its progress belongs to the previous movie."""
    movie = _write_full_scan(tmp_path)
    st = AppState()
    st.update_settings({"video_path": str(movie)})
    st.render.status = "done"
    st.render.progress = 1.0
    st.render.steps["render"] = 1.0

    other = tmp_path / "Other.mkv"
    other.write_bytes(b"x")
    st.update_settings({"video_path": str(other)})
    assert st.render.status == "idle"
    assert st.render.progress == 0.0
    assert st.render.steps["render"] == 0.0


def test_unrelated_settings_tweak_keeps_render_display(tmp_path):
    """The GUI re-POSTs the full settings (video_path included) on every
    tweak, so an UNCHANGED movie path must reset nothing — a completed
    render's done/100% display survives."""
    movie = _write_full_scan(tmp_path)
    st = AppState()
    st.update_settings({"video_path": str(movie)})
    st.render.status = "done"
    st.render.progress = 1.0
    st.render.steps["render"] = 1.0

    st.update_settings({"video_path": str(movie), "blur_strength": 2.0})
    assert st.render.status == "done"
    assert st.render.progress == 1.0
    assert st.render.steps["render"] == 1.0
    # The scan display is untouched too (the movie still has its scan).
    assert st.scan.status == "done"


def test_pwa_manifest_served_without_theme_keys():
    """The manifest must parse, offer the 512 icon for installability, and
    carry NO theme_color/background_color (Spec: no theme change)."""
    _, c = make_client()
    r = c.get("/manifest.webmanifest")
    assert r.status_code == 200
    assert "manifest+json" in r.headers["content-type"]
    m = json.loads(r.text)
    assert m["display"] == "standalone"
    assert "theme_color" not in m and "background_color" not in m
    assert any(i["sizes"] == "512x512" for i in m["icons"])


def test_pwa_icons_served_as_png():
    _, c = make_client()
    for name in ("apple-touch-icon.png", "icon-192.png", "icon-512.png",
                 "icon-maskable-512.png"):
        r = c.get(f"/icons/{name}")
        assert r.status_code == 200, name
        assert r.content[:4] == b"\x89PNG", name


def test_pwa_icon_allowlist_blocks_anything_else():
    """Allow-list first, join second: odd names and traversal must 404."""
    _, c = make_client()
    assert c.get("/icons/index.html").status_code == 404
    assert c.get("/icons/..%2f..%2fsettings.json").status_code == 404
    assert c.get("/icons/settings.json").status_code == 404


# --------------------------------------------------------------------------
# The single-page GUI, as a contract with the server
# --------------------------------------------------------------------------

def _page(c):
    return c.get("/").text


def test_the_page_has_no_leftovers_from_the_severity_ui():
    """index.html is one file with the whole UI in it, and the old levels table
    was 60 lines of hardcoded HTML. Any of these strings coming back means a
    widget that talks to an endpoint which no longer exists — it would fail
    silently in the browser, one click at a time."""
    page = _page(make_client()[1])
    for dead in ("/api/definitions", "/api/detection/override", "LEVEL_RANK",
                 "matrixValues", "OVERRIDE_OPTIONS", "exhigh", "foul_language",
                 'data-key="level"', "level_override"):
        assert dead not in page, f"the page still uses {dead!r}"


def test_the_page_builds_the_chooser_from_the_api_not_from_itself():
    """The point of /api/taxonomy is that the browser ships no category list. If
    someone hardcodes one, a tag added in taxonomy.py appears in scans but not in
    the UI — the worst kind of miss, because the film IS being filtered without
    the viewer being offered the choice."""
    page = _page(make_client()[1])
    assert 'fetch("/api/taxonomy")' in page
    assert "buildChooser" in page
    for hardcoded in ("nudity.shirtless_male", "gore.blood_heavy"):
        assert hardcoded not in page, f"{hardcoded} is hardcoded in the page"


def test_page_offers_the_controls_the_settings_api_accepts():
    """Every key collectSettings() sends must be one update_settings() reads, and
    vice versa — otherwise a setting is silently dropped on the floor."""
    page = _page(make_client()[1])
    for key in ("selection:", "family_actions:", "rendering:", "minors:",
                "polish_descriptions:"):
        assert key in page, f"collectSettings no longer sends {key}"
    _, c = make_client()
    settings = c.get("/api/state").json()["settings"]
    for key in ("selection", "family_actions", "rendering", "minors",
                "polish_descriptions"):
        assert key in settings


def test_page_renders_marks_by_family_class_the_server_sends():
    """_visual_marker colours marks via `families`; a family with no CSS class
    draws an invisible mark — the detection is there and the user cannot see it."""
    from nuclearcutter.taxonomy import Family

    page = _page(make_client()[1])
    for fam in Family:
        assert f".tl-mark.{fam.value}" in page, f"no mark colour for {fam.value}"


def test_page_search_matches_tag_hints_not_only_labels():
    """Searching 'surgery' has to find the wound tags, whose hints mention it and
    whose labels do not. Otherwise the search box is a label filter and people
    conclude a category does not exist."""
    page = _page(make_client()[1])
    assert "tag-search" in page
    assert "hidden-by-search" in page
    assert "tag-hint" in page


def test_detection_edit_endpoint_rejects_an_out_of_range_index():
    _, c = make_client()
    r = c.post("/api/detection/edit",
               json={"type": "visual", "index": 0, "removed": True})
    assert r.status_code == 400  # no scan loaded at all


def test_detection_edit_changes_tags_and_the_dismissed_flag():
    st = AppState()
    st.update_settings({"video_path": ""})
    st.scan_result = ScanResult(
        schema_version=SCHEMA_VERSION,
        identity=FilmIdentity(title="T", year=None, duration_seconds=60.0,
                              phash_samples=[]),
        visual_detections=[VisualDetection(start=1.0, end=2.0,
                                           tags=["nudity.shirtless_male"],
                                           description="x", confidence=0.9)],
        language_detections=[],
    )
    _client = TestClient(create_app(st))

    # 1) dismissing keeps the detection on the timeline, it only stops censoring.
    r = _client.post("/api/detection/edit",
                     json={"type": "visual", "index": 0, "removed": True})
    assert r.status_code == 200
    assert st.scan_result.visual_detections[0].removed is True
    assert st.scan_result.visual_detections[0].tags == ["nudity.shirtless_male"]
    marker = st.timeline_dict()["visual_detections"][0]
    assert marker["removed"] is True
    assert marker["tags"] == ["nudity.shirtless_male"], \
        "dismissing must not rewrite what the scan saw"

    # 2) correcting the tags is a statement about what is on screen, so it is a
    #    different edit from a dismissal and both must be possible independently.
    r = _client.post("/api/detection/edit",
                     json={"type": "visual", "index": 0,
                           "tags": ["gore.blood_moderate", "violence.beating_prolonged"]})
    assert r.status_code == 200
    det = st.scan_result.visual_detections[0]
    assert det.tags == ["gore.blood_moderate", "violence.beating_prolonged"]
    assert det.removed is True  # the dismissal was not silently reset

    # 3) editing the tags of a detection that involves a minor re-applies the
    #    child floor: removing the exposure tags by hand must not walk around it.
    st.scan_result.visual_detections[0].minors = Minors.YES.value
    st.scan_result.visual_detections[0].tags = ["nudity.breasts_full_female"]
    st.scan_result.visual_detections[0].removed = True
    r = _client.post("/api/detection/edit",
                     json={"type": "visual", "index": 0, "tags": ["violence.argument"]})
    assert r.status_code == 200, r.text
    protected = st.scan_result.visual_detections[0]
    assert "safety.exposure_with_minor" in protected.tags, \
        "the floor tag must come back even though the viewer asked to replace them"

    # 4) an unknown tag id is refused rather than stored: it would match nothing.
    r = _client.post("/api/detection/edit",
                     json={"type": "visual", "index": 0, "tags": ["nudity.nope"]})
    assert r.status_code == 400


def test_timeline_marks_carry_labels_the_browser_cannot_compute():
    """The browser has no copy of the vocabulary, so the server must hand it the
    human names and families for every mark."""
    st = AppState()
    st.scan_result = ScanResult(
        schema_version=SCHEMA_VERSION,
        identity=FilmIdentity(title="T", year=None, duration_seconds=60.0,
                              phash_samples=[]),
        visual_detections=[VisualDetection(start=1.0, end=2.0,
                                           tags=["nudity.shirtless_male"],
                                           description="x", confidence=0.9,
                                           minors=Minors.UNCERTAIN.value,
                                           triggers=["skin"])],
        language_detections=[],
    )
    marker = TestClient(create_app(st)).get("/api/state").json()["timeline"]["visual_detections"][0]
    assert marker["kinds"] == ["Shirtless (men)"]
    assert marker["families"] == ["nudity"]
    assert marker["minors"] == Minors.UNCERTAIN.value
    assert marker["triggers"] == ["skin"]
    assert marker["tags"] == ["nudity.shirtless_male"]


def test_legend_colours_and_names_every_family():
    """A timeline mark in a colour the legend does not name is a mystery swatch.

    The colours are necessarily declared twice (CSS cannot read JSON), so what the
    test pins is the thing that does not have to be duplicated: coverage. Every
    family the API reports must have BOTH a `.tl-mark.<family>` colour and a place
    in the legend colour table, and no family may be named in the legend colour
    table that the API no longer reports (a stale swatch teaches people to
    distrust the legend).
    """
    from nuclearcutter.taxonomy import Family

    page = _page(make_client()[1])
    table = page[page.index("const LEGEND_COLORS = {"):]
    table = table[:table.index("};")]
    for fam in Family:
        assert f".tl-mark.{fam.value}" in page, f"no timeline colour for {fam.value}"
        assert f"{fam.value}:" in table, f"{fam.value} missing from LEGEND_COLORS"
    for stale in ("nudity_candidate", "foul_language", "intimate"):
        assert f"{stale}:" not in table, f"{stale} is not a family but is in the legend"


def test_preset_buttons_follow_the_server_without_naming_families_wrongly():
    """The button row is built from preset_order, but the tooltips are words a
    human wrote — and a tooltip that promises more than the preset contains is a
    small lie with censoring consequences.

    `graphic` specifically must not advertise self-harm: it is the button a viewer
    reaches for when a film has a battle in it, and inheriting suicide-method
    detail from it is the worst surprise this product could produce.
    """
    page = _page(make_client()[1])
    _, c = make_client()
    presets = c.get("/api/taxonomy").json()["presets"]

    assert "preset_order" in page, "the button row stopped following the server's order"
    for name in presets:
        assert f"{name}:" in page, f"no GUI label for the {name!r} preset"

    graphic_line = next(ln for ln in page.splitlines() if "graphic:" in ln and "Graphic" in ln)
    assert "self-harm" in graphic_line.lower(), "graphic tooltip should state the exclusion"
    assert "not self-harm" in graphic_line.lower(), "graphic tooltip must EXCLUDE self-harm"
    assert not any(t.startswith("harm.") for t in presets["graphic"])


def test_every_scan_phase_that_moves_the_job_also_moves_its_message():
    """A progress line that stops moving is worse than no progress line.

    `job.message` is set once to "Connecting to the model server..." before the
    scan starts, and each phase branch of the progress callback used to update
    only `phase`/counters — so the GUI displayed "connecting" for the five hours
    of a sweep that was plainly running, and the honest reading of the screen was
    "the job is stuck on a dead backend". Pressing stop felt like the only option.

    This checks the invariant structurally, per `elif stage == ...` branch: a
    branch that assigns job.phase must also assign job.message. Testing it
    behaviourally would need a whole fake scan_pass, which is far more machinery
    than the bug is worth, and the machinery would be what people maintain.
    """
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "nuclearcutter" / "server.py").read_text()
    body = src[src.index("def _progress(stage, detail)"):]
    body = body[:body.index("result = scan_pass(")]

    # Split on BOTH `if` and `elif`, or the first phase (transcribing — the one
    # that waits on whisper, and the first thing a viewer watches) escapes the
    # check while every later phase is held to it.
    branches = re.split(r"\n                (?:elif|if) stage == ", body)
    assert len(branches) > 5, "the progress callback lost its phase branches"
    silent = []
    for br in branches[1:]:
        stage = br.split('"')[1]
        code = br.split("elif stage ==")[0].split("if stage ==")[0]
        if "job.phase = " in code and "job.message = " not in code:
            silent.append(stage)
    assert not silent, f"phase(s) update without a message: {silent}"


def test_a_stale_status_file_does_not_seed_the_live_timeline(tmp_path):
    """A scan paints its timeline from the last status file — but only its own kind.

    The seed used to read whatever file was there. Scan a film that has a v1
    status file next to it and ~1000 v1 candidates appeared on the v2 timeline, each
    with a `category` and a `level` the new UI does not have: the film looked far
    more flagged than the scan had found, and the rows never cleared because the
    new scan cannot address what it did not produce. The scanner already refuses to
    RESUME across a taxonomy change; displaying across one was the leak.
    """
    import json

    video = tmp_path / "Film.mkv"
    video.write_bytes(b"\0")
    status = tmp_path / "Film.nuclearcutter.status.json"
    stale = {
        "SCHEMA": 1,
        "visual_candidates": [{"category": "violence", "level": "low",
                               "start": 340.0, "end": 340.0, "confidence": 0.95}],
        "visual_detections": [{"category": "nudity", "level": "high",
                               "start": 90.0, "end": 100.0, "description": "old"}],
        "language_detections": [{"word": "damn", "start": 12.0, "end": 12.4}],
    }
    status.write_text(json.dumps(stale))

    st = AppState()
    st.video_path = str(video)
    seeded = st._seed_live_timeline(video)
    assert seeded["visual_candidates"] == [], "v1 candidates leaked into a v2 timeline"
    assert seeded["visual_detections"] == [], "v1 detections leaked into a v2 timeline"
    assert seeded["language_detections"] == []

    # A current-schema file still seeds, or resume loses the progress it exists for.
    from nuclearcutter.utils.scan_status import SCHEMA_VERSION

    fresh = dict(stale, SCHEMA=SCHEMA_VERSION,
                 visual_candidates=[{"triggers": ["skin"], "start": 30.0,
                                     "end": 30.0, "confidence": 0.6}])
    status.write_text(json.dumps(fresh))
    seeded = st._seed_live_timeline(video)
    assert len(seeded["visual_candidates"]) == 1, "a resumable scan started from nothing"

    # Unreadable file: no crash, empty timeline (the scan rebuilds it).
    status.write_text("{not json")
    assert st._seed_live_timeline(video)["visual_candidates"] == []


def test_step_keys_the_page_reads_are_keys_the_server_sends():
    """Every `steps.<key>` the STATUS panel reads must exist on the server.

    This is the test that should have existed when the panel was written. VERIFY read
    `scanJob.steps.verify`; no server code has ever set `verify` (the pass is called
    `identify` in the scanner, the prompts and the phase names). The row therefore got
    `undefined`, and setBar's old `val == null` check painted it as the indeterminate
    sliding block — the bar moved constantly for hours and reported no position at
    all, while a real (done, total) pair sat in steps.identify going unread. The user
    asked for a progress bar that shows WHERE the job is; that is a contract, and this
    checks the contract instead of the pixels.

    Checks both directions:
      - a row reading a key nobody sets  -> dead bar (this bug)
      - a key the server reports that no row displays -> a stage with no bar, so the
        panel claims to show where the job is while a whole pass runs invisible
        (steps.language had no row for exactly this reason)
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    page = (root / "nuclearcutter" / "static" / "index.html").read_text()
    srv = (root / "nuclearcutter" / "server.py").read_text()

    # Keys the page reads: setBar(..., X.steps.key)
    read = set(re.findall(r"\.steps\.([a-z_]+)", page))
    assert read, "the panel stopped reading job.steps at all"

    # Keys the server produces, from the one function that builds them.
    helper = srv[srv.index("def _scan_steps("):]
    helper = helper[:helper.index("return steps")]
    emitted = set(re.findall(r'"([a-z_]+)": 0\.0', helper))
    assert len(emitted) >= 5, f"could not read the step list from _scan_steps: {emitted}"

    missing = sorted(read - emitted)
    assert not missing, (
        f"the panel reads steps.{missing}, which no server code sets — that row will "
        "draw as broken/no data. Either name the right key or add it to _scan_steps()"
    )

    undisplayed = sorted(emitted - read)
    assert not undisplayed, (
        f"steps.{undisplayed} is reported but no bar row reads it — that stage runs "
        "with nothing on screen showing it"
    )


def test_scan_job_step_dicts_always_carry_every_key():
    """A half-updated literal at one of the five scan checkpoints used to be enough
    to break a bar, because a dict missing a key is indistinguishable from a step that
    has not started. _scan_steps() is now the only way to build one; assert no call
    site has regressed to a bare literal.
    """
    from pathlib import Path
    import re

    srv = (Path(__file__).resolve().parent.parent
           / "nuclearcutter" / "server.py").read_text()
    leftovers = re.findall(r"\.steps = \{[^{}]*\}", srv)
    bad = [b for b in leftovers if "transcribe" in b]
    assert not bad, f"bare steps literals remain (use _scan_steps): {bad}"

    # And the helper really does cover all of them.
    st = AppState()
    assert set(st.scan.steps) == set(_scan_steps()), "job default diverged from _scan_steps()"
    for key, val in st.scan.steps.items():
        assert isinstance(val, float), f"steps.{key} must be a float, got {val!r}"
