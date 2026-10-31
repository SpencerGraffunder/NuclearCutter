"""Run the browser-side JavaScript as a test.

The GUI is one HTML file with a <script> block, and two separate bugs taught us
that Python tests cannot see it:

  * the VERIFY bar read `steps.verify`, a key no server code has ever set, so it
    animated forever instead of showing where the scan was;
  * removing the "Picture style" checkboxes nearly switched censorship off
    completely, because collectSettings() went on reading the removed checkboxes,
    got [], and an empty renderings list means rendering_allowed() is false for
    every detection.

Both are disagreements between the page and the server. So we serve the real
/api/taxonomy through TestClient, hand it to node, and let the page's own code run
against it. Skipped when node is not installed rather than failing, because the
Linux test box is the one that has it and CI images vary.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nuclearcutter.server import AppState, JobInfo, create_app

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tests" / "gui_dom.test.mjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_gui_javascript_against_the_real_taxonomy(tmp_path):
    client = TestClient(create_app(AppState()))
    tax = client.get("/api/taxonomy").json()
    assert tax["tags"], "the API served an empty vocabulary"

    fixture = tmp_path / "taxonomy.json"
    fixture.write_text(json.dumps(tax))

    proc = subprocess.run(
        ["node", str(SCRIPT)],
        cwd=REPO, capture_output=True, text=True, timeout=180,
        # Inherit PATH: node is commonly installed by nvm under ~/.nvm, and a
        # sanitised PATH hides the very interpreter this test needs, turning a
        # real check into a silent skip on the box that has node.
        env={**os.environ, "NC_TAXONOMY_JSON": str(fixture)},
    )
    out = proc.stdout + proc.stderr
    # Assert on the summary line, not just the exit code, so a harness that stops
    # running assertions halfway cannot pass by running none of them.
    assert "0 failed" in out, f"GUI checks failed:\n{out}"
    assert proc.returncode == 0, f"node exited {proc.returncode}:\n{out}"
    assert "0 passed" not in out, "the GUI harness ran no assertions"


def test_the_page_names_no_key_the_server_never_sends():
    """Same contract, without needing node.

    `steps.verify` broke the bar precisely because it looked like a plausible key.
    The server's own step list is the only place to check a GUI key against.
    """
    html = (REPO / "nuclearcutter" / "static" / "index.html").read_text()
    rows = dict(  # bar row id -> the steps key its setBar() call passes
        (m[0], m[1]) for m in __import__("re").findall(
            r'setBar\("bar-([a-z]+)", "pct-[a-z]+", (?:scanJob|jobs\.render)\.steps\.([a-z]+)\)',
            html))
    assert rows, "no progress bars found in the page — did the markup change?"
    job = JobInfo('render').to_dict()
    assert {"summary", "render"} <= set(job["steps"]), "render job reports no step keys"
    for row, key in rows.items():
        if row in ("transcribe", "language", "scan", "verify"):
            continue  # scan-job keys, asserted by the pytest contract test below
        assert key in job["steps"], f"bar-{row} reads steps.{key}, which no job sends"


def test_step_keys_the_page_reads_are_keys_the_server_sends():
    """The scan-job half of the same contract, incl. the VERIFY/identify misnomer."""
    import re

    html = (REPO / "nuclearcutter" / "static" / "index.html").read_text()
    read = set(re.findall(r'setBar\("bar-[a-z]+", "pct-[a-z]+", scanJob\.steps\.([a-z]+)\)',
                          html))
    assert read, "the page reads no scan step keys"
    # A job at the start of a scan carries every key; that dict is the contract.
    keys = set(JobInfo("scan").steps)
    missing = read - keys
    assert not missing, (
        f"the page reads steps.{sorted(missing)} but the server never sets "
        f"them (it sets {sorted(keys)}) — such a bar animates forever")
    assert "verify" not in read, (
        "VERIFY must read steps.identify: the pass is called identify in the "
        "scanner, the prompts and the phase names, and `verify` was only ever a "
        "bar label, which is how the key went missing for weeks")
