"""End-to-end smoke tests for the vault-search CLI tools.

Stdlib only. Builds a small synthetic vault, runs the scripts as subprocesses
with an isolated HOME, and stands in for Ollama with a tiny in-process HTTP
server that speaks the same API shape (/api/embed, /api/embeddings,
/api/generate). Run with:

    python3 -m unittest discover -s tests -v
"""

import hashlib
import importlib.util
import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

NOTES = {
    "Knowledge/dopamine.md": (
        "---\ntitle: Dopamine and Reward Prediction Error\ntags: [neuro]\n---\n"
        "# Dopamine\n"
        "The phasic dopamine signal encodes a temporal difference error, identical "
        "in structure to the TD learning update in [[reinforcement learning]].\n"
    ),
    "Knowledge/td-learning.md": (
        "# Temporal Difference Learning\n"
        "TD learning updates value estimates using bootstrapped targets. "
        "Related: [[dopamine]], reinforcement learning, Q-learning.\n"
    ),
    "Knowledge/attention.md": (
        "# Attention Mechanism\n"
        "The transformer attention mechanism computes weighted sums over values "
        "using query-key similarity.\n"
    ),
    "Daily/2026-04-01.md": (
        "# Daily note\nRead a paper on attention and sparse transformers.\n"
    ),
}
# Filler so BM25 IDF is meaningful (FTS5 gives ~0 IDF to terms in half the docs)
NOTES.update({
    f"Knowledge/filler-{i}.md": f"# Filler {i}\nGardening, cooking and hiking log {i}.\n"
    for i in range(12)
})

GRAPH_JSON = json.dumps({
    "entities": [
        {"name": "dopamine", "type": "biology"},
        {"name": "reward prediction error", "type": "concept"},
        {"name": "reinforcement learning", "type": "concept"},
        {"name": "attention mechanism", "type": "technique"},
    ],
    "relations": [
        {"source": "dopamine", "relation": "causes", "target": "reward prediction error"},
        {"source": "reward prediction error", "relation": "relates_to",
         "target": "reinforcement learning"},
        {"source": "reinforcement learning", "relation": "builds_on",
         "target": "attention mechanism"},
    ],
})


def _fake_vector(text: str, dims: int = 1024) -> list[float]:
    out: list[float] = []
    i = 0
    while len(out) < dims:
        digest = hashlib.sha256(f"{i}:{text[:64]}".encode()).digest()
        out.extend(b / 255 - 0.5 for b in digest)
        i += 1
    return out[:dims]


class _FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep test output quiet
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/embed":
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            resp = {"model": body["model"], "embeddings": [_fake_vector(t) for t in inputs]}
        elif self.path == "/api/embeddings":
            resp = {"embedding": _fake_vector(body["prompt"])}
        elif self.path == "/api/generate":
            text = GRAPH_JSON if "Extract entities" in body["prompt"] else "0.5 A short summary."
            resp = {"model": body["model"], "response": text, "done": True}
        else:
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _unused_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class VaultTestCase(unittest.TestCase):
    """Fresh vault + HOME per test class; helpers to run the scripts."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.vault = tmp / "vault"
        for rel, text in NOTES.items():
            path = self.vault / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.env = {
            **os.environ,
            "HOME": str(tmp / "home"),
            "VAULT_SEARCH_CACHE_DIR": str(tmp / "cache"),
            # Nothing listens here, so Ollama is "down" unless a test starts the fake
            "OLLAMA_BASE": f"http://127.0.0.1:{_unused_port()}",
        }
        for var in ("VAULT_SEARCH_DB", "VAULT_DIR"):
            self.env.pop(var, None)
        self.server = None

    def tearDown(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        self._tmp.cleanup()

    def start_ollama(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllama)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.env["OLLAMA_BASE"] = f"http://127.0.0.1:{self.server.server_address[1]}"

    def run_tool(self, script, *args, ok=True):
        proc = subprocess.run(
            [sys.executable, str(REPO / script), *map(str, args)],
            env=self.env, capture_output=True, text=True, timeout=120,
        )
        self.assertNotIn("Traceback", proc.stderr, proc.stderr)
        if ok:
            self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc

    def db_path(self) -> Path:
        dbs = list((Path(self.env["HOME"]) / ".local/share/vault-search").glob("*.db"))
        self.assertEqual(len(dbs), 1)
        return dbs[0]

    def search_json(self, *args):
        out = json.loads(self.run_tool("vault-search.py", *args, "--json").stdout)
        return out["results"] if isinstance(out, dict) else out


class TestWithoutOllama(VaultTestCase):
    """The README quick start: no Ollama anywhere."""

    def setUp(self):
        super().setUp()
        proc = self.run_tool("vault-index.py", self.vault, "--no-summary")
        self.assertIn("BM25-only", proc.stderr)

    def test_index_is_bm25_only(self):
        conn = sqlite3.connect(self.db_path())
        files, embedded, fts = conn.execute(
            "SELECT COUNT(*), COUNT(embedding), (SELECT COUNT(*) FROM files_fts) FROM files"
        ).fetchone()
        conn.close()
        self.assertEqual(files, len(NOTES))
        self.assertEqual(embedded, 0)
        self.assertEqual(fts, len(NOTES))

    def test_search_finds_notes_without_warning(self):
        proc = self.run_tool("vault-search.py", "temporal difference dopamine", self.vault)
        # Index has no embeddings, so search shouldn't try (and warn about) Ollama
        self.assertNotIn("Cannot reach Ollama", proc.stderr)
        self.assertIn("Knowledge/td-learning.md", proc.stdout)
        # No summaries: results show the note's title and a snippet instead
        self.assertIn("Temporal Difference Learning", proc.stdout)
        self.assertIn('"TD learning updates', proc.stdout)

    def test_json_has_title_and_snippet(self):
        results = self.search_json("phasic dopamine", self.vault)
        top = results[0]
        self.assertEqual(top["path"], "Knowledge/dopamine.md")
        self.assertEqual(top["title"], "Dopamine and Reward Prediction Error")  # frontmatter
        self.assertIn("phasic dopamine", top["snippet"])

    def test_bm25_mode_and_typed_subqueries(self):
        results = self.search_json("attention", self.vault, "--mode", "bm25")
        self.assertIn("Knowledge/attention.md", [r["path"] for r in results])
        results = self.search_json('lex:"temporal" vec:"reward" hyde:"what is TD"',
                                   self.vault, "--no-embeddings")
        self.assertEqual(results[0]["path"], "Knowledge/td-learning.md")

    def test_semantic_mode_explains_missing_embeddings(self):
        proc = self.run_tool("vault-search.py", "attention", self.vault,
                             "--mode", "semantic", ok=False)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("no embeddings", proc.stderr)

    def test_status(self):
        proc = self.run_tool("vault-search.py", "status", self.vault)
        self.assertIn(f"files indexed: {len(NOTES)}", proc.stdout)


class TestWithOllama(VaultTestCase):
    """Full pipeline against the fake Ollama server."""

    def test_bm25_index_upgrades_when_ollama_returns(self):
        self.run_tool("vault-index.py", self.vault, "--no-summary")
        self.start_ollama()
        # Searching a not-yet-embedded index with Ollama up must not crash
        self.run_tool("vault-search.py", "dopamine", self.vault)
        self.run_tool("vault-index.py", self.vault)
        conn = sqlite3.connect(self.db_path())
        missing = conn.execute(
            "SELECT COUNT(*) FROM files WHERE embedding IS NULL OR content_hash IS NULL"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(missing, 0)
        # Second run is incremental
        proc = self.run_tool("vault-index.py", self.vault)
        self.assertIn(f"({len(NOTES)} unchanged, 0 indexed)", proc.stdout)

    def test_hybrid_search_and_cache_dir(self):
        self.start_ollama()
        self.run_tool("vault-index.py", self.vault)
        for query in ("dopamine reward", 'lex:"temporal" vec:"reward" hyde:"what is TD"'):
            self.assertTrue(self.search_json(query, self.vault, "--expand"))
        cache = Path(self.env["VAULT_SEARCH_CACHE_DIR"])
        self.assertTrue(any(cache.glob("*.json")), "embedding cache not written")

    def test_graph_tools(self):
        self.start_ollama()
        self.run_tool("vault-index.py", self.vault)
        self.run_tool("vault-graph.py", "index", self.vault)

        proc = self.run_tool("vault-search.py", "reinforcement learning", self.vault)
        graph = proc.stdout.split("── Graph Context ──", 1)[1]
        edges = [l.strip() for l in graph.splitlines() if l.strip().startswith(("→", "←"))]
        self.assertTrue(edges)
        # Same fact extracted from several notes is shown once
        self.assertEqual(len(edges), len(set(edges)), graph)

        proc = self.run_tool("knowledge-path.py", "dopamine", "attention",
                             "--root", self.vault, "--include-leaves")
        self.assertIn("reward prediction error", proc.stdout)
        proc = self.run_tool("causal-trace.py", "dopamine", "reinforcement learning",
                             "--root", self.vault)
        self.assertIn("reward prediction error", proc.stdout)
        self.run_tool("synthesis-suggest.py", "--root", self.vault)
        self.run_tool("vault-search.py", "init", self.vault)
        proc = self.run_tool("vault-search.py", "dopamine", self.vault, "--fast")
        self.assertIn("dopamine", proc.stdout)

    @unittest.skipUnless(importlib.util.find_spec("igraph") and importlib.util.find_spec("leidenalg"),
                         "python-igraph + leidenalg not installed")
    def test_leiden_communities(self):
        self.run_tool("vault-index.py", self.vault, "--no-summary")
        proc = self.run_tool("leiden-communities.py", "--root", self.vault, "--stats-only", ok=False)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("No knowledge graph", proc.stderr)
        self.start_ollama()
        self.run_tool("vault-graph.py", "index", self.vault)
        self.run_tool("leiden-communities.py", "--root", self.vault, "--stats-only")
        proc = self.run_tool("leiden-communities.py", "--root", self.vault, "--query", "dopamine")
        self.assertIn("reward prediction error", proc.stdout)

    def test_root_flag_rejects_unindexed_vault(self):
        proc = self.run_tool("knowledge-path.py", "a", "b", "--root",
                             self.vault / "nope", ok=False)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("No vault-search database found", proc.stderr)


class TestSnippets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("vault_search", REPO / "vault-search.py")
        cls.vs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.vs)

    def test_heading_title_and_matching_line(self):
        title, snippet = self.vs._title_and_snippet(
            "# My Note\nFirst line.\nSecond line mentions dopamine here.\n", "dopamine")
        self.assertEqual(title, "My Note")
        self.assertEqual(snippet, "Second line mentions dopamine here.")

    def test_long_line_is_windowed_around_match(self):
        line = "x " * 200 + "needle" + " y" * 200
        _, snippet = self.vs._title_and_snippet(line, "needle", width=60)
        self.assertIn("needle", snippet)
        self.assertTrue(snippet.startswith("...") and snippet.endswith("..."))

    def test_typed_prefixes_are_not_search_terms(self):
        _, snippet = self.vs._title_and_snippet(
            "# N\nThe vec field is unrelated.\nTemporal difference learning.\n", 'vec:"temporal"')
        self.assertEqual(snippet, "Temporal difference learning.")

    def test_summary_detection(self):
        for s in ("", None, "(pending)", "(summary failed: HTTP 404)"):
            self.assertFalse(self.vs._has_real_summary(s))
        self.assertTrue(self.vs._has_real_summary("Explains TD learning."))


if __name__ == "__main__":
    unittest.main()
