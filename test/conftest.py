"""Fixtures for the OpenAI-compatible proxy smoke tests (test_proxy.py).

A fake vLLM-like backend and the proxy under test are each started once per
test session on loopback. The fake backend records every request it receives
in ``backend_log`` (cleared before each test), so individual tests can assert
on what, if anything, the proxy forwarded upstream.
"""
import http.server
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

# Must match the Authorization headers used in test_proxy.py.
CLIENT_KEY = "test-client-key"
UPSTREAM_KEY = "UPSTREAM_KEY"

_proxy_output = ""


def pytest_addoption(parser):
    parser.addoption(
        "--proxy",
        default=str(Path(__file__).resolve().parent.parent / "openai-proxy"),
        help="path to the proxy script under test")
    parser.addoption(
        "--backend-port", type=int, default=0,
        help="fixed port for the fake backend (default: ephemeral)")
    parser.addoption(
        "--proxy-port", type=int, default=0,
        help="fixed port for the proxy (default: ephemeral)")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # Print the proxy's own log after the summary if anything failed, so
    # proxy crashes stay diagnosable (as in the old single-script version).
    if exitstatus and _proxy_output:
        width = 70
        terminalreporter.write_line("=" * width)
        terminalreporter.write_line("proxy output".center(width))
        terminalreporter.write_line("=" * width)
        terminalreporter.write(_proxy_output)


def _loopback_env() -> dict[str, str]:
    """The current environment without any ambient HTTP proxy settings, so
    the proxy under test talks directly to the loopback fake backend.
    """
    return {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}


def free_port() -> int:
    """Ask the OS for a free loopback port (best effort)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_backend(port: int, log: list[dict]) -> http.server.ThreadingHTTPServer:
    """A minimal stand-in for a vLLM server that logs every request it sees."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            log.append({"method": "GET", "path": self.path})
            if self.path == "/v1/models":
                self._send(200, {"object": "list", "data": [
                    {"id": "model-a", "object": "model"},
                    {"id": "other-model", "object": "model"},
                ]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n).decode()
            log.append({
                "method": "POST",
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "xff": self.headers.get("X-Forwarded-For"),
                "xri": self.headers.get("X-Real-IP"),
                "te": self.headers.get("Te"),
                "pauth": self.headers.get("Proxy-Authorization"),
                "body": body,
            })
            data = json.loads(body)
            model = data.get("model")
            if model == "model-a":
                if data.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for i in range(3):
                        self.wfile.write(f'{{"i": {i}}}'.encode())
                        self.wfile.write(b"\n\n")
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    self._send(200, {
                        "id": "c1",
                        "object": "chat.completion",
                        "choices": [
                            {"message": {"content": "hello from backend"}},
                        ],
                    })
            else:
                self._send(
                    404, {"error": {"message": f"model '{model}' not found"}})

    return http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)


@pytest.fixture(scope="session")
def backend_log() -> list[dict]:
    """Requests observed by the fake backend, cleared before each test."""
    return []


@pytest.fixture(autouse=True)
def _fresh_backend_log(backend_log: list[dict]) -> None:
    backend_log.clear()


@pytest.fixture(scope="session")
def backend_port(
    backend_log: list[dict],
    request: pytest.FixtureRequest,
) -> Iterator[int]:
    """Start the fake backend and yield the port it listens on."""
    port = request.config.getoption("--backend-port") or free_port()
    server = make_backend(port, backend_log)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()
    server.server_close()


@pytest.fixture(scope="session")
def proxy_port(
    backend_port: int,
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> Iterator[int]:
    """Start the proxy under test and yield the port it listens on."""
    proxy = request.config.getoption("--proxy")
    port = request.config.getoption("--proxy-port") or free_port()

    config_file = tmp_path_factory.mktemp("openai-proxy-test") / "test-proxy.yaml"
    config_file.write_text(f"""listen_host: "127.0.0.1"
listen_port: {port}
backends:
  be1:
    base_url: http://127.0.0.1:{backend_port}
    api_key: {UPSTREAM_KEY}
client_keys:
  {CLIENT_KEY}:
    allowed_models:
    - "model*"
""")

    # sys.executable guarantees the proxy gets the same dependencies as
    # this test (relevant when this file is run via `uv run`).
    proc = subprocess.Popen(
        [sys.executable, proxy, str(config_file)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=_loopback_env())

    # Wait for the proxy to come up. Unauthenticated requests must be
    # rejected by the proxy itself, so 401/403 means it is ready.
    up = False
    with httpx.Client(timeout=10, trust_env=False) as client:
        for _ in range(50):
            if proc.poll() is not None:
                break
            try:
                r = client.get(f"http://127.0.0.1:{port}/v1/models")
            except httpx.HTTPError:
                time.sleep(0.2)
                continue
            if r.status_code in (401, 403):
                up = True
                break
            time.sleep(0.2)
    if not up:
        proc.terminate()
        out, _ = proc.communicate(timeout=10)
        raise RuntimeError(f"proxy did not come up:\n{out}")

    yield port

    global _proxy_output
    proc.terminate()
    try:
        out, _ = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    _proxy_output = out


@pytest.fixture(scope="session")
def proxy_base_url(proxy_port: int) -> str:
    return f"http://127.0.0.1:{proxy_port}"


@pytest.fixture
def client(proxy_base_url: str) -> Iterator[httpx.Client]:
    """A fresh httpx client (per test) already pointed at the proxy."""
    with httpx.Client(base_url=proxy_base_url, timeout=10,
                      trust_env=False) as client:
        yield client
