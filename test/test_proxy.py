#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "httpx",
#   "fastapi",
#   "pydantic",
#   "PyYAML",
#   "typed_argparse",
#   "uvicorn",
#   "pytest",
# ]
# ///
"""Smoke tests for the OpenAI-compatible proxy (run against a built-in fake
vLLM backend).

The fake backend and the proxy under test are started on loopback once per
test session (see conftest.py). These tests then verify:

  - unauthenticated requests reach no backend at all
  - /docs, /redoc, and /openapi.json are not served
  - malformed Authorization headers yield 401/403, never 500
  - the client's API key is never forwarded to the backend
  - X-Forwarded-For / X-Real-IP reflect the direct peer, hop-by-hop and
    proxy headers are stripped
  - streaming and non-streaming responses (including upstream errors) are
    passed through with the correct status codes
  - model allow-listing and the request-body size limit are enforced

Usage:
    uv run --script test/test_proxy.py     # self-contained (uses uv)
    python -m pytest test/ -v              # or with a prepared interpreter

Extra options (registered in test/conftest.py):
    --proxy PATH       proxy script under test
                       (default: <this dir>/../openai-proxy)
    --backend-port N   fixed port for the fake backend (default: ephemeral)
    --proxy-port N     fixed port for the proxy (default: ephemeral)

If any test fails, the proxy's own log is printed after the summary.
"""
from __future__ import annotations

import socket
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import httpx

# Must match the keys configured for the proxy in conftest.py.
CLIENT_KEY = "test-client-key"
UPSTREAM_KEY = "UPSTREAM_KEY"


def raw_status(port: int, raw_request: bytes) -> str:
    """Send a raw HTTP request and return the response status line."""
    with socket.create_connection(("127.0.0.1", port), timeout=10) as s:
        s.sendall(raw_request)
        s.shutdown(socket.SHUT_WR)
        data = b""
        while chunk := s.recv(65536):
            data += chunk
    return data.split(b"\r\n", 1)[0].decode()


def test_unauthenticated_models_rejected(
    client: httpx.Client, backend_log: list[dict[str, str | None]],
) -> None:
    r = client.get("/v1/models")
    assert r.status_code == 401, r.text
    assert not backend_log, backend_log


def test_wrong_key_rejected(
    client: httpx.Client,
    backend_log: list[dict[str, str | None]],
) -> None:
    r = client.get("/v1/models",
                   headers={"Authorization": "Bearer wrong-key"})
    assert r.status_code == 403, r.text
    assert not backend_log, backend_log


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_docs_and_openapi_not_served(client: httpx.Client, path: str) -> None:
    r = client.get(path)
    assert r.status_code == 404, r.status_code


def test_bearer_without_key_rejected(proxy_port: int) -> None:
    # Sent via raw socket: the httpx client refuses to send such a header,
    # but a real attacker is not limited by that.
    status = raw_status(proxy_port, (
        b"GET /v1/models HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Authorization: Bearer \r\n"
        b"\r\n"))
    assert " 401 " in status, status


def test_non_ascii_key_rejected(proxy_port: int) -> None:
    status = raw_status(proxy_port, (
        b"GET /v1/models HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Authorization: Bearer B\xc3\xa9ar\xc3\xa9r\r\n"
        b"\r\n"))
    assert " 403 " in status, status


def test_unauthenticated_completion_rejected(
    client: httpx.Client, backend_log: list[dict[str, str | None]],
) -> None:
    r = client.post("/v1/chat/completions", json={"model": "model-a"})
    assert r.status_code == 401, r.text
    assert not backend_log, backend_log


def test_authenticated_completion_passthrough(
    client: httpx.Client, backend_log: list[dict[str, str | None]],
) -> None:
    r = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {CLIENT_KEY}",
            "X-Forwarded-For": "6.6.6.6",
            "X-Real-IP": "7.7.7.7",
            "Te": "trailers",
            "Proxy-Authorization": "Basic abc",
        },
        json={"model": "model-a"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == (
        "hello from backend"), r.text
    entry = backend_log[-1]
    assert entry["auth"] == f"Bearer {UPSTREAM_KEY}", entry
    assert entry["xff"] == "6.6.6.6, 127.0.0.1", entry
    assert entry["xri"] == "127.0.0.1", entry
    assert entry["te"] is None, entry
    assert entry["pauth"] is None, entry


def test_streaming_completion(client: httpx.Client) -> None:
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {CLIENT_KEY}"},
        json={"model": "model-a", "stream": True},
    )
    assert r.status_code == 200, r.text[:200]
    assert '{"i": 0}' in r.text and "data: [DONE]" in r.text, r.text[:200]


@pytest.mark.parametrize("stream", [False, True])
def test_backend_error_status_preserved(client: httpx.Client, stream: bool) -> None:
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {CLIENT_KEY}"},
        json={"model": "model-unknown-xyz", "stream": stream},
    )
    assert r.status_code == 404, f"{r.status_code} {r.text}"
    assert "not found" in r.text, r.text


def test_unknown_model_rejected(
    client: httpx.Client, backend_log: list[dict[str, str | None]],
) -> None:
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {CLIENT_KEY}"},
        json={"model": "model-b"},
    )
    assert r.status_code == 404, f"{r.status_code} {r.text}"
    assert "Model not found" in r.text, r.text
    assert not backend_log, backend_log


def test_non_object_json_body_rejected(client: httpx.Client) -> None:
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {CLIENT_KEY}"},
        content=b"[]",
    )
    assert r.status_code == 400, f"{r.status_code} {r.text}"


def test_disallowed_model_rejected(
    client: httpx.Client, backend_log: list[dict[str, str | None]],
) -> None:
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {CLIENT_KEY}"},
        json={"model": "other-model"},
    )
    assert r.status_code == 403, f"{r.status_code} {r.text}"
    assert not backend_log, backend_log


def test_model_list_filtered_per_client(client: httpx.Client) -> None:
    r = client.get(
        "/v1/models", headers={"Authorization": f"Bearer {CLIENT_KEY}"})
    ids = [m.get("id") for m in r.json().get("data", [])]
    assert ids == ["model-a"], ids


def test_oversized_content_length_rejected(proxy_port: int) -> None:
    auth_line = f"Authorization: Bearer {CLIENT_KEY}\r\n".encode()
    status = raw_status(proxy_port, (
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        + auth_line
        + b"Content-Length: 999999999\r\n"
        b"\r\n"))
    assert " 413 " in status, status


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
