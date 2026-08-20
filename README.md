# openai-proxy

A lightweight, OpenAI-compatible reverse proxy that routes requests to multiple upstream
backends (e.g. several vLLM instances). It is a single Python script with inline
[PEP 723](https://peps.python.org/pep-0723) dependencies, so it runs directly with
`uv run --script` (no install step):

```sh
uv run --script openai-proxy openai-proxy-config.yaml
```

The YAML config file (see [`openai-proxy-config.yaml`](openai-proxy-config.yaml) for a
working example) defines where to listen (`listen_host`/`listen_port`), the named
`backends` with their `base_url` and optional upstream `api_key`, and the `client_keys`
that clients authenticate with via `Authorization: Bearer <key>`.

Clients call the standard OpenAI endpoints — `/v1/models`, `/v1/chat/completions`,
`/v1/completions`, and `/v1/embeddings` — and each request is routed to the first allowed
backend that currently serves the requested model. Available models are discovered by
querying each backend's `/v1/models` (results are cached for 60 seconds). Every client
key can be restricted with `allowed_models` (glob patterns) and `allowed_backends`
(backend names), so different users/teams can be granted access to different models and
machines. Streaming (`stream: true`) and non-streaming responses are both supported, and
upstream errors are forwarded with their original status code.

For deployment: the proxy listens on `127.0.0.1` by default and is meant to be put behind
a reverse proxy (e.g. nginx, with `client_max_body_size` set to a similar or smaller
value than the proxy's 64 MiB limit). Requests are forwarded with the client's headers
(hop-by-hop headers and the client's `Authorization` are stripped; the real TCP peer
address is appended to `X-Forwarded-For`/`X-Real-IP`), and the built-in docs/OpenAPI
endpoints are disabled so the API surface is not disclosed to unauthenticated clients.
