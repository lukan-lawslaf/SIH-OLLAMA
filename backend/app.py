"""SIH 26171 — local OpenAI-compatible gateway for the Aegis-Agent extension.

The browser extension talks to this service only. It exposes the
OpenAI-compatible surface the extension's LangChain client expects and
forwards to a local Ollama daemon (or any OpenAI-compatible server).

Client-facing API (what the extension requires):
  GET  /health                -> { status, backend, model }         (settings ping)
  GET  /v1/models             -> OpenAI-style model list
  POST /v1/chat/completions   -> OpenAI-style chat completion (streaming supported)

Privacy invariant: the extension sends only sanitized payloads. This gateway
never logs message content, image data, or credentials.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

load_dotenv()

BACKEND = os.getenv("QWEN_BACKEND", "ollama").lower()
DEFAULT_MODEL = os.getenv("QWEN_MODEL", "qwen3-vl:4b")
# The client may request any of these; anything else falls back to DEFAULT_MODEL.
# Primary demo models: qwen3-vl 2b/4b (local). Test matrix: qwen3.5 2b/4b,
# qwen3-vl:235b-cloud and gemma4:31b-cloud (Ollama-cloud, need `ollama signin`).
# Edit QWEN_ALLOWED_MODELS in .env to add/remove — no code change needed.
ALLOWED_MODELS = {
    model.strip()
    for model in os.getenv(
        "QWEN_ALLOWED_MODELS",
        "qwen3-vl:2b,qwen3-vl:4b,qwen3-vl:235b-cloud,qwen3.5:2b,qwen3.5:4b,gemma4:31b-cloud",
    ).split(",")
    if model.strip()
}
# True plug-and-play: forward ANY client-requested model id to the backend
# (LAN demo convenience; keep false if you want a strict allow-list).
ALLOW_ANY_MODEL = os.getenv("ALLOW_ANY_MODEL", "false").lower() in ("1", "true", "yes")
# Thinking-capable models (gemma4, qwen3.5, ...) emit reasoning tokens that count
# against the client's max_tokens budget — the planner's structured JSON arrives
# empty or truncated (finish_reason "length") while simpler executor calls
# survive. Suppress reasoning unless the client sets its own effort level.
# Values: any reasoning_effort Ollama accepts ("none" disables thinking);
# "passthrough" leaves the request untouched.
REASONING_EFFORT = os.getenv("OLLAMA_REASONING_EFFORT", "none").strip().lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "local")
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

SHORT_TIMEOUT = 5.0  # health/models probes
CHAT_TIMEOUT = None  # model inference can be slow; no cap while streaming


app = FastAPI(title="SIH Local Qwen Gateway", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def resolve_model(requested: Any) -> str:
    """Trust the client's model when allowed, else use the default."""
    if isinstance(requested, str):
        candidate = requested.strip()
        if candidate and (ALLOW_ANY_MODEL or candidate in ALLOWED_MODELS):
            return candidate
    return DEFAULT_MODEL


def strip_json_fences(text: str) -> str:
    """Remove the ```json fence some models (notably gemma4) wrap around
    structured output even when response_format asked for raw JSON — strict
    client parsers fail on the fence. No-op for already-raw JSON."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        s = s.strip()
        if s.endswith("```"):
            s = s[: s.rfind("```")].rstrip()
    return s


def clean_structured_body(body: bytes) -> bytes:
    """Fence-strip a non-streaming chat completion when the client asked for
    a structured response. Passthrough on any parse surprise."""
    try:
        data = json.loads(body)
        message = data["choices"][0]["message"]
        content = message["content"]
        if isinstance(content, str) and content.lstrip().startswith("```"):
            message["content"] = strip_json_fences(content)
            return json.dumps(data).encode()
    except (ValueError, KeyError, IndexError, TypeError):
        pass
    return body


@app.get("/health")
async def health() -> dict[str, Any]:
    """Return configuration and Ollama reachability without exposing credentials."""
    result: dict[str, Any] = {
        "status": "ok",
        "backend": BACKEND,
        "model": DEFAULT_MODEL,
        "allowed_models": sorted(ALLOWED_MODELS),
    }
    if BACKEND == "ollama":
        try:
            async with httpx.AsyncClient(timeout=SHORT_TIMEOUT) as client:
                probe = await client.get(f"{OLLAMA_BASE_URL}/api/version")
            result["ollama"] = probe.json() if probe.status_code < 400 else {"error": str(probe.status_code)}
        except httpx.HTTPError as exc:
            # Gateway is up, but the model daemon is not — say so explicitly.
            result["status"] = "degraded"
            result["ollama"] = {"error": f"unreachable: {exc}"}
    return result


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    if BACKEND == "ollama":
        try:
            async with httpx.AsyncClient(timeout=SHORT_TIMEOUT) as client:
                response = await client.get(f"{OLLAMA_BASE_URL}/v1/models")
            if response.status_code < 400:
                return response.json()
        except httpx.HTTPError:
            pass  # fall through to the static allow-list
        return {
            "object": "list",
            "data": [{"id": m, "object": "model", "owned_by": "ollama"} for m in sorted(ALLOWED_MODELS)],
        }
    return await forward("GET", "/models")


@app.api_route("/v1/chat/completions", methods=["POST"])
async def chat_completions(request: Request) -> Response:
    payload = await request.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        raise HTTPException(status_code=400, detail="Request must include a messages array")
    payload["model"] = resolve_model(payload.get("model"))

    if BACKEND == "ollama":
        return await ollama_chat(payload)
    return await forward("POST", "/chat/completions", payload)


async def ollama_chat(payload: dict[str, Any]) -> Response:
    # Ollama's OpenAI-compatible route preserves messages (including image
    # content parts) and supports streaming and response_format passthrough.
    if REASONING_EFFORT != "passthrough" and "reasoning_effort" not in payload:
        payload["reasoning_effort"] = REASONING_EFFORT
    target = f"{OLLAMA_BASE_URL}/v1/chat/completions"
    try:
        client = httpx.AsyncClient(timeout=CHAT_TIMEOUT)
        upstream = await client.send(
            client.build_request("POST", target, json=payload, headers={"content-type": "application/json"}),
            stream=bool(payload.get("stream")),
        )
    except httpx.HTTPError as exc:
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"Could not reach Ollama: {exc}") from exc

    if upstream.status_code >= 400:
        body = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        return Response(content=body, status_code=upstream.status_code, media_type="application/json")

    if payload.get("stream"):
        if payload.get("response_format"):
            # Structured output: buffer, strip fences, re-emit (see restream_structured).
            return await restream_structured(client, upstream)

        async def body():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()

        return StreamingResponse(body(), status_code=upstream.status_code, media_type="text/event-stream")

    body = await upstream.aread()
    await upstream.aclose()
    await client.aclose()
    if payload.get("response_format"):
        body = clean_structured_body(body)
    return Response(content=body, status_code=upstream.status_code, media_type="application/json")


async def restream_structured(client: httpx.AsyncClient, upstream: httpx.Response) -> StreamingResponse:
    """Buffer a structured-output SSE stream, fence-strip the assembled JSON,
    and re-emit it as chunks. Structured planner JSON is short and parsed as a
    whole by the client, so this is equivalent to streaming for the client
    while guaranteeing its parser never sees a ```json fence."""
    raw = bytearray()
    async for chunk in upstream.aiter_bytes():
        raw.extend(chunk)
    await upstream.aclose()
    await client.aclose()

    content_parts: list[str] = []
    final: dict[str, Any] | None = None
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        choices = obj.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            if isinstance(delta.get("content"), str):
                content_parts.append(delta["content"])
            if choices[0].get("finish_reason"):
                final = obj
    content = strip_json_fences("".join(content_parts))

    base = final or {"id": "chatcmpl", "object": "chat.completion.chunk", "model": "ollama"}
    chunks = [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}]},
    ]
    if final is not None:
        chunks.append({**final, "choices": [{"index": 0, "delta": {}, "finish_reason": final["choices"][0].get("finish_reason")}]})
    stream_bytes = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"

    async def body():
        yield stream_bytes

    return StreamingResponse(body(), status_code=200, media_type="text/event-stream")


async def forward(method: str, path: str, payload: dict[str, Any] | None = None) -> Response | dict[str, Any]:
    headers = {"Authorization": f"Bearer {QWEN_API_KEY}"}
    try:
        async with httpx.AsyncClient(timeout=CHAT_TIMEOUT) as client:
            response = await client.request(method, f"{QWEN_BASE_URL}{path}", json=payload, headers=headers)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach Qwen server: {exc}") from exc
    if response.status_code >= 400:
        return Response(content=response.content, status_code=response.status_code, media_type="application/json")
    if method == "GET":
        return response.json()
    return Response(content=response.content, status_code=response.status_code, media_type="application/json")
