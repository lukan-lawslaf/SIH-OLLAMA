"""Smoke tests for the SIH gateway. Run with the server up:

    python smoke_test.py [base_url]

Checks health, model listing, model pass-through, and a tiny chat completion.
Exits non-zero on the first failure.
"""

from __future__ import annotations

import sys

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        sys.exit(1)


def main() -> None:
    with httpx.Client(timeout=120) as client:
        health = client.get(f"{BASE}/health")
        check("health 200", health.status_code == 200, str(health.json()))
        check("health reports ollama", "ollama" in health.json())

        models = client.get(f"{BASE}/v1/models")
        ids = [m.get("id") for m in models.json().get("data", [])]
        check("models 200", models.status_code == 200, ", ".join(str(i) for i in ids))
        check("2b/4b present", "qwen3-vl:2b" in ids and "qwen3-vl:4b" in ids)

        chat = client.post(
            f"{BASE}/v1/chat/completions",
            json={
                "model": "qwen3-vl:2b",
                "messages": [{"role": "user", "content": "Reply with exactly: gateway-ok"}],
                "max_tokens": 20,
                "temperature": 0,
            },
        )
        ok = chat.status_code == 200 and chat.json().get("choices")
        text = chat.json().get("choices", [{}])[0].get("message", {}).get("content", "") if ok else chat.text[:200]
        check("chat completion 2b", bool(ok), str(text)[:120])

        unknown = client.post(
            f"{BASE}/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 10,
            },
        )
        # Unknown model must be remapped to the default, not forwarded raw.
        check("unknown model remapped", unknown.status_code != 404, unknown.text[:120])

    print("all smoke tests passed")


if __name__ == "__main__":
    main()
