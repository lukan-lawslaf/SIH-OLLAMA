# SIH-OLLAMA — Server Side (Laptop B)

FastAPI gateway + Ollama for the SIH 26171 Aegis-Agent extension.
**Read `client_context.md` first** — it is the full client/server contract.

## Quick start (Windows PowerShell)

```powershell
# 1. Ollama daemon (install from https://ollama.com if missing)
ollama serve   # or start the Ollama app

# 2. Pull models (2b/4b local; 235b-cloud needs `ollama signin` first)
ollama pull qwen3-vl:4b
ollama pull qwen3-vl:2b
ollama signin
ollama pull qwen3-vl:235b-cloud   # optional, test-only

# 3. Gateway
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item ..\.env.example .env
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Verify

```powershell
# same laptop
python backend/smoke_test.py

# from the client laptop (replace IP)
curl http://<THIS-LAPTOP-IP>:8000/health
```

If the client laptop cannot reach it: allow port 8000 through Windows Firewall
(Settings → Network & Internet → Advanced security → Inbound rule, TCP 8000),
and make sure both laptops are on the same hotspot/LAN.

## Connect the extension (client laptop)

In the **nanobrowser repo root** create/edit `.env`:

```text
VITE_SIH_FASTAPI_URL=http://<server-laptop-ip>:8000/v1
VITE_SIH_QWEN_MODEL=qwen3-vl:4b
```

Then `pnpm build`, reload the unpacked extension from `dist/`.

## Switching models

The gateway honors any model id listed in `QWEN_ALLOWED_MODELS` (see `.env.example`).
To test a different model on the client: set `VITE_SIH_QWEN_MODEL=qwen3-vl:2b`
(or `qwen3-vl:235b-cloud`), rebuild, reload. `qwen3-vl:235b-cloud` requires
`ollama signin` on this laptop and runs via Ollama's cloud — test model only.

## Repo layout

```text
backend/
  app.py          # FastAPI gateway (OpenAI-compatible proxy to Ollama)
  requirements.txt
  smoke_test.py   # health/models/chat smoke checks
client_context.md # full client↔server contract (read this first)
.env.example
```

## Rules for anyone (human or AI) editing this repo

- Never log message content, images, or credentials — metadata only.
- Keep the client-facing API byte-compatible: `/health`, `/v1/models`,
  `/v1/chat/completions` (streaming + multimodal + response_format).
- Model allow-list lives in `QWEN_ALLOWED_MODELS` (`.env`); unknown ids remap
  to `QWEN_MODEL`, never 404.
- Provider selection stays server-side (`QWEN_BACKEND=ollama|openai`).
