# Hybrid Standalone App

This folder is a standalone root for the hybrid PDF + weather chatbot.

## Structure

- `hybrid.py`: core hybrid router + tools + CLI chatbot
- `hybrid_api.py`: FastAPI wrapper
- `static/index.html`: simple web UI
- `docs/`: place your PDF here as `transformers.pdf`
- `chroma_db/`: created automatically on first run

## 1) Setup

```bash
cd hybrid_standalone
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Configure env

```bash
cp .env.example .env
```

Then edit `.env` and set:

- `OPENAI_API_KEY`
- `OPENWEATHER_API_KEY`
- optional `PDF_PATH` (defaults to `docs/transformers.pdf`)

If your env file is stored elsewhere, pass it at runtime:

```bash
ENV_FILE=/absolute/path/to/.env ./run.sh
```

## 3) Add PDF

Put your PDF at:

- `docs/transformers.pdf`

## 4) Run CLI

```bash
python hybrid.py
```

## 5) Run API + UI

```bash
uvicorn hybrid_api:app --reload --host 0.0.0.0 --port 8000
```

Or start with the helper script:

```bash
./run.sh
```

Open:

- `http://127.0.0.1:8000/`


Command for Terminal:

Route decision only
curl -s -X POST "http://127.0.0.1:8000/route" \
  -H "Content-Type: application/json" \
  -d '{"question":"weather in cairo"}' | jq

Full answer (non-streaming)
curl -s -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the PDF say about transformers?"}' | jq

Streaming answer
curl -N -X POST "http://127.0.0.1:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"question":"Explain self-attention from the PDF"}'