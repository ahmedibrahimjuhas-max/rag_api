#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ENV_PATH="${ENV_FILE:-$ROOT_DIR/.env}"
if [[ ! -f "$ENV_PATH" ]]; then
  echo "Missing env file at: $ENV_PATH"
  echo "Either create local .env (cp .env.example .env) or set ENV_FILE=/absolute/path/.env"
  exit 1
fi
export ENV_FILE="$ENV_PATH"

if [[ ! -f "docs/transformers.pdf" ]]; then
  echo "Missing PDF at docs/transformers.pdf"
  echo "Add a PDF there or set PDF_PATH in .env"
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  echo "Missing virtual environment .venv"
  echo "Create it with: python -m venv .venv"
  exit 1
fi

source .venv/bin/activate

if ! python -c "import fastapi, uvicorn" >/dev/null 2>&1; then
  echo "Dependencies not installed. Run: pip install -r requirements.txt"
  exit 1
fi

exec uvicorn hybrid_api:app --reload --host 0.0.0.0 --port 8000
