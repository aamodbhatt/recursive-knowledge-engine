#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1"
    exit 1
  fi
}

require_cmd python3
require_cmd npm

hash_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
    return
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
    return
  fi
  echo "Missing hash tool (shasum/sha256sum)."
  exit 1
}

if [ ! -f ".env" ]; then
  cp ".env.example" ".env"
  echo "Created .env from .env.example."
  echo "Set LLM_API_KEY in .env before production use."
fi

if [ ! -d ".venv" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv .venv
fi

REQ_HASH="$(hash_file requirements.txt)"
REQ_STAMP=".venv/.requirements.sha256"
if [ ! -f "${REQ_STAMP}" ] || [ "$(cat "${REQ_STAMP}")" != "${REQ_HASH}" ]; then
  echo "Installing backend dependencies..."
  ./.venv/bin/python -m pip install --disable-pip-version-check -q --upgrade pip
  ./.venv/bin/pip install --disable-pip-version-check -q -r requirements.txt
  echo "${REQ_HASH}" > "${REQ_STAMP}"
fi

PKG_LOCK="frontend/package-lock.json"
if [ -f "${PKG_LOCK}" ]; then
  FRONT_HASH="$(hash_file "${PKG_LOCK}")"
else
  FRONT_HASH="$(hash_file frontend/package.json)"
fi
FRONT_STAMP="frontend/.frontend.sha256"
if [ ! -d "frontend/node_modules" ] || [ ! -f "${FRONT_STAMP}" ] || [ "$(cat "${FRONT_STAMP}")" != "${FRONT_HASH}" ]; then
  echo "Installing frontend dependencies..."
  (cd frontend && npm install --silent)
  echo "${FRONT_HASH}" > "${FRONT_STAMP}"
fi

echo "Building frontend..."
(cd frontend && npm run build --silent)

echo "Starting app at http://127.0.0.1:${PORT}"
exec ./.venv/bin/uvicorn app.main:app --host "${HOST}" --port "${PORT}"
