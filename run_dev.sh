#!/usr/bin/env bash
# Local development: API on :8000, React dev server on :5173 (proxies /api to the API).
set -e
cd "$(dirname "$0")"
python3 -m uvicorn gve.api:app --reload --port 8000 &
API=$!
trap "kill $API" EXIT
cd web && [ -d node_modules ] || npm install
npm run dev
