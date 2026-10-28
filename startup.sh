#!/usr/bin/env bash
# Start command for Azure App Service (Linux, Python): build the demo match on first start, then serve.
set -euo pipefail
cd "$(dirname "$0")"
python -m server.prepare --if-missing
exec python -m uvicorn server.app:app --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers
