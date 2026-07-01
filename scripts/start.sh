#!/usr/bin/env bash
set -e

# Seed demo data (no-op if runs already exist), then launch the dashboard.
python scripts/seed_demo.py || echo "[start] seed skipped/failed (continuing)"

exec streamlit run app/dashboard/main.py \
  --server.port "${PORT:-7860}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
