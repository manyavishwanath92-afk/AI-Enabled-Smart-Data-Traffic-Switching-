# Switchboard Dashboard

A responsive React/Vite dashboard for the AI-Enabled Smart Data Traffic Switching simulator. The frontend uses the existing FastAPI contract from `docs/FRONTEND_API_HANDOFF.md` and does not reproduce simulator calculations.

## Run locally

From the repository root, start the API server:

```bash
pip install -e '.[api]'
python -m traffic_sim.api
```

The standard Uvicorn WebSocket runtime is required for the live route:

```bash
pip install 'uvicorn[standard]'
```

In a second terminal:

```bash
cd frontend
pnpm install
pnpm dev
```

The Vite development server proxies `/api` and WebSocket upgrades to `http://127.0.0.1:8000`. To point at another deployment, set `VITE_API_BASE_URL` and `VITE_WS_BASE_URL`.

## What is connected

The dashboard starts, pauses, resumes, and stops real simulations; receives one `tick` payload at a time over `WS /api/simulations/{id}/live`; refreshes the slowly-changing UE population through `GET /ues`; renders backend KPI records, cells, groups, decisions, and persisted history; and runs the synchronous matched-seed `/api/comparison` endpoint. Empty states represent the absence of a real backend run rather than fabricated values.
