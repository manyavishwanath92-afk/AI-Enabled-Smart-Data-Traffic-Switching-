"""Start the dashboard API server.

    python -m traffic_sim.api
    python -m traffic_sim.api --host 0.0.0.0 --port 8080 --reload
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the simulator dashboard API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on source changes (development).")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on the install extra
        raise SystemExit(
            "uvicorn is not installed. Install the API extra:\n    pip install -e '.[api]'"
        ) from exc

    print(f"API docs:  http://{args.host}:{args.port}/docs")
    uvicorn.run("traffic_sim.api.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
