"""
BTC Prediction API — Startup Script

Usage:
    python run_api.py
    python run_api.py --host 0.0.0.0 --port 8000
    python run_api.py --reload   (development mode)
"""
import argparse
import sys
from pathlib import Path

# Add src/ folder to Python path
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="BTC Prediction FastAPI Server")
    parser.add_argument("--host",   default="0.0.0.0",  help="Host (default: 0.0.0.0)")
    parser.add_argument("--port",   default=8000, type=int, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true",    help="Development mode (hot-reload)")
    args = parser.parse_args()

    print(f"""
==========================================
  BTC Prediction API is starting
==========================================
  Host    : {args.host}
  Port    : {args.port}
  Swagger : http://localhost:{args.port}/docs
  Reload  : {'Active' if args.reload else 'Disabled'}
==========================================
""")

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        app_dir=str(SRC_DIR),
        log_level="info",
    )


if __name__ == "__main__":
    main()
