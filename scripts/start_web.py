#!/usr/bin/env python3
"""
Start PathOptLearn web stack (DeepTutor-like: backend + frontend).

Usage:
  python scripts/start_web.py          # Start backend (8001) + Gradio (7860)
  python scripts/start_web.py --api    # Backend only
  python scripts/start_web.py --ui     # Gradio only (in-process model)
"""

import argparse
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Start PathOptLearn web services")
    parser.add_argument("--api", action="store_true", help="Start backend API only")
    parser.add_argument("--ui", action="store_true", help="Start Gradio UI only (no API)")
    parser.add_argument("--port-api", type=int, default=None, help="Backend port (default: 8001)")
    parser.add_argument("--port-ui", type=int, default=None, help="Gradio port (default: 7860)")
    args = parser.parse_args()

    port_api = args.port_api or int(os.environ.get("BACKEND_PORT", "8001"))
    port_ui = args.port_ui or int(os.environ.get("FRONTEND_PORT", "7860"))

    if args.api:
        print(f"Starting backend at http://0.0.0.0:{port_api}")
        os.execv(sys.executable, [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", str(port_api)])
        return

    if args.ui:
        print(f"Starting Gradio UI at http://0.0.0.0:{port_ui}")
        import app
        app.init_model()
        demo = app.build_ui(app.init_model())
        demo.launch(server_name="0.0.0.0", server_port=port_ui)
        return

    # Both: start API in subprocess, then Gradio with API_BASE
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    os.environ["API_BASE"] = f"http://127.0.0.1:{port_api}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", str(port_api)],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    time.sleep(2)
    if proc.poll() is not None:
        print("Backend failed to start")
        sys.exit(1)
    print(f"Backend running at http://0.0.0.0:{port_api}")
    print(f"Starting Gradio UI at http://0.0.0.0:{port_ui} (uses API)")
    from app import build_ui
    demo = build_ui(initial_status=f"Using API at {os.environ['API_BASE']}. Backend ready.")
    demo.launch(server_name="0.0.0.0", server_port=port_ui)


if __name__ == "__main__":
    main()
