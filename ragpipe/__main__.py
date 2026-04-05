"""CLI entry point: python -m ragpipe serve --port 8000 --config pipeline.yml"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="ragpipe", description="ragpipe CLI")
    sub = parser.add_subparsers(dest="command")

    serve_parser = sub.add_parser("serve", help="Start the ragpipe REST API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    serve_parser.add_argument("--config", default=None, help="Path to pipeline YAML config file")
    serve_parser.add_argument("--api-key", default=None, help="API key for authentication")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    if args.command == "serve":
        _serve(args)
    else:
        parser.print_help()
        sys.exit(1)


def _serve(args):
    try:
        import uvicorn
    except ImportError:
        print("Install server dependencies: pip install 'ragpipe[server]'")
        sys.exit(1)

    pipeline = None

    if args.config:
        from ragpipe.config import PipelineConfig
        pipeline = PipelineConfig.from_yaml(args.config).build()

    from ragpipe.server.app import create_app
    app = create_app(pipeline=pipeline, api_key=args.api_key)

    print(f"Starting ragpipe server on {args.host}:{args.port}")
    if not pipeline:
        print("No pipeline config provided — set one via POST /ingest or restart with --config")

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
