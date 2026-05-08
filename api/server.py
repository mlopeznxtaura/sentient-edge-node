"""
api/server.py — FastAPI app entry point (Cluster 01: Sentient Edge Node)

Wires routes, starts uvicorn. Designed to run alongside the agent loop.
SDKs: FastAPI, uvicorn
"""
import logging
import json
import threading
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router, set_node

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentient Edge Node API",
    description="Control and observe the edge node perception-decision-action loop.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


def start_api_server(config: dict, node=None) -> threading.Thread:
    """Start FastAPI in a background daemon thread alongside the agent loop."""
    if node:
        set_node(node)

    api_cfg = config.get("api", {})
    host = api_cfg.get("host", "0.0.0.0")
    port = api_cfg.get("port", 8000)

    def _run():
        logger.info("API server starting at http://%s:%d", host, port)
        uvicorn.run(app, host=host, port=port, log_level="warning")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    logger.info("API server thread started")
    return t


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    uvicorn.run(app, host=cfg.get("api", {}).get("host", "0.0.0.0"),
                port=cfg.get("api", {}).get("port", 8000))
