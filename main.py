"""
main.py — Sentient Edge Node entrypoint (Cluster 01)

Wires all modules: perception → agent → actuators → API → telemetry.
Run with: python main.py --config config.json
"""
import json
import logging
import argparse
import signal

from agent.loop import EdgeNodeLoop
from api.server import start_api_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Sentient Edge Node — Cluster 01")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    logger.info("Starting Sentient Edge Node | node_id=%s", config.get("node_id", "unknown"))

    node = EdgeNodeLoop(config)

    # Start REST API alongside the loop
    start_api_server(config, node=node)

    def _shutdown(sig, frame):
        logger.info("Shutdown signal received")
        node.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    node.start()


if __name__ == "__main__":
    main()
