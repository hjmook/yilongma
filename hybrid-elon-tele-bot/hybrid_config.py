"""
Hybrid configuration for Telegram bot modes.

Modes:
- elon-fast: uses the legacy Flask server in `model_server.py` (faster)
- elon-detailed: uses the dual-model HTTP server in this folder (slower, more accurate)
"""

import os


# Expose a single environment-driven selection for mode
MODE = os.environ.get("ELON_MODE", "elon-fast")  # "elon-fast" or "elon-detailed"


# Legacy fast server (already exists)
FAST_SERVER_URL = os.environ.get("FAST_SERVER_URL", "http://localhost:5001/predict")


# Detailed server (new, started by hybrid launcher)
DETAILED_SERVER_URL = os.environ.get("DETAILED_SERVER_URL", "http://localhost:5055/predict")


def get_model_endpoint_url() -> str:
    if MODE == "elon-detailed":
        return DETAILED_SERVER_URL
    return FAST_SERVER_URL


