import logging
import os
import sys
import requests

# Support running as script or module
try:
    from .hybrid_config import get_model_endpoint_url, MODE
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hybrid_config import get_model_endpoint_url, MODE


logger = logging.getLogger(__name__)


class HybridModelClient:
    """Client that routes to fast or thinking server based on MODE (single-user)."""

    def __init__(self):
        # Default URL can be overridden after mode selection via set_endpoint
        self.fast_url = os.environ.get("FAST_SERVER_URL", "http://localhost:5001/predict")
        self.thinking_url = os.environ.get("THINKING_SERVER_URL", "http://localhost:5055/predict")
        self._current_url = os.environ.get("HYBRID_MODEL_URL", self.thinking_url)
        logger.info(f"HybridModelClient mode={MODE} endpoint={self._current_url}")

    def get_url(self) -> str:
        return self._current_url

    def set_endpoint(self, url: str):
        self._current_url = url
        logger.info(f"HybridModelClient endpoint set to {url}")

    def send_request(self, input_data: dict) -> dict:
        try:
            url = self.get_url()
            response = requests.post(url, json=input_data, timeout=90)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error("Request to model timed out")
            return {"error": "Model request timed out."}
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to model at {url}")
            return {"error": "Could not connect to model server. Is it running?"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with the model: {e}")
            return {"error": str(e)}

    def get_response(self, user_input: str, user_id: str = "telegram_user", use_rag: bool = True) -> str:
        input_data = {"input": user_input, "user_id": user_id, "use_rag": use_rag}
        data = self.send_request(input_data)
        if "error" in data:
            return f"Sorry, I encountered an error: {data['error']}"
        return data.get("output", "No response from model.")

    def reset_history(self, user_id: str = "telegram_user") -> bool:
        try:
            url = self.get_url().replace('/predict', '/reset')
            r = requests.post(url, json={"user_id": user_id}, timeout=10)
            r.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error resetting history: {e}")
            return False

    def health_check(self) -> dict:
        try:
            url = self.get_url().replace('/predict', '/health')
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}


