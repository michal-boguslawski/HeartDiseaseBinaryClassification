import requests
from typing import Any


class APICalls:
    def __init__(self, api_uri: str):
        self.api_uri = api_uri

    def _send_request(self, endpoint: str, data: dict | None = None) -> dict:
        if data is None:
            response = requests.get(f"{self.api_uri}/{endpoint}")
        else:
            response = requests.post(f"{self.api_uri}/{endpoint}", json=data)
        return response.json()

    def feature_names(self) -> dict:
        return self._send_request("feature_names")

    def predict(self, data: dict) -> dict:
        return self._send_request("predict", data)

    def predict_proba(self, data: dict) -> dict:
        return self._send_request("predict_proba", data)
