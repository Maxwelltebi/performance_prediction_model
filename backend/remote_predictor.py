"""Call our custom Hugging Face Inference Endpoint without loading ML libraries."""

from urllib.parse import urlsplit

import httpx

from schemas import PredictionResponse


class RemotePredictionUnavailable(RuntimeError):
    pass


class RemoteGradePredictor:
    def __init__(self, url: str, token: str) -> None:
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or not parsed.hostname
                or not parsed.hostname.endswith(".endpoints.huggingface.cloud")
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("HF_ENDPOINT_URL must be a Hugging Face dedicated HTTPS endpoint URL")
        if not token.strip():
            raise ValueError("HF_TOKEN is required for remote inference")
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(60.0, connect=10.0),
            follow_redirects=False,
        )
        self.url = url

    def predict(self, raw: dict) -> dict:
        try:
            response = self.client.post(self.url, json={"inputs": raw})
            response.raise_for_status()
            return PredictionResponse.model_validate(response.json()).model_dump()
        except (httpx.HTTPError, ValueError) as exc:
            # Do not pass upstream bodies, URLs or credentials to the browser.
            raise RemotePredictionUnavailable(
                "The prediction service is unavailable or starting. Please try again shortly."
            ) from exc

    def close(self) -> None:
        self.client.close()
