"""
API Wrapper Tool — Retry + Rate Limit Pattern

Demonstrates a production-ready tool that wraps an external API with:
- Exponential backoff on transient errors (429, 500, 502, 503)
- Max retry limit (default: 3)
- Structured error reporting
- Request/response logging

Usage:
    from tools.api_wrapper import ResilientAPI

    api = ResilientAPI(base_url="https://api.example.com", api_key="...")
    result = api.get("/products", params={"limit": 10})
"""

import time
import json
import logging
import urllib.request
import urllib.error
import urllib.parse
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Transient HTTP status codes that warrant a retry
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class ResilientAPI:
    """
    API client with automatic retry and exponential backoff.

    Designed for use as a LangChain tool backend — wraps any REST API
    and handles the messy parts (rate limits, transient failures) so
    the agent only sees clean success/failure responses.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        max_retries: int = 3,
        base_delay: float = 1.0,
        timeout: int = 15,
    ):
        """
        Args:
            base_url: API base URL (no trailing slash)
            api_key: Bearer token or API key for Authorization header
            max_retries: Maximum retry attempts on transient errors
            base_delay: Base delay in seconds (doubles each retry)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.timeout = timeout

    def _build_headers(self, extra_headers: Optional[dict] = None) -> dict:
        """Build request headers with auth and content type."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "LangChainAgentPatterns/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        body: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> dict:
        """
        Make an HTTP request with retry logic.

        Returns:
            dict with 'success', 'data', 'status_code', and optionally 'error'
        """
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)

        req_headers = self._build_headers(headers)
        data = json.dumps(body).encode("utf-8") if body else None

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    url, data=data, headers=req_headers, method=method
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    response_data = json.loads(resp.read().decode("utf-8"))
                    logger.info(f"{method} {path} → {resp.status} (attempt {attempt + 1})")
                    return {
                        "success": True,
                        "data": response_data,
                        "status_code": resp.status,
                    }

            except urllib.error.HTTPError as e:
                last_error = e
                status = e.code
                logger.warning(
                    f"{method} {path} → {status} (attempt {attempt + 1}/{self.max_retries + 1})"
                )

                if status not in RETRYABLE_STATUS_CODES:
                    # Non-retryable error — fail immediately
                    error_body = ""
                    try:
                        error_body = e.read().decode("utf-8")[:500]
                    except Exception:
                        pass
                    return {
                        "success": False,
                        "error": f"HTTP {status}: {error_body or e.reason}",
                        "status_code": status,
                    }

                # Retryable — check for Retry-After header
                retry_after = e.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = self.base_delay * (2 ** attempt)
                else:
                    delay = self.base_delay * (2 ** attempt)

                if attempt < self.max_retries:
                    logger.info(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)

            except urllib.error.URLError as e:
                last_error = e
                logger.warning(f"{method} {path} → URLError: {e.reason} (attempt {attempt + 1})")
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    time.sleep(delay)

            except Exception as e:
                last_error = e
                logger.error(f"{method} {path} → Unexpected error: {e}")
                return {
                    "success": False,
                    "error": f"Unexpected error: {str(e)}",
                    "status_code": 0,
                }

        return {
            "success": False,
            "error": f"All {self.max_retries + 1} attempts failed. Last error: {last_error}",
            "status_code": getattr(last_error, "code", 0),
        }

    def get(self, path: str, params: Optional[dict] = None, **kwargs) -> dict:
        """GET request with retry."""
        return self._request("GET", path, params=params, **kwargs)

    def post(self, path: str, body: Optional[dict] = None, **kwargs) -> dict:
        """POST request with retry."""
        return self._request("POST", path, body=body, **kwargs)

    def put(self, path: str, body: Optional[dict] = None, **kwargs) -> dict:
        """PUT request with retry."""
        return self._request("PUT", path, body=body, **kwargs)

    def delete(self, path: str, **kwargs) -> dict:
        """DELETE request with retry."""
        return self._request("DELETE", path, **kwargs)


# ─── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    api = ResilientAPI(
        base_url="https://jsonplaceholder.typicode.com",
        max_retries=2,
    )
    result = api.get("/posts/1")
    print(json.dumps(result, indent=2))
