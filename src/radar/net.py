import time
from collections.abc import Callable
from typing import Any

import httpx

USER_AGENT = "internship-radar (+https://github.com/lviss-1/internship-radar)"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class NetworkError(Exception):
    """Raised after retries are exhausted.

    The message carries a caller-supplied label instead of the URL, because some URLs
    (Discord webhooks) are secrets and exception text ends up in public logs.
    """

    def __init__(self, label: str, reason: str, status: int | None = None) -> None:
        super().__init__(f"{label}: {reason}")
        self.label = label
        self.reason = reason
        self.status = status


def make_client(timeout_seconds: float = 30.0) -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(timeout_seconds, connect=10.0),
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    )


def _retry_after_seconds(response: httpx.Response) -> float | None:
    header = response.headers.get("Retry-After")
    if header is None:
        return None
    try:
        return max(0.0, float(header))
    except ValueError:
        return None


def request_with_retries(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    label: str,
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    sleep: Callable[[float], None] | None = None,
    **kwargs: Any,
) -> httpx.Response:
    """Send a request, retrying transport errors, 429 and 5xx with exponential backoff.

    Non-retryable responses (including 4xx) are returned to the caller to interpret.
    """
    pause = sleep or time.sleep
    for attempt in range(1, attempts + 1):
        server_delay: float | None = None
        status: int | None = None
        try:
            response = client.request(method, url, **kwargs)
        except httpx.TimeoutException:
            reason = "timed out"
        except httpx.TransportError as exc:
            reason = f"transport error ({type(exc).__name__})"
        else:
            if response.status_code not in RETRYABLE_STATUS:
                return response
            status = response.status_code
            reason = f"HTTP {status}"
            server_delay = _retry_after_seconds(response)

        if attempt == attempts:
            raise NetworkError(label, f"{reason} after {attempts} attempts", status)
        backoff = base_delay * 2 ** (attempt - 1)
        pause(min(max_delay, server_delay if server_delay is not None else backoff))

    raise AssertionError("unreachable")
