"""Base API client with rate limiting, error handling, and HTTP functionality."""

import json
import logging
import time
from typing import Any, Optional, TypeVar, Union
from urllib.parse import urljoin

from pydantic import BaseModel, ValidationError
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

T = TypeVar("T", bound=BaseModel)

# HTTP Status Code Constants
HTTP_OK = 200
HTTP_NOT_FOUND = 404
HTTP_RATE_LIMIT = 429
HTTP_SERVER_ERROR = 500


class RateLimiter:
    """Simple rate limiter to prevent API abuse."""

    def __init__(self, max_requests: int = 4, time_window: float = 1.0):
        """Initialize rate limiter.

        Args:
            max_requests: Maximum number of requests allowed in time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: list[float] = []

    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        now = time.time()

        # Remove old requests outside the time window
        self.requests = [
            req_time for req_time in self.requests if now - req_time < self.time_window
        ]

        # If we're at the limit, wait
        if len(self.requests) >= self.max_requests:
            sleep_time = self.time_window - (now - self.requests[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                # Clean up old requests again
                now = time.time()
                self.requests = [
                    req_time
                    for req_time in self.requests
                    if now - req_time < self.time_window
                ]

        # Record this request
        self.requests.append(now)


class APIError(Exception):
    """Base API error."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[dict] = None,
    ):
        """Initialize API error.

        Args:
            message: Error message
            status_code: HTTP status code
            response_data: Response data if available
        """
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}


class APITimeoutError(APIError):
    """API timeout error."""

    pass


class APIRateLimitError(APIError):
    """API rate limit error."""

    pass


class APINotFoundError(APIError):
    """API not found error."""

    pass


class APIServerError(APIError):
    """API server error."""

    pass


class BaseAPIClient:
    """Base API client with rate limiting, error handling, and retry logic."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 0.3,
        rate_limit_requests: int = 4,
        rate_limit_window: float = 1.0,
        headers: Optional[dict[str, str]] = None,
    ):
        """Initialize the base API client.

        Args:
            base_url: Base URL for the API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Backoff factor for retry delays
            rate_limit_requests: Maximum requests per time window
            rate_limit_window: Rate limit time window in seconds
            headers: Additional headers to include in requests
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.logger = logging.getLogger(self.__class__.__name__)

        # Set up rate limiter
        self.rate_limiter = RateLimiter(rate_limit_requests, rate_limit_window)

        # Set up session with retry strategy
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set default headers
        default_headers = {
            "User-Agent": "F1-Predict/1.0.0 (Python Ergast API Client)",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        }

        if headers:
            default_headers.update(headers)

        self.session.headers.update(default_headers)

        self.logger.info(
            "API client initialized: base_url=%s, timeout=%s, max_retries=%s, "
            "rate_limit=%s",
            self.base_url,
            timeout,
            max_retries,
            f"{rate_limit_requests}/{rate_limit_window}s",
        )

    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint.

        Args:
            endpoint: API endpoint

        Returns:
            Full URL
        """
        return urljoin(f"{self.base_url}/", endpoint.lstrip("/"))

    def _handle_response(self, response: requests.Response) -> dict[str, Any]:
        """Handle API response and extract data.

        Args:
            response: HTTP response object

        Returns:
            Parsed JSON data

        Raises:
            APIError: For various API error conditions
        """
        # Log response details
        self.logger.debug(
            "API response received: status_code=%d, url=%s",
            response.status_code,
            response.url,
        )

        # Handle different status codes
        if response.status_code == HTTP_OK:
            try:
                data = response.json()
                self.logger.debug(
                    "Response parsed successfully: data_keys=%s",
                    list(data.keys()) if isinstance(data, dict) else "non-dict",
                )
                return data
            except json.JSONDecodeError as e:
                self.logger.error(
                    "Failed to parse JSON response: %s (content: %s)",
                    str(e),
                    response.text[:500],
                )
                raise APIError(f"Invalid JSON response: {e}") from e

        elif response.status_code == HTTP_NOT_FOUND:
            self.logger.warning("Resource not found: %s", response.url)
            raise APINotFoundError("Resource not found", status_code=HTTP_NOT_FOUND)

        elif response.status_code == HTTP_RATE_LIMIT:
            self.logger.warning("Rate limit exceeded: %s", response.url)
            raise APIRateLimitError("Rate limit exceeded", status_code=HTTP_RATE_LIMIT)

        elif response.status_code >= HTTP_SERVER_ERROR:
            self.logger.error(
                "Server error: status_code=%d, url=%s",
                response.status_code,
                response.url,
            )
            raise APIServerError(
                f"Server error: {response.status_code}",
                status_code=response.status_code,
            )

        else:
            self.logger.error(
                "Unexpected status code: %d, url=%s", response.status_code, response.url
            )
            raise APIError(
                f"Unexpected status code: {response.status_code}",
                status_code=response.status_code,
            )

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make HTTP request with rate limiting and error handling.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            **kwargs: Additional request arguments

        Returns:
            Parsed response data

        Raises:
            APIError: For various API error conditions
        """
        url = self._build_url(endpoint)

        # Apply rate limiting
        self.rate_limiter.wait_if_needed()

        # Prepare request parameters
        request_kwargs = {
            "timeout": self.timeout,
            "params": params,
            **kwargs,
        }

        self.logger.debug(
            "Making API request: method=%s, url=%s, params=%s", method, url, params
        )

        try:
            start_time = time.time()
            response = self.session.request(method, url, **request_kwargs)
            duration = time.time() - start_time

            self.logger.info(
                "API request completed: method=%s, url=%s, status_code=%d, "
                "duration_ms=%.1f",
                method,
                url,
                response.status_code,
                round(duration * 1000, 2),
            )

            return self._handle_response(response)

        except requests.exceptions.Timeout as e:
            self.logger.error("Request timeout: url=%s, timeout=%s", url, self.timeout)
            raise APITimeoutError(f"Request timeout after {self.timeout}s") from e

        except requests.exceptions.ConnectionError as e:
            self.logger.error("Connection error: url=%s, error=%s", url, str(e))
            raise APIError(f"Connection error: {e}") from e

        except requests.exceptions.RequestException as e:
            self.logger.error("Request error: url=%s, error=%s", url, str(e))
            raise APIError(f"Request error: {e}") from e

    def get(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        model: Optional[type[T]] = None,
    ) -> Union[dict[str, Any], T]:
        """Make GET request.

        Args:
            endpoint: API endpoint
            params: Query parameters
            model: Optional Pydantic model to parse response

        Returns:
            Raw data or parsed model instance
        """
        data = self._make_request("GET", endpoint, params=params)

        if model:
            try:
                return model.model_validate(data)
            except ValidationError as e:
                self.logger.error(
                    "Failed to parse response with model %s: %s (data_keys: %s)",
                    model.__name__,
                    str(e),
                    list(data.keys()) if isinstance(data, dict) else "non-dict",
                )
                # Return raw data if model parsing fails
                return data

        return data

    def post(
        self,
        endpoint: str,
        data: Optional[dict[str, Any]] = None,
        json_data: Optional[dict[str, Any]] = None,
        model: Optional[type[T]] = None,
    ) -> Union[dict[str, Any], T]:
        """Make POST request.

        Args:
            endpoint: API endpoint
            data: Form data
            json_data: JSON data
            model: Optional Pydantic model to parse response

        Returns:
            Raw data or parsed model instance
        """
        kwargs = {}
        if data:
            kwargs["data"] = data
        if json_data:
            kwargs["json"] = json_data

        response_data = self._make_request("POST", endpoint, **kwargs)

        if model:
            try:
                return model.model_validate(response_data)
            except ValidationError as e:
                self.logger.error(
                    "Failed to parse response with model %s: %s", model.__name__, str(e)
                )
                return response_data

        return response_data

    def close(self) -> None:
        """Close the HTTP session."""
        if self.session:
            self.session.close()
            self.logger.info("API client session closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
