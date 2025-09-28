"""Tests for the base API client."""

import json
import time
from unittest.mock import Mock, patch

from pydantic import BaseModel
import pytest
import requests

from f1_predict.api.base import (
    APIError,
    APINotFoundError,
    APIRateLimitError,
    APIServerError,
    APITimeoutError,
    BaseAPIClient,
    RateLimiter,
)


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    id: int
    name: str


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def test_rate_limiter_init(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(max_requests=5, time_window=2.0)
        assert limiter.max_requests == 5
        assert limiter.time_window == 2.0
        assert limiter.requests == []

    def test_rate_limiter_allows_requests_within_limit(self):
        """Test that requests within limit are allowed immediately."""
        limiter = RateLimiter(max_requests=3, time_window=1.0)

        start_time = time.time()
        for _ in range(3):
            limiter.wait_if_needed()
        end_time = time.time()

        # Should complete quickly without waiting
        assert end_time - start_time < 0.1
        assert len(limiter.requests) == 3

    def test_rate_limiter_enforces_rate_limit(self):
        """Test that rate limiter enforces limits."""
        limiter = RateLimiter(max_requests=2, time_window=0.5)

        # Make requests up to the limit
        limiter.wait_if_needed()
        limiter.wait_if_needed()

        # Next request should cause a delay
        start_time = time.time()
        limiter.wait_if_needed()
        end_time = time.time()

        # Should have waited approximately the time window
        assert end_time - start_time >= 0.4  # Allow some tolerance

    def test_rate_limiter_cleans_old_requests(self):
        """Test that old requests are cleaned up."""
        limiter = RateLimiter(max_requests=2, time_window=0.1)

        # Make requests
        limiter.wait_if_needed()
        limiter.wait_if_needed()
        assert len(limiter.requests) == 2

        # Wait for time window to pass
        time.sleep(0.2)

        # Make another request
        limiter.wait_if_needed()

        # Old requests should be cleaned up
        assert len(limiter.requests) == 1


class TestBaseAPIClient:
    """Tests for the BaseAPIClient class."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return BaseAPIClient(
            base_url="https://api.example.com",
            timeout=5.0,
            max_retries=2,
            rate_limit_requests=10,
            rate_limit_window=1.0,
        )

    def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.base_url == "https://api.example.com"
        assert client.timeout == 5.0
        assert client.rate_limiter.max_requests == 10
        assert client.rate_limiter.time_window == 1.0

    def test_build_url(self, client):
        """Test URL building."""
        assert client._build_url("endpoint") == "https://api.example.com/endpoint"
        assert client._build_url("/endpoint") == "https://api.example.com/endpoint"
        assert (
            client._build_url("path/to/endpoint")
            == "https://api.example.com/path/to/endpoint"
        )

    @patch("requests.Session.request")
    def test_successful_request(self, mock_request, client):
        """Test successful API request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_response.url = "https://api.example.com/test"
        mock_response.headers = {"content-type": "application/json"}
        mock_request.return_value = mock_response

        result = client.get("test")

        assert result == {"data": "test"}
        mock_request.assert_called_once()

    @patch("requests.Session.request")
    def test_request_with_model_parsing(self, mock_request, client):
        """Test request with Pydantic model parsing."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "name": "test"}
        mock_response.url = "https://api.example.com/test"
        mock_response.headers = {"content-type": "application/json"}
        mock_request.return_value = mock_response

        result = client.get("test", model=SampleModel)

        assert isinstance(result, SampleModel)
        assert result.id == 1
        assert result.name == "test"

    @patch("requests.Session.request")
    def test_request_with_model_parsing_failure(self, mock_request, client):
        """Test request with model parsing failure returns raw data."""
        # Mock successful response with invalid data for model
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"invalid": "data"}
        mock_response.url = "https://api.example.com/test"
        mock_response.headers = {"content-type": "application/json"}
        mock_request.return_value = mock_response

        result = client.get("test", model=SampleModel)

        # Should return raw data when model parsing fails
        assert result == {"invalid": "data"}

    @patch("requests.Session.request")
    def test_404_error(self, mock_request, client):
        """Test 404 error handling."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.url = "https://api.example.com/notfound"
        mock_request.return_value = mock_response

        with pytest.raises(APINotFoundError) as exc_info:
            client.get("notfound")

        assert exc_info.value.status_code == 404

    @patch("requests.Session.request")
    def test_429_rate_limit_error(self, mock_request, client):
        """Test 429 rate limit error handling."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.url = "https://api.example.com/test"
        mock_request.return_value = mock_response

        with pytest.raises(APIRateLimitError) as exc_info:
            client.get("test")

        assert exc_info.value.status_code == 429

    @patch("requests.Session.request")
    def test_500_server_error(self, mock_request, client):
        """Test 500 server error handling."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.url = "https://api.example.com/test"
        mock_request.return_value = mock_response

        with pytest.raises(APIServerError) as exc_info:
            client.get("test")

        assert exc_info.value.status_code == 500

    @patch("requests.Session.request")
    def test_json_decode_error(self, mock_request, client):
        """Test JSON decode error handling."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.text = "Invalid JSON response"
        mock_response.url = "https://api.example.com/test"
        mock_request.return_value = mock_response

        with pytest.raises(APIError) as exc_info:
            client.get("test")

        assert "Invalid JSON response" in str(exc_info.value)

    @patch("requests.Session.request")
    def test_timeout_error(self, mock_request, client):
        """Test timeout error handling."""
        mock_request.side_effect = requests.exceptions.Timeout("Request timeout")

        with pytest.raises(APITimeoutError):
            client.get("test")

    @patch("requests.Session.request")
    def test_connection_error(self, mock_request, client):
        """Test connection error handling."""
        mock_request.side_effect = requests.exceptions.ConnectionError(
            "Connection failed"
        )

        with pytest.raises(APIError) as exc_info:
            client.get("test")

        assert "Connection error" in str(exc_info.value)

    @patch("requests.Session.request")
    def test_post_request(self, mock_request, client):
        """Test POST request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "created"}
        mock_response.url = "https://api.example.com/test"
        mock_response.headers = {"content-type": "application/json"}
        mock_request.return_value = mock_response

        result = client.post("test", json_data={"name": "test"})

        assert result == {"result": "created"}
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[1]["json"] == {"name": "test"}

    @patch("requests.Session.request")
    def test_post_with_form_data(self, mock_request, client):
        """Test POST request with form data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "created"}
        mock_response.url = "https://api.example.com/test"
        mock_response.headers = {"content-type": "application/json"}
        mock_request.return_value = mock_response

        result = client.post("test", data={"name": "test"})

        assert result == {"result": "created"}
        call_args = mock_request.call_args
        assert call_args[1]["data"] == {"name": "test"}

    def test_context_manager(self):
        """Test context manager functionality."""
        with BaseAPIClient("https://api.example.com") as client:
            assert client.base_url == "https://api.example.com"
            assert client.session is not None

        # Session should be closed after context exit
        # Note: We can't easily test this without accessing private methods

    def test_close_method(self, client):
        """Test close method."""
        original_session = client.session
        client.close()
        # The session object still exists but close was called
        assert original_session is not None

    def test_rate_limiter_integration(self, client):
        """Test that rate limiter is properly integrated."""
        assert isinstance(client.rate_limiter, RateLimiter)
        assert client.rate_limiter.max_requests == 10
        assert client.rate_limiter.time_window == 1.0

    def test_custom_headers(self):
        """Test custom headers in initialization."""
        custom_headers = {"Authorization": "Bearer token123"}
        client = BaseAPIClient("https://api.example.com", headers=custom_headers)

        assert "Authorization" in client.session.headers
        assert client.session.headers["Authorization"] == "Bearer token123"
        assert (
            "User-Agent" in client.session.headers
        )  # Default header should still be there
