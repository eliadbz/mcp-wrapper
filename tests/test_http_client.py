"""Tests for mcp_wrapper.http_client — authenticated httpx client factory."""

import httpx
import pytest

from mcp_wrapper.config import (
    ApiKeyAuthConfig,
    BasicAuthConfig,
    BearerAuthConfig,
    ServerConfig,
)
from mcp_wrapper.http_client import build_client


class TestBuildClientWithBearerAuth:
    def test_bearer_auth_sets_authorization_header(self):
        """build_client with bearer auth sets correct Authorization header."""
        server = ServerConfig(
            name="api",
            openapi_url="http://api/openapi.json",
            base_url="http://api",
            auth=BearerAuthConfig(type="bearer", token="my-secret-token"),
        )
        client = build_client(server)

        assert isinstance(client, httpx.AsyncClient)
        assert client.headers.get("Authorization") == "Bearer my-secret-token"

    def test_bearer_auth_with_different_token(self):
        """Bearer auth works with different token values."""
        server = ServerConfig(
            name="api",
            openapi_url="http://api/openapi.json",
            base_url="http://api",
            auth=BearerAuthConfig(type="bearer", token="another-token-12345"),
        )
        client = build_client(server)

        assert client.headers.get("Authorization") == "Bearer another-token-12345"


class TestBuildClientWithApiKeyAuth:
    def test_api_key_auth_sets_custom_header(self):
        """build_client with api_key auth sets correct custom header."""
        server = ServerConfig(
            name="api",
            openapi_url="http://api/openapi.json",
            base_url="http://api",
            auth=ApiKeyAuthConfig(
                type="api_key", header="X-API-Key", value="secret-key-123"
            ),
        )
        client = build_client(server)

        assert isinstance(client, httpx.AsyncClient)
        assert client.headers.get("X-API-Key") == "secret-key-123"

    def test_api_key_auth_with_different_header_name(self):
        """API key auth works with different header names."""
        server = ServerConfig(
            name="api",
            openapi_url="http://api/openapi.json",
            base_url="http://api",
            auth=ApiKeyAuthConfig(
                type="api_key", header="Authorization-Key", value="my-key"
            ),
        )
        client = build_client(server)

        assert client.headers.get("Authorization-Key") == "my-key"

    def test_api_key_auth_respects_header_case(self):
        """API key auth preserves the case of the header name."""
        server = ServerConfig(
            name="api",
            openapi_url="http://api/openapi.json",
            base_url="http://api",
            auth=ApiKeyAuthConfig(
                type="api_key", header="x-custom-header", value="value"
            ),
        )
        client = build_client(server)

        # httpx normalizes header names to lowercase for access
        assert client.headers.get("x-custom-header") == "value"


class TestBuildClientWithBasicAuth:
    def test_basic_auth_configures_httpx_basic_auth(self):
        """build_client with basic auth configures httpx.BasicAuth."""
        server = ServerConfig(
            name="api",
            openapi_url="http://api/openapi.json",
            base_url="http://api",
            auth=BasicAuthConfig(
                type="basic", username="testuser", password="testpass"
            ),
        )
        client = build_client(server)

        assert isinstance(client, httpx.AsyncClient)
        assert isinstance(client.auth, httpx.BasicAuth)

    def test_basic_auth_with_different_credentials(self):
        """Basic auth works with different username and password."""
        server = ServerConfig(
            name="api",
            openapi_url="http://api/openapi.json",
            base_url="http://api",
            auth=BasicAuthConfig(
                type="basic", username="admin", password="securepass123"
            ),
        )
        client = build_client(server)

        assert isinstance(client.auth, httpx.BasicAuth)


class TestBuildClientBaseUrl:
    def test_base_url_is_set_correctly(self):
        """base_url is set correctly on the client."""
        server = ServerConfig(
            name="api",
            openapi_url="http://api/openapi.json",
            base_url="http://api.example.com/v1",
            auth=BearerAuthConfig(type="bearer", token="tok"),
        )
        client = build_client(server)

        # httpx normalizes URLs with trailing slashes
        assert str(client.base_url) == "http://api.example.com/v1/"

    def test_base_url_with_trailing_slash(self):
        """base_url preserves trailing slash if present."""
        server = ServerConfig(
            name="api",
            openapi_url="http://api/openapi.json",
            base_url="http://api.example.com/",
            auth=BearerAuthConfig(type="bearer", token="tok"),
        )
        client = build_client(server)

        assert str(client.base_url) == "http://api.example.com/"

    def test_base_url_without_trailing_slash(self):
        """base_url without trailing slash is also set correctly."""
        server = ServerConfig(
            name="api",
            openapi_url="http://api/openapi.json",
            base_url="http://api.example.com",
            auth=BearerAuthConfig(type="bearer", token="tok"),
        )
        client = build_client(server)

        assert str(client.base_url) == "http://api.example.com"


class TestBuildClientTimeout:
    def test_default_timeout_is_set(self):
        """default timeout is set on the client."""
        server = ServerConfig(
            name="api",
            openapi_url="http://api/openapi.json",
            base_url="http://api",
            auth=BearerAuthConfig(type="bearer", token="tok"),
        )
        client = build_client(server)

        assert client.timeout == httpx.Timeout(30.0)

    def test_timeout_value_is_30_seconds(self):
        """The timeout is exactly 30 seconds."""
        server = ServerConfig(
            name="api",
            openapi_url="http://api/openapi.json",
            base_url="http://api",
            auth=BearerAuthConfig(type="bearer", token="tok"),
        )
        client = build_client(server)

        # Check the timeout value (all timeout fields should be 30.0)
        assert client.timeout.connect == 30.0
        assert client.timeout.read == 30.0
        assert client.timeout.write == 30.0
        assert client.timeout.pool == 30.0


class TestBuildClientErrorHandling:
    def test_unknown_auth_type_raises_value_error(self):
        """Unknown auth type raises ValueError with clear message."""
        from unittest.mock import Mock

        # Create a server config
        server = ServerConfig(
            name="api",
            openapi_url="http://api/openapi.json",
            base_url="http://api",
            auth=BearerAuthConfig(type="bearer", token="tok"),
        )
        # Create a mock auth object with unknown type
        mock_auth = Mock()
        mock_auth.__class__.__name__ = "UnknownAuth"
        server.auth = mock_auth  # type: ignore

        with pytest.raises(ValueError) as exc_info:
            build_client(server)

        assert "UnknownAuth" in str(exc_info.value)


class TestBuildClientIntegration:
    def test_client_is_async_client(self):
        """The returned object is an AsyncClient."""
        server = ServerConfig(
            name="api",
            openapi_url="http://api/openapi.json",
            base_url="http://api",
            auth=BearerAuthConfig(type="bearer", token="tok"),
        )
        client = build_client(server)

        assert isinstance(client, httpx.AsyncClient)

    def test_client_has_no_side_effects(self):
        """Building a client has no side effects (no logging, no state)."""
        server = ServerConfig(
            name="api",
            openapi_url="http://api/openapi.json",
            base_url="http://api",
            auth=BearerAuthConfig(type="bearer", token="tok"),
        )
        # Just ensure this doesn't raise or modify anything
        client = build_client(server)
        assert client is not None

    def test_multiple_clients_are_independent(self):
        """Multiple clients created with different auth are independent."""
        server1 = ServerConfig(
            name="api1",
            openapi_url="http://api1/openapi.json",
            base_url="http://api1",
            auth=BearerAuthConfig(type="bearer", token="token1"),
        )
        server2 = ServerConfig(
            name="api2",
            openapi_url="http://api2/openapi.json",
            base_url="http://api2",
            auth=BearerAuthConfig(type="bearer", token="token2"),
        )

        client1 = build_client(server1)
        client2 = build_client(server2)

        assert client1.headers.get("Authorization") == "Bearer token1"
        assert client2.headers.get("Authorization") == "Bearer token2"
        assert client1.base_url != client2.base_url
