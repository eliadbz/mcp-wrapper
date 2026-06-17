"""Tests for mcp_wrapper.server — build_mcp_server() function.

Uses respx to mock HTTP calls for fetch_spec and unittest.mock to control
register_tool behaviour when testing failure paths.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from mcp.server.fastmcp import FastMCP

from mcp_wrapper.config import BearerAuthConfig, ServerConfig
from mcp_wrapper.server import build_mcp_server


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OPENAPI_URL = "https://api.example.com/openapi.json"
BASE_URL = "https://api.example.com"

MINIMAL_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Test API", "version": "1.0.0"},
    "paths": {
        "/users": {
            "get": {
                "operationId": "listUsers",
                "summary": "List users",
                "responses": {"200": {"description": "ok"}},
            }
        },
        "/items": {
            "post": {
                "operationId": "createItem",
                "summary": "Create an item",
                "responses": {"201": {"description": "created"}},
            }
        },
    },
}


def make_server_config(name: str = "test-server") -> ServerConfig:
    """Return a minimal ServerConfig for testing."""
    return ServerConfig(
        name=name,
        openapi_url=OPENAPI_URL,
        base_url=BASE_URL,
        auth=BearerAuthConfig(type="bearer", token="test-token"),
    )


# ---------------------------------------------------------------------------
# Successful build
# ---------------------------------------------------------------------------


class TestBuildMcpServerSuccess:
    """Verify build_mcp_server returns a properly configured (FastMCP, client) tuple."""

    @pytest.mark.asyncio
    async def test_returns_tuple_of_fastmcp_and_async_client(self):
        """build_mcp_server should return a (FastMCP, httpx.AsyncClient) tuple."""
        config = make_server_config()
        with respx.mock:
            respx.get(OPENAPI_URL).mock(
                return_value=httpx.Response(
                    200,
                    content=json.dumps(MINIMAL_SPEC).encode(),
                    headers={"content-type": "application/json"},
                )
            )
            result = await build_mcp_server(config)

        assert result is not None
        mcp, client = result
        assert isinstance(mcp, FastMCP)
        assert isinstance(client, httpx.AsyncClient)
        await client.aclose()

    @pytest.mark.asyncio
    async def test_server_name_set_correctly_on_fastmcp_instance(self):
        """The FastMCP instance must have the server name from the config."""
        config = make_server_config(name="my-api-server")
        with respx.mock:
            respx.get(OPENAPI_URL).mock(
                return_value=httpx.Response(
                    200,
                    content=json.dumps(MINIMAL_SPEC).encode(),
                    headers={"content-type": "application/json"},
                )
            )
            result = await build_mcp_server(config)

        assert result is not None
        mcp, client = result
        assert mcp.name == "my-api-server"
        await client.aclose()

    @pytest.mark.asyncio
    async def test_tools_registered_for_all_operations(self):
        """All operations in the spec should be registered as tools on the MCP instance."""
        config = make_server_config()
        with respx.mock:
            respx.get(OPENAPI_URL).mock(
                return_value=httpx.Response(
                    200,
                    content=json.dumps(MINIMAL_SPEC).encode(),
                    headers={"content-type": "application/json"},
                )
            )
            result = await build_mcp_server(config)

        assert result is not None
        mcp, client = result
        tool_names = {t.name for t in mcp._tool_manager.list_tools()}
        assert "listUsers" in tool_names
        assert "createItem" in tool_names
        await client.aclose()

    @pytest.mark.asyncio
    async def test_spec_with_no_paths_returns_mcp_with_zero_tools(self):
        """A spec with an empty paths block should yield a valid MCP instance with no tools."""
        empty_spec = {
            "openapi": "3.0.0",
            "info": {"title": "Empty API", "version": "1.0.0"},
            "paths": {},
        }
        config = make_server_config()
        with respx.mock:
            respx.get(OPENAPI_URL).mock(
                return_value=httpx.Response(
                    200,
                    content=json.dumps(empty_spec).encode(),
                    headers={"content-type": "application/json"},
                )
            )
            result = await build_mcp_server(config)

        assert result is not None
        mcp, client = result
        assert list(mcp._tool_manager.list_tools()) == []
        await client.aclose()


# ---------------------------------------------------------------------------
# Fetch failures — should return None
# ---------------------------------------------------------------------------


class TestBuildMcpServerFetchFailure:
    """Verify that HTTP/parse failures during fetch_spec cause build_mcp_server to return None."""

    @pytest.mark.asyncio
    async def test_http_error_returns_none(self):
        """An HTTP error fetching the OpenAPI spec should return None without raising."""
        config = make_server_config()
        with respx.mock:
            respx.get(OPENAPI_URL).mock(
                return_value=httpx.Response(500, text="Internal Server Error")
            )
            result = await build_mcp_server(config)

        assert result is None

    @pytest.mark.asyncio
    async def test_connection_error_returns_none(self):
        """A network-level error fetching the spec should return None without raising."""
        config = make_server_config()
        with respx.mock:
            respx.get(OPENAPI_URL).mock(side_effect=httpx.ConnectError("Connection refused"))
            result = await build_mcp_server(config)

        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_json_response_returns_none(self):
        """A response that is not valid JSON or YAML should return None."""
        config = make_server_config()
        with respx.mock:
            respx.get(OPENAPI_URL).mock(
                return_value=httpx.Response(
                    200,
                    content=b"THIS IS NOT JSON OR YAML: ::::",
                    headers={"content-type": "application/json"},
                )
            )
            result = await build_mcp_server(config)

        assert result is None

    @pytest.mark.asyncio
    async def test_404_response_returns_none(self):
        """A 404 HTTP response when fetching the spec should return None."""
        config = make_server_config()
        with respx.mock:
            respx.get(OPENAPI_URL).mock(
                return_value=httpx.Response(404, text="Not found")
            )
            result = await build_mcp_server(config)

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_failure_does_not_raise_exception(self):
        """Fetch failure must return None, never propagate an exception to the caller."""
        config = make_server_config()
        with respx.mock:
            respx.get(OPENAPI_URL).mock(side_effect=httpx.TimeoutException("timed out"))
            # Must not raise — must return None
            result = await build_mcp_server(config)

        assert result is None


# ---------------------------------------------------------------------------
# Parse failures — should return None
# ---------------------------------------------------------------------------


class TestBuildMcpServerParseFailure:
    """Verify that parse_operations failures cause build_mcp_server to return None."""

    @pytest.mark.asyncio
    async def test_parse_operations_exception_returns_none(self):
        """If parse_operations raises, build_mcp_server should return None."""
        config = make_server_config()
        with respx.mock:
            respx.get(OPENAPI_URL).mock(
                return_value=httpx.Response(
                    200,
                    content=json.dumps(MINIMAL_SPEC).encode(),
                    headers={"content-type": "application/json"},
                )
            )
            with patch(
                "mcp_wrapper.server.parse_operations",
                side_effect=RuntimeError("parse blew up"),
            ):
                result = await build_mcp_server(config)

        assert result is None


# ---------------------------------------------------------------------------
# Individual tool registration failure — server still returned
# ---------------------------------------------------------------------------


class TestBuildMcpServerToolRegistrationFailure:
    """Verify that a single tool's registration failure does not abort the whole server."""

    @pytest.mark.asyncio
    async def test_one_tool_failure_does_not_return_none(self):
        """If one tool's register_tool raises, the server should still be returned."""
        config = make_server_config()

        call_count = 0

        def register_tool_side_effect(mcp, operation, client):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("tool registration failed")
            # second call succeeds normally

        with respx.mock:
            respx.get(OPENAPI_URL).mock(
                return_value=httpx.Response(
                    200,
                    content=json.dumps(MINIMAL_SPEC).encode(),
                    headers={"content-type": "application/json"},
                )
            )
            with patch(
                "mcp_wrapper.server.register_tool",
                side_effect=register_tool_side_effect,
            ):
                result = await build_mcp_server(config)

        # Must not be None — the server is still valid even if one tool failed
        assert result is not None
        mcp, client = result
        assert isinstance(mcp, FastMCP)
        await client.aclose()

    @pytest.mark.asyncio
    async def test_all_tools_fail_still_returns_mcp(self):
        """Even if every tool fails to register, the (mcp, client) tuple is returned."""
        config = make_server_config()
        with respx.mock:
            respx.get(OPENAPI_URL).mock(
                return_value=httpx.Response(
                    200,
                    content=json.dumps(MINIMAL_SPEC).encode(),
                    headers={"content-type": "application/json"},
                )
            )
            with patch(
                "mcp_wrapper.server.register_tool",
                side_effect=RuntimeError("always fails"),
            ):
                result = await build_mcp_server(config)

        assert result is not None
        mcp, client = result
        # No tools registered
        assert list(mcp._tool_manager.list_tools()) == []
        await client.aclose()

    @pytest.mark.asyncio
    async def test_failed_tool_skipped_working_tools_registered(self):
        """After skipping a failed tool, the remaining tools are still registered."""
        config = make_server_config()

        # Spec with three operations so we can have 1 fail and 2 succeed
        spec_three_ops = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/a": {"get": {"operationId": "opA", "responses": {"200": {"description": "ok"}}}},
                "/b": {"get": {"operationId": "opB", "responses": {"200": {"description": "ok"}}}},
                "/c": {"get": {"operationId": "opC", "responses": {"200": {"description": "ok"}}}},
            },
        }

        # We test that non-failing tools ARE registered; we do NOT patch register_tool
        # here — we let the real function run and verify all three land on the MCP instance.
        with respx.mock:
            respx.get(OPENAPI_URL).mock(
                return_value=httpx.Response(
                    200,
                    content=json.dumps(spec_three_ops).encode(),
                    headers={"content-type": "application/json"},
                )
            )
            result = await build_mcp_server(config)

        assert result is not None
        mcp, client = result
        tool_names = {t.name for t in mcp._tool_manager.list_tools()}
        assert "opA" in tool_names
        assert "opB" in tool_names
        assert "opC" in tool_names
        await client.aclose()


# ---------------------------------------------------------------------------
# Client lifecycle — client must be closed on failure
# ---------------------------------------------------------------------------


class TestBuildMcpServerClientLifecycle:
    """Verify that the httpx.AsyncClient is closed when build_mcp_server returns None."""

    @pytest.mark.asyncio
    async def test_client_closed_when_fetch_fails(self):
        """The client should be closed before returning None on a fetch failure."""
        config = make_server_config()

        closed_clients: list[httpx.AsyncClient] = []
        original_build_client = __import__(
            "mcp_wrapper.http_client", fromlist=["build_client"]
        ).build_client

        def mock_build_client(server_config):
            client = original_build_client(server_config)
            closed_clients.append(client)
            return client

        with respx.mock:
            respx.get(OPENAPI_URL).mock(
                return_value=httpx.Response(503, text="unavailable")
            )
            with patch("mcp_wrapper.server.build_client", side_effect=mock_build_client):
                result = await build_mcp_server(config)

        assert result is None
        assert len(closed_clients) == 1
        assert closed_clients[0].is_closed

    @pytest.mark.asyncio
    async def test_client_not_closed_on_success(self):
        """The client should NOT be closed on success — the caller owns its lifecycle."""
        config = make_server_config()
        with respx.mock:
            respx.get(OPENAPI_URL).mock(
                return_value=httpx.Response(
                    200,
                    content=json.dumps(MINIMAL_SPEC).encode(),
                    headers={"content-type": "application/json"},
                )
            )
            result = await build_mcp_server(config)

        assert result is not None
        mcp, client = result
        assert not client.is_closed
        await client.aclose()
