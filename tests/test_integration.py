"""Integration tests for mcp-wrapper.

Exercises the full system stack: config → build_mcp_server → FastMCP tool
registration → tool call → HTTP request to target API.

Test strategy:
- Tool isolation tests call build_mcp_server directly (no FastAPI app needed)
  and inspect the resulting FastMCP instances' tool registries.
- Full request flow tests call mcp.call_tool() against a respx-mocked target
  API to verify the end-to-end handler path.
- App-level tests use FastAPI TestClient (triggers lifespan) with load_config
  patched so no filesystem or network I/O escapes to the real world.

All HTTP calls (OpenAPI spec fetch + target API calls) are intercepted by
respx — no real network traffic is made.
"""

from __future__ import annotations

import importlib
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from mcp.server.fastmcp import FastMCP

import mcp_wrapper.main
from mcp_wrapper.config import AppConfig, BearerAuthConfig, ApiKeyAuthConfig, ServerConfig
from mcp_wrapper.server import build_mcp_server


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

# Two distinct OpenAPI specs — server1 exposes get_user, server2 exposes list_items.
SPEC1 = {
    "openapi": "3.0.0",
    "info": {"title": "API1", "version": "1.0"},
    "paths": {
        "/users/{id}": {
            "get": {
                "operationId": "get_user",
                "summary": "Get a user by ID",
                "parameters": [
                    {
                        "name": "id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
                "responses": {"200": {"description": "ok"}},
            }
        }
    },
}

SPEC2 = {
    "openapi": "3.0.0",
    "info": {"title": "API2", "version": "1.0"},
    "paths": {
        "/items": {
            "get": {
                "operationId": "list_items",
                "summary": "List all items",
                "responses": {"200": {"description": "ok"}},
            }
        }
    },
}

OPENAPI_URL1 = "http://api1.test/openapi.json"
OPENAPI_URL2 = "http://api2.test/openapi.json"
BASE_URL1 = "http://api1.test"
BASE_URL2 = "http://api2.test"
TOKEN1 = "bearer-token-server1"
TOKEN2 = "bearer-token-server2"


def make_server_config(
    name: str,
    openapi_url: str,
    base_url: str,
    token: str,
) -> ServerConfig:
    """Return a ServerConfig with bearer auth."""
    return ServerConfig(
        name=name,
        openapi_url=openapi_url,
        base_url=base_url,
        auth=BearerAuthConfig(type="bearer", token=token),
    )


def tool_names(mcp: FastMCP) -> set[str]:
    """Return the set of tool names registered on a FastMCP instance."""
    return {t.name for t in mcp._tool_manager.list_tools()}


async def call_tool(mcp: FastMCP, tool_name: str, args: dict) -> str:
    """Invoke a registered MCP tool and return its text result."""
    results = await mcp.call_tool(tool_name, args)
    if isinstance(results, list) and results:
        first = results[0]
        return first.text if hasattr(first, "text") else str(first)
    return str(results)


def reload_main():
    """Reload mcp_wrapper.main and return the fresh module.

    Reloading ensures each test starts with a pristine FastAPI app that has
    no accumulated route mounts from previous tests.
    """
    importlib.reload(mcp_wrapper.main)
    return mcp_wrapper.main


def make_test_app_config(
    *,
    include_server2: bool = True,
) -> AppConfig:
    """Return an AppConfig with up to two servers."""
    servers: dict[str, ServerConfig] = {
        "server1": make_server_config("server1", OPENAPI_URL1, BASE_URL1, TOKEN1),
    }
    if include_server2:
        servers["server2"] = make_server_config("server2", OPENAPI_URL2, BASE_URL2, TOKEN2)
    return AppConfig(servers=servers)


# ---------------------------------------------------------------------------
# 1. Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """GET /health must return 200 {"status": "ok"} regardless of server state."""

    def test_health_endpoint_returns_ok(self):
        """Health check works when servers load successfully."""
        mod = reload_main()
        config = make_test_app_config()

        with respx.mock:
            respx.get(OPENAPI_URL1).mock(return_value=httpx.Response(200, json=SPEC1))
            respx.get(OPENAPI_URL2).mock(return_value=httpx.Response(200, json=SPEC2))

            with patch.object(mod, "load_config", return_value=config):
                with TestClient(mod.app) as client:
                    response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_endpoint_ok_when_all_servers_fail(self):
        """Health endpoint is unaffected even when every server fails to build."""
        mod = reload_main()
        config = make_test_app_config()

        # Both spec fetches return 503 — build_mcp_server returns None for each.
        with respx.mock:
            respx.get(OPENAPI_URL1).mock(return_value=httpx.Response(503, text="unavailable"))
            respx.get(OPENAPI_URL2).mock(return_value=httpx.Response(503, text="unavailable"))

            with patch.object(mod, "load_config", return_value=config):
                with TestClient(mod.app) as client:
                    response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# 2. Tool isolation — primary invariant
# ---------------------------------------------------------------------------


class TestToolIsolation:
    """server1 tools and server2 tools must be completely disjoint sets.

    server1 tools ∩ server2 tools = ∅ (empty — complete isolation).
    """

    @pytest.fixture
    async def both_servers(self):
        """Build both FastMCP instances and yield (mcp1, mcp2, client1, client2)."""
        config1 = make_server_config("server1", OPENAPI_URL1, BASE_URL1, TOKEN1)
        config2 = make_server_config("server2", OPENAPI_URL2, BASE_URL2, TOKEN2)

        with respx.mock:
            respx.get(OPENAPI_URL1).mock(return_value=httpx.Response(200, json=SPEC1))
            respx.get(OPENAPI_URL2).mock(return_value=httpx.Response(200, json=SPEC2))

            result1 = await build_mcp_server(config1)
            result2 = await build_mcp_server(config2)

        assert result1 is not None, "server1 build must succeed"
        assert result2 is not None, "server2 build must succeed"

        mcp1, client1 = result1
        mcp2, client2 = result2

        yield mcp1, mcp2, client1, client2

        await client1.aclose()
        await client2.aclose()

    async def test_server1_has_its_own_tool(self, both_servers):
        """server1 must expose get_user (defined in SPEC1)."""
        mcp1, _mcp2, *_ = both_servers
        assert "get_user" in tool_names(mcp1)

    async def test_server2_has_its_own_tool(self, both_servers):
        """server2 must expose list_items (defined in SPEC2)."""
        _mcp1, mcp2, *_ = both_servers
        assert "list_items" in tool_names(mcp2)

    async def test_server1_tools_not_in_server2(self, both_servers):
        """server1's get_user tool must NOT appear on server2's FastMCP instance."""
        _mcp1, mcp2, *_ = both_servers
        assert "get_user" not in tool_names(mcp2), (
            "Tool isolation violated: get_user (server1 tool) found in server2's tool registry"
        )

    async def test_server2_tools_not_in_server1(self, both_servers):
        """server2's list_items tool must NOT appear on server1's FastMCP instance."""
        mcp1, _mcp2, *_ = both_servers
        assert "list_items" not in tool_names(mcp1), (
            "Tool isolation violated: list_items (server2 tool) found in server1's tool registry"
        )

    async def test_tool_sets_are_disjoint(self, both_servers):
        """The complete tool sets of server1 and server2 must be disjoint."""
        mcp1, mcp2, *_ = both_servers
        intersection = tool_names(mcp1) & tool_names(mcp2)
        assert intersection == set(), (
            f"Tool isolation violated: shared tools found: {intersection}"
        )

    async def test_server1_has_exactly_one_tool(self, both_servers):
        """server1 should expose exactly one tool (get_user from SPEC1)."""
        mcp1, _mcp2, *_ = both_servers
        assert len(tool_names(mcp1)) == 1

    async def test_server2_has_exactly_one_tool(self, both_servers):
        """server2 should expose exactly one tool (list_items from SPEC2)."""
        _mcp1, mcp2, *_ = both_servers
        assert len(tool_names(mcp2)) == 1

    async def test_isolation_with_overlapping_paths(self):
        """Two servers with the same operationId must each have only their own tool.

        This is the hardest isolation case: both specs have a 'ping' operationId.
        Each FastMCP instance must only have its own version.
        """
        ping_spec_a = {
            "openapi": "3.0.0",
            "info": {"title": "A", "version": "1.0"},
            "paths": {
                "/ping": {
                    "get": {
                        "operationId": "ping",
                        "summary": "Ping A",
                        "responses": {"200": {"description": "ok"}},
                    }
                }
            },
        }
        ping_spec_b = {
            "openapi": "3.0.0",
            "info": {"title": "B", "version": "1.0"},
            "paths": {
                "/ping": {
                    "get": {
                        "operationId": "ping",
                        "summary": "Ping B",
                        "responses": {"200": {"description": "ok"}},
                    }
                }
            },
        }

        url_a = "http://server-a.test/openapi.json"
        url_b = "http://server-b.test/openapi.json"
        config_a = make_server_config("server-a", url_a, "http://server-a.test", "tok-a")
        config_b = make_server_config("server-b", url_b, "http://server-b.test", "tok-b")

        with respx.mock:
            respx.get(url_a).mock(return_value=httpx.Response(200, json=ping_spec_a))
            respx.get(url_b).mock(return_value=httpx.Response(200, json=ping_spec_b))

            result_a = await build_mcp_server(config_a)
            result_b = await build_mcp_server(config_b)

        assert result_a is not None
        assert result_b is not None

        mcp_a, client_a = result_a
        mcp_b, client_b = result_b

        # Both have a 'ping' tool, but they are registered on separate FastMCP
        # instances — the handlers inside each instance close over a different
        # httpx.AsyncClient bound to a different base_url.
        assert "ping" in tool_names(mcp_a)
        assert "ping" in tool_names(mcp_b)

        # Each instance has exactly one tool — no cross-contamination.
        assert len(tool_names(mcp_a)) == 1
        assert len(tool_names(mcp_b)) == 1

        await client_a.aclose()
        await client_b.aclose()


# ---------------------------------------------------------------------------
# 3. Full request flow — tool call → HTTP to target API
# ---------------------------------------------------------------------------


class TestFullRequestFlow:
    """A tool call must produce the correct HTTP request to the target API."""

    async def test_get_tool_routes_to_target_api(self):
        """Calling get_user routes a GET request to the correct target URL."""
        config = make_server_config("server1", OPENAPI_URL1, BASE_URL1, TOKEN1)

        with respx.mock:
            respx.get(OPENAPI_URL1).mock(return_value=httpx.Response(200, json=SPEC1))
            result = await build_mcp_server(config)

        assert result is not None
        mcp, client = result

        try:
            user_response = {"id": "42", "name": "Alice"}

            with respx.mock:
                route = respx.get(f"{BASE_URL1}/users/42").mock(
                    return_value=httpx.Response(200, json=user_response)
                )
                text = await call_tool(mcp, "get_user", {"id": "42"})

            assert route.called, "Target API was not called"
            assert route.calls.last.request.url == f"{BASE_URL1}/users/42"
            assert json.loads(text) == user_response
        finally:
            await client.aclose()

    async def test_tool_result_contains_target_api_response(self):
        """The tool result must carry the exact response body from the target API."""
        config = make_server_config("server2", OPENAPI_URL2, BASE_URL2, TOKEN2)

        with respx.mock:
            respx.get(OPENAPI_URL2).mock(return_value=httpx.Response(200, json=SPEC2))
            result = await build_mcp_server(config)

        assert result is not None
        mcp, client = result

        try:
            items_response = [{"id": 1, "name": "Widget"}, {"id": 2, "name": "Gadget"}]

            with respx.mock:
                respx.get(f"{BASE_URL2}/items").mock(
                    return_value=httpx.Response(200, json=items_response)
                )
                text = await call_tool(mcp, "list_items", {})

            assert json.loads(text) == items_response
        finally:
            await client.aclose()

    async def test_path_param_interpolated_into_url(self):
        """Path parameters must be correctly interpolated into the URL."""
        config = make_server_config("server1", OPENAPI_URL1, BASE_URL1, TOKEN1)

        with respx.mock:
            respx.get(OPENAPI_URL1).mock(return_value=httpx.Response(200, json=SPEC1))
            result = await build_mcp_server(config)

        assert result is not None
        mcp, client = result

        try:
            with respx.mock:
                route = respx.get(f"{BASE_URL1}/users/user-xyz").mock(
                    return_value=httpx.Response(200, json={"id": "user-xyz"})
                )
                await call_tool(mcp, "get_user", {"id": "user-xyz"})

            assert route.called
            assert "/users/user-xyz" in str(route.calls.last.request.url)
        finally:
            await client.aclose()

    async def test_http_error_from_target_api_returned_as_string(self):
        """A 4xx/5xx from the target API is returned as an 'HTTP N: ...' string."""
        config = make_server_config("server1", OPENAPI_URL1, BASE_URL1, TOKEN1)

        with respx.mock:
            respx.get(OPENAPI_URL1).mock(return_value=httpx.Response(200, json=SPEC1))
            result = await build_mcp_server(config)

        assert result is not None
        mcp, client = result

        try:
            with respx.mock:
                respx.get(f"{BASE_URL1}/users/999").mock(
                    return_value=httpx.Response(404, text="not found")
                )
                text = await call_tool(mcp, "get_user", {"id": "999"})

            assert text == "HTTP 404: not found"
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# 4. Auth header forwarded correctly
# ---------------------------------------------------------------------------


class TestAuthHeaderForwarded:
    """The correct auth header must be sent to the target API on every tool call."""

    async def test_bearer_token_forwarded_to_target_api(self):
        """Bearer token from server config must appear in Authorization header."""
        config = make_server_config("server1", OPENAPI_URL1, BASE_URL1, TOKEN1)

        with respx.mock:
            respx.get(OPENAPI_URL1).mock(return_value=httpx.Response(200, json=SPEC1))
            result = await build_mcp_server(config)

        assert result is not None
        mcp, client = result

        try:
            with respx.mock:
                route = respx.get(f"{BASE_URL1}/users/1").mock(
                    return_value=httpx.Response(200, json={"id": "1"})
                )
                await call_tool(mcp, "get_user", {"id": "1"})

            assert route.called
            auth_header = route.calls.last.request.headers.get("authorization", "")
            assert auth_header == f"Bearer {TOKEN1}", (
                f"Expected 'Bearer {TOKEN1}', got {auth_header!r}"
            )
        finally:
            await client.aclose()

    async def test_server2_bearer_token_forwarded(self):
        """server2's distinct bearer token must be used for server2 tool calls."""
        config = make_server_config("server2", OPENAPI_URL2, BASE_URL2, TOKEN2)

        with respx.mock:
            respx.get(OPENAPI_URL2).mock(return_value=httpx.Response(200, json=SPEC2))
            result = await build_mcp_server(config)

        assert result is not None
        mcp, client = result

        try:
            with respx.mock:
                route = respx.get(f"{BASE_URL2}/items").mock(
                    return_value=httpx.Response(200, json=[])
                )
                await call_tool(mcp, "list_items", {})

            assert route.called
            auth_header = route.calls.last.request.headers.get("authorization", "")
            assert auth_header == f"Bearer {TOKEN2}"
        finally:
            await client.aclose()

    async def test_each_server_uses_its_own_token(self):
        """server1 and server2 must each send their own distinct bearer token."""
        config1 = make_server_config("server1", OPENAPI_URL1, BASE_URL1, TOKEN1)
        config2 = make_server_config("server2", OPENAPI_URL2, BASE_URL2, TOKEN2)

        with respx.mock:
            respx.get(OPENAPI_URL1).mock(return_value=httpx.Response(200, json=SPEC1))
            respx.get(OPENAPI_URL2).mock(return_value=httpx.Response(200, json=SPEC2))

            result1 = await build_mcp_server(config1)
            result2 = await build_mcp_server(config2)

        assert result1 is not None
        assert result2 is not None
        mcp1, client1 = result1
        mcp2, client2 = result2

        try:
            with respx.mock:
                route1 = respx.get(f"{BASE_URL1}/users/1").mock(
                    return_value=httpx.Response(200, json={"id": "1"})
                )
                route2 = respx.get(f"{BASE_URL2}/items").mock(
                    return_value=httpx.Response(200, json=[])
                )

                await call_tool(mcp1, "get_user", {"id": "1"})
                await call_tool(mcp2, "list_items", {})

            auth1 = route1.calls.last.request.headers.get("authorization", "")
            auth2 = route2.calls.last.request.headers.get("authorization", "")

            assert auth1 == f"Bearer {TOKEN1}"
            assert auth2 == f"Bearer {TOKEN2}"
            assert auth1 != auth2, "Both servers must use different tokens"
        finally:
            await client1.aclose()
            await client2.aclose()

    async def test_api_key_auth_forwarded(self):
        """API key auth must place the key in the configured header."""
        api_key_spec_url = "http://apikey.test/openapi.json"
        api_key_base = "http://apikey.test"
        api_key_spec = {
            "openapi": "3.0.0",
            "info": {"title": "KeyAPI", "version": "1.0"},
            "paths": {
                "/data": {
                    "get": {
                        "operationId": "get_data",
                        "responses": {"200": {"description": "ok"}},
                    }
                }
            },
        }
        config = ServerConfig(
            name="apikey-server",
            openapi_url=api_key_spec_url,
            base_url=api_key_base,
            auth=ApiKeyAuthConfig(type="api_key", header="X-API-Key", value="secret-key-123"),
        )

        with respx.mock:
            respx.get(api_key_spec_url).mock(return_value=httpx.Response(200, json=api_key_spec))
            result = await build_mcp_server(config)

        assert result is not None
        mcp, client = result

        try:
            with respx.mock:
                route = respx.get(f"{api_key_base}/data").mock(
                    return_value=httpx.Response(200, json={"data": "value"})
                )
                await call_tool(mcp, "get_data", {})

            assert route.called
            api_key_header = route.calls.last.request.headers.get("x-api-key", "")
            assert api_key_header == "secret-key-123"
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# 5. Failed server does not block other servers
# ---------------------------------------------------------------------------


class TestFailedServerDoesNotBlockOthers:
    """When one server fails to initialize, the rest must still be available."""

    def test_failed_server_does_not_prevent_other_servers_from_mounting(self):
        """If server2's OpenAPI fetch fails, server1 must still be mounted and accessible."""
        mod = reload_main()

        config = AppConfig(
            servers={
                "server1": make_server_config("server1", OPENAPI_URL1, BASE_URL1, TOKEN1),
                "server2": make_server_config("server2", OPENAPI_URL2, BASE_URL2, TOKEN2),
            }
        )

        with respx.mock:
            # server1 succeeds; server2 spec URL is unreachable.
            respx.get(OPENAPI_URL1).mock(return_value=httpx.Response(200, json=SPEC1))
            respx.get(OPENAPI_URL2).mock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            with patch.object(mod, "load_config", return_value=config):
                with TestClient(mod.app) as client:
                    # App is alive — health check passes.
                    health_response = client.get("/health")
                    assert health_response.status_code == 200
                    assert health_response.json() == {"status": "ok"}

                    # server1 is mounted.
                    from starlette.routing import Mount
                    mount_paths = {r.path for r in mod.app.routes if isinstance(r, Mount)}
                    assert "/servers/server1" in mount_paths, (
                        "server1 must be mounted even though server2 failed"
                    )

    def test_failed_server_not_mounted(self):
        """The server that failed to build must NOT have an endpoint mounted."""
        mod = reload_main()

        config = AppConfig(
            servers={
                "server1": make_server_config("server1", OPENAPI_URL1, BASE_URL1, TOKEN1),
                "server2": make_server_config("server2", OPENAPI_URL2, BASE_URL2, TOKEN2),
            }
        )

        with respx.mock:
            respx.get(OPENAPI_URL1).mock(return_value=httpx.Response(200, json=SPEC1))
            respx.get(OPENAPI_URL2).mock(return_value=httpx.Response(404, text="not found"))

            with patch.object(mod, "load_config", return_value=config):
                with TestClient(mod.app) as _:
                    from starlette.routing import Mount
                    mount_paths = {r.path for r in mod.app.routes if isinstance(r, Mount)}
                    assert "/servers/server2" not in mount_paths, (
                        "server2 (which failed) must not have a mounted endpoint"
                    )

    async def test_failed_server_does_not_prevent_tool_calls_on_working_server(self):
        """Tools on a working server remain callable even when another server failed."""
        # Build server1 (succeeds); try to build server2 (fails).
        config1 = make_server_config("server1", OPENAPI_URL1, BASE_URL1, TOKEN1)
        config2 = make_server_config("server2", OPENAPI_URL2, BASE_URL2, TOKEN2)

        with respx.mock:
            respx.get(OPENAPI_URL1).mock(return_value=httpx.Response(200, json=SPEC1))
            respx.get(OPENAPI_URL2).mock(return_value=httpx.Response(503, text="unavailable"))

            result1 = await build_mcp_server(config1)
            result2 = await build_mcp_server(config2)

        assert result1 is not None, "server1 must build successfully"
        assert result2 is None, "server2 must fail with a 503 response"

        mcp1, client1 = result1

        try:
            # server1 tools must still work after server2 failed.
            with respx.mock:
                route = respx.get(f"{BASE_URL1}/users/7").mock(
                    return_value=httpx.Response(200, json={"id": "7", "name": "Bob"})
                )
                text = await call_tool(mcp1, "get_user", {"id": "7"})

            assert route.called
            assert json.loads(text) == {"id": "7", "name": "Bob"}
        finally:
            await client1.aclose()


# ---------------------------------------------------------------------------
# 6. Server routing correctness via FastAPI app
# ---------------------------------------------------------------------------


class TestServerRoutingViaApp:
    """Verify that the FastAPI app mounts each server at the correct path."""

    def test_two_servers_mounted_at_distinct_paths(self):
        """server1 and server2 each get their own /servers/{id} mount (sub-app handles /mcp)."""
        mod = reload_main()
        config = make_test_app_config()

        with respx.mock:
            respx.get(OPENAPI_URL1).mock(return_value=httpx.Response(200, json=SPEC1))
            respx.get(OPENAPI_URL2).mock(return_value=httpx.Response(200, json=SPEC2))

            with patch.object(mod, "load_config", return_value=config):
                with TestClient(mod.app) as _:
                    from starlette.routing import Mount
                    mount_paths = {r.path for r in mod.app.routes if isinstance(r, Mount)}

        assert "/servers/server1" in mount_paths
        assert "/servers/server2" in mount_paths

    def test_single_server_config_mounts_one_endpoint(self):
        """A config with one server results in exactly one server mount."""
        mod = reload_main()
        config = make_test_app_config(include_server2=False)

        with respx.mock:
            respx.get(OPENAPI_URL1).mock(return_value=httpx.Response(200, json=SPEC1))

            with patch.object(mod, "load_config", return_value=config):
                with TestClient(mod.app) as _:
                    from starlette.routing import Mount
                    server_mounts = [
                        r for r in mod.app.routes
                        if isinstance(r, Mount) and r.path.startswith("/servers/")
                    ]

        assert len(server_mounts) == 1
        assert server_mounts[0].path == "/servers/server1"
