"""Tests for mcp_wrapper.tools — register_tool() function.

Uses respx to mock HTTP calls and a real FastMCP instance to validate both
tool registration behaviour and the tool handler's HTTP request logic.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from mcp.server.fastmcp import FastMCP

from mcp_wrapper.openapi import OperationDef, ParamDef
from mcp_wrapper.tools import register_tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_URL = "https://api.example.com"


def make_mcp() -> FastMCP:
    """Return a fresh FastMCP instance with duplicate-tool warnings disabled."""
    return FastMCP("test-server", warn_on_duplicate_tools=False)


def make_client() -> httpx.AsyncClient:
    """Return an AsyncClient pointed at the mock base URL."""
    return httpx.AsyncClient(base_url=BASE_URL)


async def call(mcp: FastMCP, tool_name: str, args: dict) -> str:
    """Call a registered tool and return its text result."""
    results = await mcp.call_tool(tool_name, args)
    # call_tool returns a sequence of ContentBlock objects (or a dict for structured output).
    # We registered all tools with structured_output=False so we always get a list.
    if isinstance(results, list) and results:
        first = results[0]
        # TextContent has a .text attribute.
        return first.text if hasattr(first, "text") else str(first)
    return str(results)


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestRegisterToolRegistration:
    """Verify that register_tool registers a tool with the correct metadata."""

    def test_tool_registered_with_correct_name(self):
        """register_tool should add a tool whose name matches operation.tool_name."""
        mcp = make_mcp()
        client = httpx.AsyncClient(base_url=BASE_URL)
        operation = OperationDef(
            tool_name="get_users",
            method="get",
            path="/users",
            description="List users",
            path_params=[],
            query_params=[],
            body_schema=None,
        )

        register_tool(mcp, operation, client)

        tool = mcp._tool_manager.get_tool("get_users")
        assert tool is not None
        assert tool.name == "get_users"

    def test_tool_registered_with_correct_description(self):
        """The registered tool should carry the operation's description."""
        mcp = make_mcp()
        client = httpx.AsyncClient(base_url=BASE_URL)
        operation = OperationDef(
            tool_name="list_items",
            method="get",
            path="/items",
            description="Retrieve all items",
            path_params=[],
            query_params=[],
            body_schema=None,
        )

        register_tool(mcp, operation, client)

        tool = mcp._tool_manager.get_tool("list_items")
        assert tool is not None
        assert tool.description == "Retrieve all items"

    def test_tool_appears_in_list_tools(self):
        """After registration the tool should appear in mcp.list_tools()."""
        mcp = make_mcp()
        client = httpx.AsyncClient(base_url=BASE_URL)
        operation = OperationDef(
            tool_name="ping",
            method="get",
            path="/ping",
            description="Ping endpoint",
            path_params=[],
            query_params=[],
            body_schema=None,
        )

        register_tool(mcp, operation, client)

        # list_tools is async but we can check the internal dict directly.
        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "ping" in tool_names

    def test_input_schema_contains_path_params(self):
        """The registered tool's inputSchema should include path params."""
        mcp = make_mcp()
        client = httpx.AsyncClient(base_url=BASE_URL)
        operation = OperationDef(
            tool_name="get_user",
            method="get",
            path="/users/{user_id}",
            description="Get a user",
            path_params=[ParamDef(name="user_id", required=True, schema={"type": "string"}, description="")],
            query_params=[],
            body_schema=None,
        )

        register_tool(mcp, operation, client)

        tool = mcp._tool_manager.get_tool("get_user")
        assert tool is not None
        assert "user_id" in tool.parameters.get("properties", {})
        assert "user_id" in tool.parameters.get("required", [])

    def test_input_schema_contains_query_params(self):
        """The registered tool's inputSchema should include query params."""
        mcp = make_mcp()
        client = httpx.AsyncClient(base_url=BASE_URL)
        operation = OperationDef(
            tool_name="search_items",
            method="get",
            path="/items",
            description="Search items",
            path_params=[],
            query_params=[ParamDef(name="q", required=False, schema={"type": "string"}, description="")],
            body_schema=None,
        )

        register_tool(mcp, operation, client)

        tool = mcp._tool_manager.get_tool("search_items")
        assert tool is not None
        assert "q" in tool.parameters.get("properties", {})

    def test_input_schema_contains_body_when_present(self):
        """The registered tool's inputSchema should include '__body__' when operation has a body."""
        mcp = make_mcp()
        client = httpx.AsyncClient(base_url=BASE_URL)
        body_schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        operation = OperationDef(
            tool_name="create_item",
            method="post",
            path="/items",
            description="Create an item",
            path_params=[],
            query_params=[],
            body_schema=body_schema,
        )

        register_tool(mcp, operation, client)

        tool = mcp._tool_manager.get_tool("create_item")
        assert tool is not None
        assert "__body__" in tool.parameters.get("properties", {})
        assert "__body__" in tool.parameters.get("required", [])


# ---------------------------------------------------------------------------
# Handler behaviour tests — require an async client and respx mocking
# ---------------------------------------------------------------------------


class TestRegisterToolHandlerGet:
    """Verify the handler makes GET requests correctly."""

    @pytest.mark.asyncio
    async def test_get_no_params_returns_response_text(self):
        """A GET with no params should return the response body as a string."""
        mcp = make_mcp()
        async with make_client() as client:
            operation = OperationDef(
                tool_name="list_users",
                method="get",
                path="/users",
                description="List users",
                path_params=[],
                query_params=[],
                body_schema=None,
            )
            register_tool(mcp, operation, client)

            with respx.mock:
                respx.get(f"{BASE_URL}/users").mock(
                    return_value=httpx.Response(200, text='[{"id":1}]')
                )
                result = await call(mcp, "list_users", {})

        assert result == '[{"id":1}]'

    @pytest.mark.asyncio
    async def test_get_with_path_param_interpolates_url(self):
        """Path param should be interpolated into the URL template."""
        mcp = make_mcp()
        async with make_client() as client:
            operation = OperationDef(
                tool_name="get_user",
                method="get",
                path="/users/{user_id}",
                description="Get user",
                path_params=[ParamDef(name="user_id", required=True, schema={"type": "string"}, description="")],
                query_params=[],
                body_schema=None,
            )
            register_tool(mcp, operation, client)

            with respx.mock:
                route = respx.get(f"{BASE_URL}/users/42").mock(
                    return_value=httpx.Response(200, text='{"id":42}')
                )
                result = await call(mcp, "get_user", {"user_id": "42"})

        assert route.called
        assert route.calls.last.request.url == f"{BASE_URL}/users/42"
        assert result == '{"id":42}'

    @pytest.mark.asyncio
    async def test_get_with_query_params_passes_them(self):
        """Query params should be forwarded as URL query string."""
        mcp = make_mcp()
        async with make_client() as client:
            operation = OperationDef(
                tool_name="search_users",
                method="get",
                path="/users",
                description="Search users",
                path_params=[],
                query_params=[
                    ParamDef(name="q", required=False, schema={"type": "string"}, description=""),
                    ParamDef(name="limit", required=False, schema={"type": "integer"}, description=""),
                ],
                body_schema=None,
            )
            register_tool(mcp, operation, client)

            with respx.mock:
                route = respx.get(f"{BASE_URL}/users").mock(
                    return_value=httpx.Response(200, text='[]')
                )
                result = await call(mcp, "search_users", {"q": "alice", "limit": "10"})

        assert route.called
        url_str = str(route.calls.last.request.url)
        assert "q=alice" in url_str
        assert "limit=10" in url_str
        assert result == "[]"

    @pytest.mark.asyncio
    async def test_get_with_path_and_query_params(self):
        """Both path and query params should be correctly placed."""
        mcp = make_mcp()
        async with make_client() as client:
            operation = OperationDef(
                tool_name="get_user_posts",
                method="get",
                path="/users/{user_id}/posts",
                description="Get user posts",
                path_params=[ParamDef(name="user_id", required=True, schema={"type": "string"}, description="")],
                query_params=[ParamDef(name="page", required=False, schema={"type": "integer"}, description="")],
                body_schema=None,
            )
            register_tool(mcp, operation, client)

            with respx.mock:
                route = respx.get(f"{BASE_URL}/users/7/posts").mock(
                    return_value=httpx.Response(200, text='{"posts":[]}')
                )
                result = await call(mcp, "get_user_posts", {"user_id": "7", "page": "2"})

        assert route.called
        url_str = str(route.calls.last.request.url)
        assert "/users/7/posts" in url_str
        assert "page=2" in url_str
        assert result == '{"posts":[]}'


class TestRegisterToolHandlerPost:
    """Verify the handler sends POST requests with a body."""

    @pytest.mark.asyncio
    async def test_post_with_body_sends_json(self):
        """POST with a body should send JSON payload."""
        mcp = make_mcp()
        async with make_client() as client:
            operation = OperationDef(
                tool_name="create_user",
                method="post",
                path="/users",
                description="Create user",
                path_params=[],
                query_params=[],
                body_schema={"type": "object", "properties": {"name": {"type": "string"}}},
            )
            register_tool(mcp, operation, client)

            with respx.mock:
                route = respx.post(f"{BASE_URL}/users").mock(
                    return_value=httpx.Response(201, text='{"id":99}')
                )
                result = await call(mcp, "create_user", {"__body__": {"name": "Alice"}})

        assert route.called
        import json
        sent_body = json.loads(route.calls.last.request.content)
        assert sent_body == {"name": "Alice"}
        assert result == '{"id":99}'

    @pytest.mark.asyncio
    async def test_post_with_path_param_and_body(self):
        """POST with both a path param and a body should use both correctly."""
        mcp = make_mcp()
        async with make_client() as client:
            operation = OperationDef(
                tool_name="update_user",
                method="put",
                path="/users/{id}",
                description="Update user",
                path_params=[ParamDef(name="id", required=True, schema={"type": "string"}, description="")],
                query_params=[],
                body_schema={"type": "object"},
            )
            register_tool(mcp, operation, client)

            with respx.mock:
                route = respx.put(f"{BASE_URL}/users/5").mock(
                    return_value=httpx.Response(200, text='{"updated":true}')
                )
                result = await call(mcp, "update_user", {"id": "5", "__body__": {"name": "Bob"}})

        assert route.called
        assert "/users/5" in str(route.calls.last.request.url)
        assert result == '{"updated":true}'


class TestRegisterToolHandlerErrors:
    """Verify that HTTP errors are returned as strings rather than raised."""

    @pytest.mark.asyncio
    async def test_404_returned_as_error_string(self):
        """A 404 response should be returned as 'HTTP 404: ...' not raised."""
        mcp = make_mcp()
        async with make_client() as client:
            operation = OperationDef(
                tool_name="get_missing",
                method="get",
                path="/missing",
                description="Missing resource",
                path_params=[],
                query_params=[],
                body_schema=None,
            )
            register_tool(mcp, operation, client)

            with respx.mock:
                respx.get(f"{BASE_URL}/missing").mock(
                    return_value=httpx.Response(404, text="not found")
                )
                result = await call(mcp, "get_missing", {})

        assert result == "HTTP 404: not found"

    @pytest.mark.asyncio
    async def test_400_returned_as_error_string(self):
        """A 400 response should be returned as an error string."""
        mcp = make_mcp()
        async with make_client() as client:
            operation = OperationDef(
                tool_name="bad_request",
                method="post",
                path="/items",
                description="Bad request endpoint",
                path_params=[],
                query_params=[],
                body_schema={"type": "object"},
            )
            register_tool(mcp, operation, client)

            with respx.mock:
                respx.post(f"{BASE_URL}/items").mock(
                    return_value=httpx.Response(400, text="invalid input")
                )
                result = await call(mcp, "bad_request", {"__body__": {}})

        assert result == "HTTP 400: invalid input"

    @pytest.mark.asyncio
    async def test_500_returned_as_error_string(self):
        """A 500 response should be returned as an error string."""
        mcp = make_mcp()
        async with make_client() as client:
            operation = OperationDef(
                tool_name="server_error",
                method="get",
                path="/boom",
                description="Server error endpoint",
                path_params=[],
                query_params=[],
                body_schema=None,
            )
            register_tool(mcp, operation, client)

            with respx.mock:
                respx.get(f"{BASE_URL}/boom").mock(
                    return_value=httpx.Response(500, text="internal server error")
                )
                result = await call(mcp, "server_error", {})

        assert result == "HTTP 500: internal server error"

    @pytest.mark.asyncio
    async def test_503_returned_as_error_string(self):
        """A 503 response should be returned as an error string."""
        mcp = make_mcp()
        async with make_client() as client:
            operation = OperationDef(
                tool_name="unavailable",
                method="get",
                path="/unavailable",
                description="Service unavailable",
                path_params=[],
                query_params=[],
                body_schema=None,
            )
            register_tool(mcp, operation, client)

            with respx.mock:
                respx.get(f"{BASE_URL}/unavailable").mock(
                    return_value=httpx.Response(503, text="service unavailable")
                )
                result = await call(mcp, "unavailable", {})

        assert result == "HTTP 503: service unavailable"


class TestRegisterToolMultipleOperations:
    """Test registering multiple tools on the same FastMCP instance."""

    def test_multiple_tools_registered_independently(self):
        """Each operation should register as a separate, distinct tool."""
        mcp = make_mcp()
        client = httpx.AsyncClient(base_url=BASE_URL)

        operations = [
            OperationDef(
                tool_name="op_one",
                method="get",
                path="/one",
                description="First op",
                path_params=[],
                query_params=[],
                body_schema=None,
            ),
            OperationDef(
                tool_name="op_two",
                method="post",
                path="/two",
                description="Second op",
                path_params=[],
                query_params=[],
                body_schema={"type": "object"},
            ),
        ]

        for op in operations:
            register_tool(mcp, op, client)

        tool_names = {t.name for t in mcp._tool_manager.list_tools()}
        assert "op_one" in tool_names
        assert "op_two" in tool_names


class TestRegisterToolClientNotClosed:
    """Verify the handler does not close the shared client."""

    @pytest.mark.asyncio
    async def test_client_still_usable_after_tool_call(self):
        """The client must remain open after a tool call (caller owns lifecycle)."""
        mcp = make_mcp()
        client = make_client()
        operation = OperationDef(
            tool_name="healthz",
            method="get",
            path="/healthz",
            description="Health check",
            path_params=[],
            query_params=[],
            body_schema=None,
        )
        register_tool(mcp, operation, client)

        with respx.mock:
            respx.get(f"{BASE_URL}/healthz").mock(return_value=httpx.Response(200, text="ok"))
            await call(mcp, "healthz", {})

        # Client should still be usable (is_closed reflects transport state).
        assert not client.is_closed
        await client.aclose()


# ---------------------------------------------------------------------------
# Regression tests — Bug 1 and Bug 2
# ---------------------------------------------------------------------------


class TestHyphenatedParamName:
    """Regression: param names with hyphens (e.g. x-api-version) must not
    crash exec and must be forwarded correctly in the HTTP request."""

    def test_hyphenated_param_registers_without_error(self):
        """register_tool must not raise SyntaxError for a hyphenated param name."""
        mcp = make_mcp()
        client = httpx.AsyncClient(base_url=BASE_URL)
        operation = OperationDef(
            tool_name="versioned_get",
            method="get",
            path="/versioned",
            description="Versioned endpoint",
            path_params=[],
            query_params=[
                ParamDef(
                    name="x-api-version",
                    required=True,
                    schema={"type": "string"},
                    description="API version header-style query param",
                )
            ],
            body_schema=None,
        )
        # Must not raise SyntaxError or any other exception.
        register_tool(mcp, operation, client)

        tool = mcp._tool_manager.get_tool("versioned_get")
        assert tool is not None
        # The input schema should preserve the original param name as the key.
        assert "x-api-version" in tool.parameters.get("properties", {})

    @pytest.mark.asyncio
    async def test_hyphenated_param_forwarded_in_request(self):
        """The hyphenated query param value must appear in the actual HTTP request."""
        mcp = make_mcp()
        async with make_client() as client:
            operation = OperationDef(
                tool_name="versioned_get2",
                method="get",
                path="/versioned",
                description="Versioned endpoint",
                path_params=[],
                query_params=[
                    ParamDef(
                        name="x-api-version",
                        required=True,
                        schema={"type": "string"},
                        description="API version query param",
                    )
                ],
                body_schema=None,
            )
            register_tool(mcp, operation, client)

            with respx.mock:
                route = respx.get(f"{BASE_URL}/versioned").mock(
                    return_value=httpx.Response(200, text="ok")
                )
                result = await call(mcp, "versioned_get2", {"x-api-version": "2024-01"})

        assert route.called
        url_str = str(route.calls.last.request.url)
        assert "x-api-version=2024-01" in url_str
        assert result == "ok"


class TestQueryParamNamedRequestBody:
    """Regression: a query param named 'request_body' with no body schema must
    be forwarded as a query string parameter and not silently dropped."""

    @pytest.mark.asyncio
    async def test_query_param_named_request_body_forwarded(self):
        """A query param literally named 'request_body' must reach the server."""
        mcp = make_mcp()
        async with make_client() as client:
            operation = OperationDef(
                tool_name="search_with_request_body_param",
                method="get",
                path="/search",
                description="Search endpoint with an unusual param name",
                path_params=[],
                query_params=[
                    ParamDef(
                        name="request_body",
                        required=False,
                        schema={"type": "string"},
                        description="Unusual param name that was previously a collision",
                    )
                ],
                body_schema=None,  # No body — has_body is False.
            )
            register_tool(mcp, operation, client)

            with respx.mock:
                route = respx.get(f"{BASE_URL}/search").mock(
                    return_value=httpx.Response(200, text='{"results":[]}')
                )
                result = await call(
                    mcp,
                    "search_with_request_body_param",
                    {"request_body": "my-filter"},
                )

        assert route.called
        url_str = str(route.calls.last.request.url)
        # The param must appear in the query string, not be silently dropped.
        assert "request_body=my-filter" in url_str
        assert result == '{"results":[]}'


# ---------------------------------------------------------------------------
# Regression tests — Fix 2 (safe-name collisions and __body__ sentinel)
# ---------------------------------------------------------------------------


class TestSafeNameCollision:
    """Regression: two params that sanitize to the same safe identifier must
    not produce a SyntaxError and both values must be forwarded correctly."""

    def test_collision_registers_without_crash(self):
        """x-foo and x_foo both sanitize to x_foo; register_tool must not raise."""
        mcp = make_mcp()
        client = httpx.AsyncClient(base_url=BASE_URL)
        operation = OperationDef(
            tool_name="collision_op",
            method="get",
            path="/collision",
            description="Operation with colliding safe param names",
            path_params=[],
            query_params=[
                ParamDef(
                    name="x-foo",
                    required=False,
                    schema={"type": "string"},
                    description="Hyphenated param that sanitizes to x_foo",
                ),
                ParamDef(
                    name="x_foo",
                    required=False,
                    schema={"type": "string"},
                    description="Param already named x_foo — collides with x-foo after sanitization",
                ),
            ],
            body_schema=None,
        )
        # Must not raise SyntaxError even though both names sanitize to x_foo.
        register_tool(mcp, operation, client)

        tool = mcp._tool_manager.get_tool("collision_op")
        assert tool is not None
        # Both original param names must appear in the schema properties.
        props = tool.parameters.get("properties", {})
        assert "x-foo" in props
        assert "x_foo" in props

    @pytest.mark.asyncio
    async def test_collision_both_params_forwarded(self):
        """Both x-foo and x_foo must be sent as separate query string parameters."""
        mcp = make_mcp()
        async with make_client() as client:
            operation = OperationDef(
                tool_name="collision_op2",
                method="get",
                path="/collision",
                description="Operation with colliding safe param names",
                path_params=[],
                query_params=[
                    ParamDef(
                        name="x-foo",
                        required=False,
                        schema={"type": "string"},
                        description="Hyphenated param",
                    ),
                    ParamDef(
                        name="x_foo",
                        required=False,
                        schema={"type": "string"},
                        description="Underscore param",
                    ),
                ],
                body_schema=None,
            )
            register_tool(mcp, operation, client)

            with respx.mock:
                route = respx.get(f"{BASE_URL}/collision").mock(
                    return_value=httpx.Response(200, text="ok")
                )
                result = await call(
                    mcp, "collision_op2", {"x-foo": "hyphen-value", "x_foo": "underscore-value"}
                )

        assert route.called
        url_str = str(route.calls.last.request.url)
        assert "x-foo=hyphen-value" in url_str
        assert "x_foo=underscore-value" in url_str
        assert result == "ok"


class TestQueryParamNamedRequestBodyWithBodySchema:
    """Regression: a query param named 'request_body' combined with a body_schema
    must not produce a SyntaxError (duplicate 'request_body' in all_param_names)
    and both must be forwarded correctly via the __body__ sentinel fix."""

    def test_request_body_param_with_body_schema_registers_without_crash(self):
        """register_tool must not raise when a query param is named 'request_body'
        and body_schema is also present (previously caused duplicate param name)."""
        mcp = make_mcp()
        client = httpx.AsyncClient(base_url=BASE_URL)
        operation = OperationDef(
            tool_name="dual_body_op",
            method="post",
            path="/dual",
            description="Operation with both a request_body query param and a body schema",
            path_params=[],
            query_params=[
                ParamDef(
                    name="request_body",
                    required=False,
                    schema={"type": "string"},
                    description="Query param coincidentally named request_body",
                ),
            ],
            body_schema={"type": "object", "properties": {"key": {"type": "string"}}},
        )
        # Must not raise SyntaxError — the sentinel is now __body__, not request_body.
        register_tool(mcp, operation, client)

        tool = mcp._tool_manager.get_tool("dual_body_op")
        assert tool is not None
        props = tool.parameters.get("properties", {})
        assert "request_body" in props   # the query param
        assert "__body__" in props       # the body sentinel

    @pytest.mark.asyncio
    async def test_request_body_param_and_body_both_forwarded(self):
        """The query param 'request_body' and the JSON body must both be sent."""
        mcp = make_mcp()
        async with make_client() as client:
            operation = OperationDef(
                tool_name="dual_body_op2",
                method="post",
                path="/dual",
                description="Operation with request_body query param and body schema",
                path_params=[],
                query_params=[
                    ParamDef(
                        name="request_body",
                        required=False,
                        schema={"type": "string"},
                        description="Query param named request_body",
                    ),
                ],
                body_schema={"type": "object", "properties": {"key": {"type": "string"}}},
            )
            register_tool(mcp, operation, client)

            with respx.mock:
                route = respx.post(f"{BASE_URL}/dual").mock(
                    return_value=httpx.Response(200, text='{"ok":true}')
                )
                result = await call(
                    mcp,
                    "dual_body_op2",
                    {"request_body": "qparam-value", "__body__": {"key": "body-value"}},
                )

        import json
        assert route.called
        url_str = str(route.calls.last.request.url)
        assert "request_body=qparam-value" in url_str
        sent_body = json.loads(route.calls.last.request.content)
        assert sent_body == {"key": "body-value"}
        assert result == '{"ok":true}'


# ---------------------------------------------------------------------------
# Regression test — Fix 1: optional None query params must not appear in URL
# ---------------------------------------------------------------------------


class TestOptionalQueryParamOmittedWhenNone:
    """Regression: omitting an optional query param must not send ?param= in the URL.

    When an MCP caller supplies only the required path param and omits the
    optional query param, the handler receives None for the optional kwarg.
    The fix filters None values from query_dict so httpx never serializes them
    as empty strings in the query string.
    """

    @pytest.mark.asyncio
    async def test_omitted_optional_query_param_not_in_url(self):
        """Omitting an optional query param must produce a URL with no query string."""
        mcp = make_mcp()
        async with make_client() as client:
            operation = OperationDef(
                tool_name="get_item_optional_page",
                method="get",
                path="/items/{item_id}",
                description="Get an item, optionally filtered by page",
                path_params=[
                    ParamDef(
                        name="item_id",
                        required=True,
                        schema={"type": "string"},
                        description="The item identifier",
                    )
                ],
                query_params=[
                    ParamDef(
                        name="page",
                        required=False,
                        schema={"type": "integer"},
                        description="Optional page number",
                    )
                ],
                body_schema=None,
            )
            register_tool(mcp, operation, client)

            with respx.mock:
                route = respx.get(f"{BASE_URL}/items/123").mock(
                    return_value=httpx.Response(200, text='{"id":"123"}')
                )
                # Call without the optional 'page' query param.
                result = await call(mcp, "get_item_optional_page", {"item_id": "123"})

        assert route.called
        url_str = str(route.calls.last.request.url)
        # The optional param must NOT be present in the URL at all.
        assert "page" not in url_str
        assert "?" not in url_str
        assert result == '{"id":"123"}'
