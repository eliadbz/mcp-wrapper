"""MCP tool registration for mcp-wrapper.

Provides register_tool(), which takes an OperationDef and an httpx.AsyncClient
and registers a live async MCP tool on a FastMCP instance.

When an MCP client calls the tool, the handler makes the real HTTP request to
the target API and returns the response text (or an error string for 4xx/5xx).
"""

from __future__ import annotations

from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

from mcp_wrapper.openapi import OperationDef


def _build_input_schema(operation: OperationDef) -> dict:
    """Build a JSON Schema dict for the given OperationDef.

    Combines path params, query params, and body into a single object schema
    that describes what arguments the tool handler accepts.
    """
    properties: dict = {}
    required: list[str] = []

    for p in operation.path_params + operation.query_params:
        properties[p.name] = p.schema
        if p.required:
            required.append(p.name)

    if operation.body_schema is not None:
        properties["body"] = operation.body_schema
        required.append("body")

    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _make_handler(
    client: httpx.AsyncClient,
    method: str,
    path_template: str,
    path_param_names: frozenset,
    query_param_names: frozenset,
    has_body: bool,
    all_param_names: list[str],
) -> Any:
    """Build an async handler function with an explicit named-param signature.

    FastMCP derives the input schema from the function's type annotations.
    Using ``exec`` to build a function with the exact parameter names expected
    by the OpenAPI operation ensures FastMCP generates a correct schema, and
    that its argument validation model accepts (and passes through) those params.

    Args:
        client: The shared httpx.AsyncClient to use for requests.
        method: Uppercase HTTP method string (e.g. "GET").
        path_template: URL path template with ``{param}`` placeholders.
        path_param_names: Set of parameter names that belong in the path.
        query_param_names: Set of parameter names that belong in the query string.
        has_body: Whether to treat a ``body`` kwarg as the JSON request body.
        all_param_names: Ordered list of all parameter names for the signature.

    Returns:
        An async function suitable for passing to ``mcp.add_tool()``.
    """

    async def _core(**kwargs: Any) -> str:
        path_dict: dict = {}
        query_dict: dict = {}
        body_value = None

        for key, value in kwargs.items():
            if key in path_param_names:
                path_dict[key] = value
            elif key in query_param_names:
                query_dict[key] = value
            elif key == "body" and has_body:
                body_value = value

        url = path_template.format_map(path_dict)

        response: httpx.Response = await client.request(
            method=method,
            url=url,
            params=query_dict if query_dict else None,
            json=body_value,
        )

        if response.is_success:
            return response.text

        return f"HTTP {response.status_code}: {response.text}"

    # Build a wrapper with an explicit named-parameter signature so that
    # FastMCP's argument validation model accepts the individual param names
    # rather than a single ``**kwargs`` field.
    if all_param_names:
        param_list = ", ".join(f"{p}=None" for p in all_param_names)
        kwargs_build = "{" + ", ".join(f"{p!r}: {p}" for p in all_param_names) + "}"
        func_code = (
            f"async def _handler({param_list}):\n"
            f"    return await _core(**{kwargs_build})\n"
        )
    else:
        func_code = "async def _handler():\n    return await _core()\n"

    globs: dict = {"_core": _core}
    exec(func_code, globs)  # noqa: S102 — intentional dynamic dispatch
    return globs["_handler"]


def register_tool(
    mcp: FastMCP,
    operation: OperationDef,
    client: httpx.AsyncClient,
) -> None:
    """Register one MCP tool on the given FastMCP instance for the given operation.

    The registered tool, when called by an MCP client, will:
    1. Separate kwargs into path params, query params, and request body.
    2. Interpolate path params into the URL template.
    3. Make the HTTP request via the provided client.
    4. Return the response text on success (2xx), or an "HTTP {code}: {body}"
       string on 4xx/5xx — never raising exceptions for HTTP errors.

    Args:
        mcp: The FastMCP instance to register the tool on.
        operation: The parsed OpenAPI operation to expose as a tool.
        client: An authenticated httpx.AsyncClient; caller owns its lifecycle.
    """
    path_param_names = frozenset(p.name for p in operation.path_params)
    query_param_names = frozenset(p.name for p in operation.query_params)
    has_body = operation.body_schema is not None

    # Collect all parameter names in a stable order: path → query → body.
    all_param_names: list[str] = (
        [p.name for p in operation.path_params]
        + [p.name for p in operation.query_params]
        + (["body"] if has_body else [])
    )

    handler = _make_handler(
        client=client,
        method=operation.method.upper(),
        path_template=operation.path,
        path_param_names=path_param_names,
        query_param_names=query_param_names,
        has_body=has_body,
        all_param_names=all_param_names,
    )

    # Register using FastMCP's add_tool (Option A from the brief).
    # The function's explicit parameter signature drives schema generation.
    # structured_output=False ensures call_tool always returns a list of
    # ContentBlock rather than a structured dict, which simplifies test helpers.
    mcp.add_tool(
        handler,
        name=operation.tool_name,
        description=operation.description,
        structured_output=False,
    )

    # Patch the registered tool's parameters schema to reflect the precise
    # OpenAPI-derived schema (correct types, required fields) rather than the
    # generic ``{type: string, default: None}`` schema FastMCP infers from the
    # ``=None`` defaults in the exec'd function signature.
    registered_tool = mcp._tool_manager.get_tool(operation.tool_name)
    if registered_tool is not None:
        input_schema = _build_input_schema(operation)
        # Tool is a Pydantic v2 model; use object.__setattr__ to bypass the
        # immutability guard and inject the custom schema.
        object.__setattr__(registered_tool, "parameters", input_schema)
