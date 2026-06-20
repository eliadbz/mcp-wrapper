"""MCP tool registration for mcp-wrapper.

Provides register_tool(), which takes an OperationDef and an httpx.AsyncClient
and registers a live async MCP tool on a FastMCP instance.

When an MCP client calls the tool, the handler makes the real HTTP request to
the target API and returns the response text (or an error string for 4xx/5xx).
"""

from __future__ import annotations

import keyword
import re
from typing import Any

import httpx
from pydantic import ConfigDict, Field, create_model
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase, FuncMetadata

from mcp_wrapper.openapi import OperationDef


def _safe_identifier(name: str) -> str:
    """Convert an arbitrary string into a valid Python identifier.

    Replaces any character that is not alphanumeric or underscore with ``_``,
    then prefixes with ``p_`` if the result starts with a digit, an underscore,
    or is a Python keyword.  The leading-underscore guard prevents FastMCP from
    rejecting the parameter name (FastMCP raises InvalidSignature for any
    parameter whose name begins with ``_``).

    Args:
        name: The original parameter name (may contain hyphens, dots, etc.).

    Returns:
        A string that is a valid Python identifier and does not start with ``_``.
    """
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if not safe or safe[0].isdigit() or safe[0] == "_" or keyword.iskeyword(safe):
        safe = f"p_{safe}"
    return safe


def _build_safe_mapping(param_names: list[str]) -> list[tuple[str, str]]:
    """Return a deduplicated list of (safe_python_name, original_name) pairs.

    For each original name, ``_safe_identifier`` produces a Python-safe alias.
    If two names produce the same alias (e.g. ``x-foo`` and ``x_foo`` both
    become ``x_foo``), the later duplicates are suffixed with ``_1``, ``_2``,
    etc. to keep each alias unique.

    Args:
        param_names: Ordered list of original parameter names.

    Returns:
        A list of ``(safe_alias, original_name)`` tuples in the same order,
        with safe aliases guaranteed to be unique across the list.
    """
    raw: list[tuple[str, str]] = [(_safe_identifier(p), p) for p in param_names]

    seen: dict[str, int] = {}
    deduped: list[tuple[str, str]] = []
    for safe, orig in raw:
        if safe in seen:
            seen[safe] += 1
            safe = f"{safe}_{seen[safe]}"
        else:
            seen[safe] = 0
        deduped.append((safe, orig))

    return deduped


def _build_input_schema(operation: OperationDef) -> dict:
    """Build a JSON Schema dict for the given OperationDef.

    Combines path params, query params, and body into a single object schema
    that describes what arguments the tool handler accepts.

    The request body, if present, is represented under the key ``"__body__"``.
    This sentinel cannot appear as a valid OpenAPI parameter name (OpenAPI
    parameter names may not use dunder-style names), so it never collides with
    real path or query params.
    """
    properties: dict = {}
    required: list[str] = []

    for p in operation.path_params + operation.query_params:
        properties[p.name] = p.schema
        if p.required:
            required.append(p.name)

    if operation.body_schema is not None:
        properties["__body__"] = operation.body_schema
        required.append("__body__")

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
    safe_to_original: list[tuple[str, str]],
) -> Any:
    """Build an async handler function with an explicit named-param signature.

    FastMCP derives the input schema from the function's type annotations.
    Using ``exec`` to build a function with the exact parameter names expected
    by the OpenAPI operation ensures FastMCP generates a correct schema, and
    that its argument validation model accepts (and passes through) those params.

    Parameter names from OpenAPI specs may contain characters that are illegal
    in Python identifiers (e.g. hyphens, dots). The ``safe_to_original`` mapping
    (built by ``_build_safe_mapping``) provides unique, FastMCP-safe aliases;
    the exec'd handler translates them back to original names when calling
    ``_core``.

    Args:
        client: The shared httpx.AsyncClient to use for requests.
        method: Uppercase HTTP method string (e.g. "GET").
        path_template: URL path template with ``{param}`` placeholders.
        path_param_names: Set of parameter names that belong in the path.
        query_param_names: Set of parameter names that belong in the query string.
        has_body: Whether to treat a ``__body__`` kwarg as the JSON request body.
        safe_to_original: Ordered list of ``(safe_alias, original_name)`` pairs
            produced by ``_build_safe_mapping``.  The body sentinel, if present,
            must have original name ``"__body__"``.

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
            elif key == "__body__" and has_body:
                body_value = value

        url = path_template.format_map(path_dict)

        response: httpx.Response = await client.request(
            method=method,
            url=url,
            params=({k: v for k, v in query_dict.items() if v is not None} or None),
            json=body_value,
        )

        if response.is_success:
            return response.text

        return f"HTTP {response.status_code}: {response.text}"

    # Build a wrapper with an explicit named-parameter signature so that
    # FastMCP's argument validation model accepts the individual param names
    # rather than a single ``**kwargs`` field.
    if safe_to_original:
        param_list = ", ".join(f"{safe}=None" for safe, _ in safe_to_original)
        kwargs_build = (
            "{"
            + ", ".join(f"{orig!r}: {safe}" for safe, orig in safe_to_original)
            + "}"
        )
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
    readonly: bool = False,
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

    # Collect all parameter names in a stable order: path → query → __body__.
    # "__body__" is used as the body sentinel because dunder-style names cannot
    # appear as valid OpenAPI parameter names, preventing any collision with real
    # path or query params.  _safe_identifier converts "__body__" to "p___body__"
    # (no leading underscore) so FastMCP's InvalidSignature guard is satisfied.
    all_param_names: list[str] = (
        [p.name for p in operation.path_params]
        + [p.name for p in operation.query_params]
        + (["__body__"] if has_body else [])
    )

    # Build the deduplicated safe-name mapping once; share it with both
    # _make_handler (for the exec'd signature) and the alias-patch block below.
    safe_to_original = _build_safe_mapping(all_param_names)

    handler = _make_handler(
        client=client,
        method=operation.method.upper(),
        path_template=operation.path,
        path_param_names=path_param_names,
        query_param_names=query_param_names,
        has_body=has_body,
        safe_to_original=safe_to_original,
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

        # When any param name differs from its safe alias (e.g. "x-api-version"
        # → "x_api_version", or "__body__" → "p___body__"), FastMCP's MCP
        # clients will pass args keyed by the original name.  Rebuild the
        # arg_model with proper Pydantic Field aliases so validation correctly
        # maps original names to the safe alias fields used by _handler.
        needs_alias_patch = any(safe != orig for safe, orig in safe_to_original)
        if needs_alias_patch:
            patched_fields: dict[str, Any] = {}
            for safe, orig in safe_to_original:
                if safe != orig:
                    # validation_alias lets Pydantic accept the original name
                    # when validating incoming args; the field name (safe alias)
                    # is used by model_dump_one_level so _handler receives valid
                    # Python keyword arguments.
                    patched_fields[safe] = (
                        Any,
                        Field(default=None, validation_alias=orig),
                    )
                else:
                    patched_fields[safe] = (Any, Field(default=None))
            new_arg_model = create_model(
                "_handlerArguments",
                __base__=ArgModelBase,
                __config__=ConfigDict(populate_by_name=True),
                **patched_fields,
            )
            new_fn_metadata = FuncMetadata(arg_model=new_arg_model)
            object.__setattr__(registered_tool, "fn_metadata", new_fn_metadata)
