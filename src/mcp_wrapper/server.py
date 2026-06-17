"""MCP server builder for mcp-wrapper.

Provides build_mcp_server(), which creates an isolated FastMCP instance for a
given ServerConfig, fetches and parses the target server's OpenAPI spec, and
registers all operations as MCP tools.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from mcp.server.fastmcp import FastMCP

from mcp_wrapper.config import ServerConfig
from mcp_wrapper.http_client import build_client
from mcp_wrapper.openapi import OperationDef, fetch_spec, parse_operations
from mcp_wrapper.tools import register_tool

logger = logging.getLogger(__name__)


async def build_mcp_server(
    server_config: ServerConfig,
) -> tuple[FastMCP, httpx.AsyncClient] | None:
    """Build a fully configured FastMCP instance for the given server.

    Steps:
    1. Create a FastMCP instance named after the server.
    2. Build an authenticated httpx.AsyncClient via build_client().
    3. Fetch the OpenAPI spec from server_config.openapi_url (in a thread, to
       avoid blocking the event loop with the sync httpx.get call).
    4. Parse operations from the spec.
    5. Register each operation as an MCP tool; individual failures are logged
       and skipped rather than aborting the whole server.
    6. Return (mcp, client) so the caller can close the client at shutdown.

    Args:
        server_config: Configuration for the target server, including name,
            openapi_url, base_url, and auth credentials.

    Returns:
        A ``(FastMCP, httpx.AsyncClient)`` tuple on success, or ``None`` if
        the server cannot be initialized due to a fetch or parse failure.
        The caller is responsible for closing the returned client.
    """
    mcp = FastMCP(server_config.name)

    # build_client can raise ValueError for unrecognised auth types.
    # Treat that the same as a fetch failure: log and return None.
    try:
        client = build_client(server_config)
    except ValueError as exc:
        logger.error(
            "Failed to build HTTP client for server %r: %s",
            server_config.name,
            exc,
        )
        return None

    # Fetch the OpenAPI spec.  fetch_spec is a sync function that uses
    # httpx.get internally, so we run it in a thread to avoid blocking the
    # async event loop.
    try:
        spec = await asyncio.to_thread(fetch_spec, server_config.openapi_url)
    except (httpx.HTTPError, ValueError) as exc:
        logger.error(
            "Failed to fetch OpenAPI spec for server %r from %r: %s",
            server_config.name,
            server_config.openapi_url,
            exc,
        )
        await client.aclose()
        return None

    # Parse operations from the spec.
    try:
        operations: list[OperationDef] = parse_operations(spec)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to parse OpenAPI spec for server %r: %s",
            server_config.name,
            exc,
        )
        await client.aclose()
        return None

    # Register each operation as an MCP tool.  Individual failures are logged
    # and skipped; one bad operation should never abort the whole server.
    for operation in operations:
        try:
            register_tool(mcp, operation, client)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to register tool %r for server %r: %s",
                operation.tool_name,
                server_config.name,
                exc,
            )

    return mcp, client
