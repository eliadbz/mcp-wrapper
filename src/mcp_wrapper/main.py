"""Top-level FastAPI application for mcp-wrapper.

Reads config.yaml at startup, creates one FastMCP instance per configured
server, and mounts each SSE ASGI app at /servers/{server_id}/mcp.

Entry point:
    uv run uvicorn mcp_wrapper.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from mcp_wrapper.config import load_config
from mcp_wrapper.server import build_mcp_server

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager.

    On startup:
        - Read config from the path given by the MCP_WRAPPER_CONFIG env var
          (defaults to "config.yaml" relative to the CWD).
        - Call build_mcp_server for each configured server.
        - Mount each server's SSE ASGI app at /servers/{server_id}/mcp.
        - Servers that fail to build are skipped with a warning; the app
          continues to start normally.

    On shutdown:
        - Close all HTTP clients that were created during startup.
    """
    config_path = os.environ.get("MCP_WRAPPER_CONFIG", "config.yaml")
    config = load_config(config_path)

    clients: dict[str, httpx.AsyncClient] = {}

    for server_id, server_config in config.servers.items():
        result = await build_mcp_server(server_config)
        if result is None:
            logger.warning(
                "Skipping server %r — build_mcp_server returned None",
                server_id,
            )
            continue

        mcp, client = result
        clients[server_id] = client
        mount_path = f"/servers/{server_id}/mcp"
        app.mount(mount_path, mcp.sse_app())
        logger.info("Mounted server %r at %s", server_id, mount_path)

    yield  # application is running

    # Shutdown — close every HTTP client in reverse order.
    for server_id, client in clients.items():
        try:
            await client.aclose()
            logger.debug("Closed HTTP client for server %r", server_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error closing client for server %r: %s", server_id, exc)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="mcp-wrapper",
    description="Proxies third-party REST APIs as MCP tools over SSE.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    """Health check endpoint.

    Returns:
        JSON ``{"status": "ok"}`` with HTTP 200.
    """
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Direct execution entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mcp_wrapper.main:app", host="0.0.0.0", port=8000, reload=False)
