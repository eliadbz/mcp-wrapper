"""HTTP client factory for authenticated requests to proxied servers.

Provides build_client() to construct configured AsyncClient instances with
pre-configured authentication headers based on ServerConfig.
"""

from __future__ import annotations

import httpx

from mcp_wrapper.config import (
    ApiKeyAuthConfig,
    BasicAuthConfig,
    BearerAuthConfig,
    ServerConfig,
)


def build_client(server_config: ServerConfig) -> httpx.AsyncClient:
    """Return a configured AsyncClient for the given server.

    Args:
        server_config: The server configuration including base_url and auth.

    Returns:
        A configured httpx.AsyncClient with auth and timeout set.

    Raises:
        ValueError: If the auth type is unrecognized.
    """
    # Set up auth based on type
    headers: dict[str, str] = {}
    auth: httpx.BasicAuth | None = None

    if server_config.auth is None:
        pass
    elif isinstance(server_config.auth, BearerAuthConfig):
        headers["Authorization"] = f"Bearer {server_config.auth.token}"
    elif isinstance(server_config.auth, ApiKeyAuthConfig):
        headers[server_config.auth.header] = server_config.auth.value
    elif isinstance(server_config.auth, BasicAuthConfig):
        auth = httpx.BasicAuth(server_config.auth.username, server_config.auth.password)
    else:
        raise ValueError(
            f"Unrecognized auth type: {type(server_config.auth).__name__}"
        )

    # Create and return the client
    return httpx.AsyncClient(
        base_url=server_config.base_url,
        headers=headers if headers else None,
        auth=auth,
        timeout=httpx.Timeout(30.0),
    )
