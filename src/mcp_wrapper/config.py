"""Configuration models and YAML loader for mcp-wrapper.

Defines Pydantic v2 models for the YAML config schema and a load_config()
function that reads a YAML file and returns a validated AppConfig instance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Auth config models (discriminated union on `type`)
# ---------------------------------------------------------------------------


class BearerAuthConfig(BaseModel):
    """Bearer token authentication."""

    type: Literal["bearer"]
    token: str


class ApiKeyAuthConfig(BaseModel):
    """API key authentication via a custom header."""

    type: Literal["api_key"]
    header: str
    value: str


class BasicAuthConfig(BaseModel):
    """HTTP Basic authentication."""

    type: Literal["basic"]
    username: str
    password: str


AuthConfig = Annotated[
    Union[BearerAuthConfig, ApiKeyAuthConfig, BasicAuthConfig],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Server and application config models
# ---------------------------------------------------------------------------


class ServerConfig(BaseModel):
    """Configuration for a single proxied server."""

    name: str
    openapi_url: str
    base_url: str
    auth: AuthConfig


class AppConfig(BaseModel):
    """Top-level application configuration."""

    servers: dict[str, ServerConfig]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> AppConfig:
    """Load and validate configuration from a YAML file.

    Args:
        path: Filesystem path to the YAML config file (str or Path).

    Returns:
        A validated AppConfig instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        pydantic.ValidationError: If the config fails schema validation,
            including unknown auth types or missing required fields.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    servers_raw: dict = raw.get("servers") or {}

    # Inject the dict key as the `name` field for each server entry before
    # passing to Pydantic for validation.
    for server_name, server_data in servers_raw.items():
        if isinstance(server_data, dict):
            server_data["name"] = server_name

    return AppConfig.model_validate(raw)
