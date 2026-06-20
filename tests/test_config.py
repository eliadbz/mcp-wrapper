"""Tests for mcp_wrapper.config — Pydantic models and YAML loader."""

import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mcp_wrapper.config import (
    AppConfig,
    ApiKeyAuthConfig,
    BasicAuthConfig,
    BearerAuthConfig,
    ServerConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# Helper: write a minimal YAML file for load_config tests
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path: Path, content: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(content))
    return path


# ---------------------------------------------------------------------------
# BearerAuthConfig unit tests
# ---------------------------------------------------------------------------

class TestBearerAuthConfig:
    def test_valid_bearer(self):
        auth = BearerAuthConfig(type="bearer", token="my-token")
        assert auth.type == "bearer"
        assert auth.token == "my-token"

    def test_missing_token_raises(self):
        with pytest.raises(ValidationError):
            BearerAuthConfig(type="bearer")

    def test_extra_fields_ignored(self):
        # Pydantic v2 default: extra fields are ignored
        auth = BearerAuthConfig(type="bearer", token="t", extra_field="ignored")
        assert auth.token == "t"


# ---------------------------------------------------------------------------
# ApiKeyAuthConfig unit tests
# ---------------------------------------------------------------------------

class TestApiKeyAuthConfig:
    def test_valid_api_key(self):
        auth = ApiKeyAuthConfig(type="api_key", header="X-API-Key", value="key123")
        assert auth.type == "api_key"
        assert auth.header == "X-API-Key"
        assert auth.value == "key123"

    def test_missing_header_raises(self):
        with pytest.raises(ValidationError):
            ApiKeyAuthConfig(type="api_key", value="key123")

    def test_missing_value_raises(self):
        with pytest.raises(ValidationError):
            ApiKeyAuthConfig(type="api_key", header="X-API-Key")


# ---------------------------------------------------------------------------
# BasicAuthConfig unit tests
# ---------------------------------------------------------------------------

class TestBasicAuthConfig:
    def test_valid_basic(self):
        auth = BasicAuthConfig(type="basic", username="user", password="pass")
        assert auth.type == "basic"
        assert auth.username == "user"
        assert auth.password == "pass"

    def test_missing_username_raises(self):
        with pytest.raises(ValidationError):
            BasicAuthConfig(type="basic", password="pass")

    def test_missing_password_raises(self):
        with pytest.raises(ValidationError):
            BasicAuthConfig(type="basic", username="user")


# ---------------------------------------------------------------------------
# ServerConfig unit tests
# ---------------------------------------------------------------------------

class TestServerConfig:
    def test_valid_server_with_bearer(self):
        server = ServerConfig(
            name="svc",
            openapi_url="http://svc/openapi.json",
            base_url="http://svc",
            auth={"type": "bearer", "token": "tok"},
        )
        assert server.name == "svc"
        assert isinstance(server.auth, BearerAuthConfig)

    def test_valid_server_with_api_key(self):
        server = ServerConfig(
            name="svc",
            openapi_url="http://svc/openapi.json",
            base_url="http://svc",
            auth={"type": "api_key", "header": "X-Key", "value": "v"},
        )
        assert isinstance(server.auth, ApiKeyAuthConfig)

    def test_valid_server_with_basic(self):
        server = ServerConfig(
            name="svc",
            openapi_url="http://svc/openapi.json",
            base_url="http://svc",
            auth={"type": "basic", "username": "u", "password": "p"},
        )
        assert isinstance(server.auth, BasicAuthConfig)

    def test_unknown_auth_type_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ServerConfig(
                name="svc",
                openapi_url="http://svc/openapi.json",
                base_url="http://svc",
                auth={"type": "oauth2", "token": "tok"},
            )

    def test_missing_openapi_url_raises(self):
        with pytest.raises(ValidationError):
            ServerConfig(
                name="svc",
                base_url="http://svc",
                auth={"type": "bearer", "token": "tok"},
            )


# ---------------------------------------------------------------------------
# AppConfig unit tests
# ---------------------------------------------------------------------------

class TestAppConfig:
    def test_empty_servers(self):
        cfg = AppConfig(servers={})
        assert cfg.servers == {}

    def test_multiple_servers_parsed(self):
        cfg = AppConfig(
            servers={
                "s1": {
                    "name": "s1",
                    "openapi_url": "http://s1/openapi.json",
                    "base_url": "http://s1",
                    "auth": {"type": "bearer", "token": "t1"},
                },
                "s2": {
                    "name": "s2",
                    "openapi_url": "http://s2/openapi.json",
                    "base_url": "http://s2",
                    "auth": {"type": "api_key", "header": "H", "value": "v"},
                },
            }
        )
        assert len(cfg.servers) == 2
        assert cfg.servers["s1"].name == "s1"
        assert isinstance(cfg.servers["s1"].auth, BearerAuthConfig)
        assert cfg.servers["s2"].name == "s2"
        assert isinstance(cfg.servers["s2"].auth, ApiKeyAuthConfig)


# ---------------------------------------------------------------------------
# load_config integration tests
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_bearer_server(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            {
                "servers": {
                    "api": {
                        "openapi_url": "http://api/openapi.json",
                        "base_url": "http://api",
                        "auth": {"type": "bearer", "token": "secret"},
                    }
                }
            },
        )
        cfg = load_config(path)
        assert isinstance(cfg, AppConfig)
        assert "api" in cfg.servers
        server = cfg.servers["api"]
        assert server.name == "api"
        assert server.base_url == "http://api"
        assert isinstance(server.auth, BearerAuthConfig)
        assert server.auth.token == "secret"

    def test_api_key_server(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            {
                "servers": {
                    "svc": {
                        "openapi_url": "http://svc/openapi.json",
                        "base_url": "http://svc",
                        "auth": {"type": "api_key", "header": "X-API-Key", "value": "my-key"},
                    }
                }
            },
        )
        cfg = load_config(path)
        server = cfg.servers["svc"]
        assert server.name == "svc"
        assert isinstance(server.auth, ApiKeyAuthConfig)
        assert server.auth.header == "X-API-Key"
        assert server.auth.value == "my-key"

    def test_basic_server(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            {
                "servers": {
                    "svc": {
                        "openapi_url": "http://svc/openapi.json",
                        "base_url": "http://svc",
                        "auth": {"type": "basic", "username": "user", "password": "pass"},
                    }
                }
            },
        )
        cfg = load_config(path)
        server = cfg.servers["svc"]
        assert server.name == "svc"
        assert isinstance(server.auth, BasicAuthConfig)
        assert server.auth.username == "user"
        assert server.auth.password == "pass"

    def test_multiple_servers_names_injected(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            {
                "servers": {
                    "server1": {
                        "openapi_url": "http://server1/openapi.json",
                        "base_url": "http://server1",
                        "auth": {"type": "bearer", "token": "t1"},
                    },
                    "server2": {
                        "openapi_url": "http://server2/openapi.json",
                        "base_url": "http://server2",
                        "auth": {"type": "api_key", "header": "X-Key", "value": "v2"},
                    },
                    "server3": {
                        "openapi_url": "http://server3/openapi.json",
                        "base_url": "http://server3",
                        "auth": {"type": "basic", "username": "u", "password": "p"},
                    },
                }
            },
        )
        cfg = load_config(path)
        assert len(cfg.servers) == 3
        # Verify names are injected from YAML keys
        for name in ("server1", "server2", "server3"):
            assert cfg.servers[name].name == name

    def test_missing_required_field_raises_validation_error(self, tmp_path):
        """A server missing openapi_url must raise ValidationError."""
        path = _write_yaml(
            tmp_path,
            {
                "servers": {
                    "bad": {
                        "base_url": "http://bad",
                        "auth": {"type": "bearer", "token": "t"},
                    }
                }
            },
        )
        with pytest.raises(ValidationError):
            load_config(path)

    def test_unknown_auth_type_raises_validation_error(self, tmp_path):
        """An unrecognized auth type must raise ValidationError."""
        path = _write_yaml(
            tmp_path,
            {
                "servers": {
                    "bad": {
                        "openapi_url": "http://bad/openapi.json",
                        "base_url": "http://bad",
                        "auth": {"type": "oauth2", "token": "t"},
                    }
                }
            },
        )
        with pytest.raises(ValidationError):
            load_config(path)

    def test_accepts_string_path(self, tmp_path):
        """load_config should accept a plain str path as well as Path."""
        path = _write_yaml(
            tmp_path,
            {
                "servers": {
                    "svc": {
                        "openapi_url": "http://svc/openapi.json",
                        "base_url": "http://svc",
                        "auth": {"type": "bearer", "token": "tok"},
                    }
                }
            },
        )
        cfg = load_config(str(path))
        assert "svc" in cfg.servers

    def test_path_does_not_exist_raises(self, tmp_path):
        """load_config must raise an error for a nonexistent file."""
        with pytest.raises((FileNotFoundError, OSError)):
            load_config(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# ServerConfig readonly tests
# ---------------------------------------------------------------------------


class TestServerConfigReadonly:
    def _base_server(self, **kwargs) -> dict:
        return {
            "name": "svc",
            "openapi_url": "http://svc/openapi.json",
            "base_url": "http://svc",
            **kwargs,
        }

    def test_readonly_defaults_to_false(self):
        server = ServerConfig(**self._base_server())
        assert server.readonly is False

    def test_readonly_overrides_defaults_to_empty_list(self):
        server = ServerConfig(**self._base_server())
        assert server.readonly_overrides == []

    def test_readonly_true_accepted(self):
        server = ServerConfig(**self._base_server(readonly=True))
        assert server.readonly is True

    def test_readonly_overrides_list_accepted(self):
        overrides = ["search_users", "POST /api/search"]
        server = ServerConfig(**self._base_server(readonly_overrides=overrides))
        assert server.readonly_overrides == overrides

    def test_readonly_round_trips_through_yaml(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(
            "servers:\n"
            "  api:\n"
            "    openapi_url: http://api/openapi.json\n"
            "    base_url: http://api\n"
            "    readonly: true\n"
            "    readonly_overrides:\n"
            "      - search_items\n"
            "      - POST /api/search\n"
        )
        cfg = load_config(path)
        server = cfg.servers["api"]
        assert server.readonly is True
        assert server.readonly_overrides == ["search_items", "POST /api/search"]
