"""Tests for mcp_wrapper.main — FastAPI app with lifespan and MCP mounts.

Uses unittest.mock to patch build_mcp_server and load_config so no real
HTTP calls or file I/O are performed.

Key testing approach:
- Each test reloads mcp_wrapper.main via importlib.reload() to get a fresh
  app instance with no accumulated mounts from prior tests.
- patch.object() is used to patch the freshly-loaded module's attributes.
- TestClient (sync, via starlette) triggers the FastAPI lifespan, so it is
  used for tests that need startup/shutdown to execute (route mounting,
  client lifecycle, env-var config path).
- httpx.AsyncClient with ASGITransport does NOT trigger lifespan; it is used
  only for the basic /health async smoke-test where lifespan is not needed.
"""

from __future__ import annotations

import importlib
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import mcp_wrapper.main
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from mcp.server.fastmcp import FastMCP
from starlette.routing import Mount


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def reload_main() -> "mcp_wrapper.main":  # type: ignore[name-defined]
    """Reload mcp_wrapper.main and return the fresh module.

    Reloading guarantees that each test starts with a pristine FastAPI app
    that has no mounts from previous tests.
    """
    importlib.reload(mcp_wrapper.main)
    return mcp_wrapper.main


def make_mock_app_config(server_ids: list[str]) -> MagicMock:
    """Return a mock AppConfig with the given server IDs."""
    config = MagicMock()
    servers: dict = {}
    for sid in server_ids:
        sc = MagicMock()
        sc.name = sid
        servers[sid] = sc
    config.servers = servers
    return config


def make_mock_build_result(server_name: str = "test") -> tuple[FastMCP, MagicMock]:
    """Return a (FastMCP, mock_client) pair as build_mcp_server would."""
    mcp = FastMCP(server_name)
    client = MagicMock()
    client.aclose = AsyncMock()
    return mcp, client


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Verify that GET /health returns the expected JSON response."""

    def test_health_returns_ok(self):
        """GET /health must return HTTP 200 with {status: ok}."""
        mod = reload_main()
        mock_config = make_mock_app_config([])

        with (
            patch.object(mod, "load_config", return_value=mock_config),
            patch.object(mod, "build_mcp_server", new_callable=AsyncMock, return_value=None),
        ):
            with TestClient(mod.app) as client:
                response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_health_returns_ok_async(self):
        """GET /health returns {status: ok} via async client (no lifespan needed)."""
        mod = reload_main()
        mock_config = make_mock_app_config([])

        with (
            patch.object(mod, "load_config", return_value=mock_config),
            patch.object(mod, "build_mcp_server", new_callable=AsyncMock, return_value=None),
        ):
            # ASGITransport does NOT run lifespan, but /health is defined at module
            # level and does not depend on startup code.
            async with AsyncClient(
                transport=ASGITransport(app=mod.app), base_url="http://test"
            ) as client:
                response = await client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Server mounting
# ---------------------------------------------------------------------------


class TestServerMounting:
    """Verify that servers are mounted at the correct paths.

    TestClient is used because it triggers the FastAPI lifespan where
    app.mount() is called.
    """

    def test_server_mounted_at_correct_path(self):
        """A successfully built server should be mounted at /servers/{id}."""
        mod = reload_main()
        mock_config = make_mock_app_config(["petstore"])
        mcp_instance, mock_client = make_mock_build_result("petstore")

        with (
            patch.object(mod, "load_config", return_value=mock_config),
            patch.object(
                mod, "build_mcp_server", new_callable=AsyncMock,
                return_value=(mcp_instance, mock_client),
            ),
        ):
            with TestClient(mod.app) as client:
                response = client.get("/health")
                assert response.status_code == 200

                # Verify the Mount is registered in the router.
                mount_paths = [r.path for r in mod.app.routes if isinstance(r, Mount)]
                assert "/servers/petstore" in mount_paths

    def test_multiple_servers_mounted(self):
        """Multiple servers should each be mounted at their own paths."""
        mod = reload_main()
        mock_config = make_mock_app_config(["alpha", "beta"])

        build_results = {
            "alpha": make_mock_build_result("alpha"),
            "beta": make_mock_build_result("beta"),
        }

        async def build_side_effect(server_config):
            return build_results[server_config.name]

        with (
            patch.object(mod, "load_config", return_value=mock_config),
            patch.object(mod, "build_mcp_server", side_effect=build_side_effect),
        ):
            with TestClient(mod.app) as client:
                response = client.get("/health")
                assert response.status_code == 200

                mount_paths = [r.path for r in mod.app.routes if isinstance(r, Mount)]
                assert "/servers/alpha" in mount_paths
                assert "/servers/beta" in mount_paths

    def test_mount_path_format(self):
        """Mount path must follow /servers/{server_id} format exactly."""
        mod = reload_main()
        mock_config = make_mock_app_config(["my-api"])
        mcp_instance, mock_client = make_mock_build_result("my-api")

        with (
            patch.object(mod, "load_config", return_value=mock_config),
            patch.object(
                mod, "build_mcp_server", new_callable=AsyncMock,
                return_value=(mcp_instance, mock_client),
            ),
        ):
            with TestClient(mod.app):
                mount_paths = [r.path for r in mod.app.routes if isinstance(r, Mount)]
                assert "/servers/my-api" in mount_paths


# ---------------------------------------------------------------------------
# Failed server — skip, app still starts
# ---------------------------------------------------------------------------


class TestFailedServerSkipped:
    """Verify that a server that fails build_mcp_server is skipped gracefully."""

    def test_failed_server_skipped_app_still_starts(self):
        """If build_mcp_server returns None, that server is skipped; app still starts."""
        mod = reload_main()
        mock_config = make_mock_app_config(["bad-server"])

        with (
            patch.object(mod, "load_config", return_value=mock_config),
            patch.object(mod, "build_mcp_server", new_callable=AsyncMock, return_value=None),
        ):
            with TestClient(mod.app) as client:
                response = client.get("/health")

        assert response.status_code == 200

    def test_failed_server_not_mounted(self):
        """A server that returns None from build_mcp_server must not have a mount."""
        mod = reload_main()
        mock_config = make_mock_app_config(["bad-server"])

        with (
            patch.object(mod, "load_config", return_value=mock_config),
            patch.object(mod, "build_mcp_server", new_callable=AsyncMock, return_value=None),
        ):
            with TestClient(mod.app):
                mount_paths = [r.path for r in mod.app.routes if isinstance(r, Mount)]
                assert "/servers/bad-server" not in mount_paths

    def test_one_failed_one_success_only_success_mounted(self):
        """With one failing and one succeeding server, only the successful one is mounted."""
        mod = reload_main()
        mock_config = make_mock_app_config(["bad", "good"])
        good_mcp, good_client = make_mock_build_result("good")

        build_results = {
            "bad": None,
            "good": (good_mcp, good_client),
        }

        async def build_side_effect(server_config):
            return build_results[server_config.name]

        with (
            patch.object(mod, "load_config", return_value=mock_config),
            patch.object(mod, "build_mcp_server", side_effect=build_side_effect),
        ):
            with TestClient(mod.app) as client:
                response = client.get("/health")
                assert response.status_code == 200

                mount_paths = [r.path for r in mod.app.routes if isinstance(r, Mount)]
                assert "/servers/good" in mount_paths
                assert "/servers/bad" not in mount_paths

    def test_all_servers_fail_zero_mounts(self):
        """When every server fails, zero mounts are added and health still returns 200."""
        mod = reload_main()
        mock_config = make_mock_app_config(["s1", "s2"])

        with (
            patch.object(mod, "load_config", return_value=mock_config),
            patch.object(mod, "build_mcp_server", new_callable=AsyncMock, return_value=None),
        ):
            with TestClient(mod.app) as client:
                response = client.get("/health")
                assert response.status_code == 200

                mounts = [r for r in mod.app.routes if isinstance(r, Mount)]
                assert mounts == []


# ---------------------------------------------------------------------------
# MCP_WRAPPER_CONFIG env var
# ---------------------------------------------------------------------------


class TestConfigEnvVar:
    """Verify that MCP_WRAPPER_CONFIG overrides the default config path."""

    def test_env_var_overrides_config_path(self, tmp_path):
        """MCP_WRAPPER_CONFIG should be used as config path when set."""
        custom_config_path = str(tmp_path / "custom.yaml")
        mock_config = make_mock_app_config([])
        captured_paths: list[str] = []

        def capturing_load_config(path):
            captured_paths.append(str(path))
            return mock_config

        mod = reload_main()

        with (
            patch.dict(os.environ, {"MCP_WRAPPER_CONFIG": custom_config_path}),
            patch.object(mod, "load_config", side_effect=capturing_load_config),
            patch.object(mod, "build_mcp_server", new_callable=AsyncMock, return_value=None),
        ):
            with TestClient(mod.app):
                pass

        assert len(captured_paths) == 1
        assert captured_paths[0] == custom_config_path

    def test_default_config_path_when_env_var_not_set(self, monkeypatch):
        """When MCP_WRAPPER_CONFIG is not set, load_config is called with 'config.yaml'."""
        monkeypatch.delenv("MCP_WRAPPER_CONFIG", raising=False)
        mock_config = make_mock_app_config([])
        captured_paths: list[str] = []

        def capturing_load_config(path):
            captured_paths.append(str(path))
            return mock_config

        mod = reload_main()

        with (
            patch.object(mod, "load_config", side_effect=capturing_load_config),
            patch.object(mod, "build_mcp_server", new_callable=AsyncMock, return_value=None),
        ):
            with TestClient(mod.app):
                pass

        assert len(captured_paths) == 1
        assert captured_paths[0] == "config.yaml"


# ---------------------------------------------------------------------------
# Client lifecycle — clients closed at shutdown
# ---------------------------------------------------------------------------


class TestClientLifecycle:
    """Verify that all HTTP clients are closed when the app shuts down.

    The lifespan context manager is driven directly (not via HTTP) to allow
    async assertions on the mock clients' aclose() calls.
    """

    @pytest.mark.asyncio
    async def test_clients_closed_at_shutdown(self):
        """All async clients should have aclose() called during lifespan teardown."""
        mod = reload_main()
        mock_config = make_mock_app_config(["svc1", "svc2"])

        mcp1, client1 = make_mock_build_result("svc1")
        mcp2, client2 = make_mock_build_result("svc2")

        build_results = {
            "svc1": (mcp1, client1),
            "svc2": (mcp2, client2),
        }

        async def build_side_effect(server_config):
            return build_results[server_config.name]

        with (
            patch.object(mod, "load_config", return_value=mock_config),
            patch.object(mod, "build_mcp_server", side_effect=build_side_effect),
        ):
            # Drive the lifespan directly: startup -> yield -> shutdown.
            async with mod.lifespan(mod.app):
                pass

        client1.aclose.assert_awaited_once()
        client2.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_clients_closed_when_all_servers_fail(self):
        """When all servers fail to build, aclose is not called (no clients stored)."""
        mod = reload_main()
        mock_config = make_mock_app_config(["bad"])

        with (
            patch.object(mod, "load_config", return_value=mock_config),
            patch.object(mod, "build_mcp_server", new_callable=AsyncMock, return_value=None),
        ):
            # Should complete without error even with no clients.
            async with mod.lifespan(mod.app):
                pass

        # No assertion needed beyond "no exception was raised".

    @pytest.mark.asyncio
    async def test_client_closed_even_when_some_servers_fail(self):
        """Clients from successful servers are closed even if other servers failed."""
        mod = reload_main()
        mock_config = make_mock_app_config(["bad", "good"])
        good_mcp, good_client = make_mock_build_result("good")

        build_results = {"bad": None, "good": (good_mcp, good_client)}

        async def build_side_effect(server_config):
            return build_results[server_config.name]

        with (
            patch.object(mod, "load_config", return_value=mock_config),
            patch.object(mod, "build_mcp_server", side_effect=build_side_effect),
        ):
            async with mod.lifespan(mod.app):
                pass

        good_client.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# Import-time safety
# ---------------------------------------------------------------------------


class TestImportTimeSafety:
    """Verify that importing main does not trigger side effects."""

    def test_import_does_not_call_load_config(self):
        """Importing mcp_wrapper.main must not call load_config at module level."""
        sys.modules.pop("mcp_wrapper.main", None)

        load_config_calls: list = []

        with patch(
            "mcp_wrapper.config.load_config",
            side_effect=lambda p: load_config_calls.append(p),
        ):
            import mcp_wrapper.main as _  # noqa: F401

        assert load_config_calls == [], "load_config must not be called at import time"

    def test_app_object_importable(self):
        """from mcp_wrapper.main import app must succeed without errors."""
        mod = reload_main()
        assert mod.app is not None
