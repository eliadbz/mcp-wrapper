"""Tests for mcp_wrapper.openapi — fetch_spec and parse_operations."""

from __future__ import annotations

import httpx
import pytest
import respx

from mcp_wrapper.openapi import (
    OperationDef,
    ParamDef,
    fetch_spec,
    parse_operations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_spec(paths: dict) -> dict:
    """Return a minimal OpenAPI 3.0 spec with the given paths block."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "Test", "version": "0.1.0"},
        "paths": paths,
    }


# ---------------------------------------------------------------------------
# parse_operations — tool_name derivation
# ---------------------------------------------------------------------------

class TestToolNameDerivation:
    def test_operation_id_used_when_present(self):
        spec = _minimal_spec({
            "/users": {
                "get": {
                    "operationId": "listUsers",
                    "summary": "List users",
                    "responses": {"200": {"description": "ok"}},
                }
            }
        })
        ops = parse_operations(spec)
        assert len(ops) == 1
        assert ops[0].tool_name == "listUsers"

    def test_tool_name_derived_from_method_and_path_when_no_operation_id(self):
        spec = _minimal_spec({
            "/users/{id}": {
                "get": {
                    "responses": {"200": {"description": "ok"}},
                }
            }
        })
        ops = parse_operations(spec)
        assert len(ops) == 1
        # method + path: "get" + "/users/{id}" -> "get_users_id"
        assert ops[0].tool_name == "get_users_id"

    def test_derived_name_strips_leading_underscore(self):
        spec = _minimal_spec({
            "/items": {
                "post": {
                    "responses": {"201": {"description": "created"}},
                }
            }
        })
        ops = parse_operations(spec)
        assert len(ops) == 1
        assert ops[0].tool_name == "post_items"

    def test_derived_name_collapses_multiple_underscores(self):
        spec = _minimal_spec({
            "/a--b/{x}": {
                "delete": {
                    "responses": {"204": {"description": "no content"}},
                }
            }
        })
        ops = parse_operations(spec)
        assert len(ops) == 1
        # "/a--b/{x}" -> replace /, {, }, - with _ -> "_a__b__x_"
        # then collapse __ -> _ and strip leading _ -> "delete_a_b_x"
        assert ops[0].tool_name == "delete_a_b_x"

    def test_derived_name_root_path(self):
        spec = _minimal_spec({
            "/": {
                "get": {
                    "responses": {"200": {"description": "ok"}},
                }
            }
        })
        ops = parse_operations(spec)
        assert len(ops) == 1
        # "/" alone -> replace / with _ -> "_" -> strip leading _ -> "" -> prepend "get_"
        assert ops[0].tool_name == "get"


# ---------------------------------------------------------------------------
# parse_operations — description extraction
# ---------------------------------------------------------------------------

class TestDescriptionExtraction:
    def test_summary_preferred_over_description(self):
        spec = _minimal_spec({
            "/x": {
                "get": {
                    "summary": "Short summary",
                    "description": "Longer description",
                    "responses": {"200": {"description": "ok"}},
                }
            }
        })
        ops = parse_operations(spec)
        assert ops[0].description == "Short summary"

    def test_description_used_when_no_summary(self):
        spec = _minimal_spec({
            "/x": {
                "get": {
                    "description": "Longer description",
                    "responses": {"200": {"description": "ok"}},
                }
            }
        })
        ops = parse_operations(spec)
        assert ops[0].description == "Longer description"

    def test_empty_string_when_neither_present(self):
        spec = _minimal_spec({
            "/x": {
                "get": {
                    "responses": {"200": {"description": "ok"}},
                }
            }
        })
        ops = parse_operations(spec)
        assert ops[0].description == ""


# ---------------------------------------------------------------------------
# parse_operations — method and path fields
# ---------------------------------------------------------------------------

class TestMethodAndPath:
    def test_method_is_lowercase(self):
        spec = _minimal_spec({
            "/things": {
                "POST": {
                    "responses": {"201": {"description": "created"}},
                }
            }
        })
        # Methods may come as uppercase in some specs; we lowercase them
        ops = parse_operations(spec)
        assert len(ops) == 1
        assert ops[0].method == "post"

    def test_path_preserved_as_is(self):
        spec = _minimal_spec({
            "/users/{userId}/orders": {
                "get": {
                    "responses": {"200": {"description": "ok"}},
                }
            }
        })
        ops = parse_operations(spec)
        assert ops[0].path == "/users/{userId}/orders"


# ---------------------------------------------------------------------------
# parse_operations — parameter extraction
# ---------------------------------------------------------------------------

class TestParameterExtraction:
    def test_path_params_extracted(self):
        spec = _minimal_spec({
            "/users/{id}": {
                "get": {
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "integer"},
                        }
                    ],
                    "responses": {"200": {"description": "ok"}},
                }
            }
        })
        ops = parse_operations(spec)
        assert len(ops[0].path_params) == 1
        p = ops[0].path_params[0]
        assert p.name == "id"
        assert p.required is True
        assert p.schema == {"type": "integer"}
        assert p.description == ""

    def test_query_params_extracted(self):
        spec = _minimal_spec({
            "/search": {
                "get": {
                    "parameters": [
                        {
                            "name": "q",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string"},
                            "description": "Search query",
                        }
                    ],
                    "responses": {"200": {"description": "ok"}},
                }
            }
        })
        ops = parse_operations(spec)
        assert len(ops[0].query_params) == 1
        q = ops[0].query_params[0]
        assert q.name == "q"
        assert q.required is False
        assert q.schema == {"type": "string"}
        assert q.description == "Search query"

    def test_path_level_parameters_inherited(self):
        spec = _minimal_spec({
            "/users/{id}": {
                "parameters": [
                    {
                        "name": "id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                        "description": "User ID",
                    }
                ],
                "get": {
                    "responses": {"200": {"description": "ok"}},
                }
            }
        })
        ops = parse_operations(spec)
        assert len(ops) == 1
        assert len(ops[0].path_params) == 1
        assert ops[0].path_params[0].name == "id"
        assert ops[0].path_params[0].description == "User ID"

    def test_operation_level_params_override_path_level(self):
        spec = _minimal_spec({
            "/users/{id}": {
                "parameters": [
                    {
                        "name": "id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "From path level",
                    }
                ],
                "get": {
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "integer"},
                            "description": "From operation level",
                        }
                    ],
                    "responses": {"200": {"description": "ok"}},
                }
            }
        })
        ops = parse_operations(spec)
        assert len(ops) == 1
        assert len(ops[0].path_params) == 1
        p = ops[0].path_params[0]
        assert p.schema == {"type": "integer"}
        assert p.description == "From operation level"

    def test_path_and_query_params_separated(self):
        spec = _minimal_spec({
            "/items/{id}": {
                "get": {
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "integer"},
                        },
                        {
                            "name": "expand",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "boolean"},
                        },
                    ],
                    "responses": {"200": {"description": "ok"}},
                }
            }
        })
        ops = parse_operations(spec)
        assert len(ops[0].path_params) == 1
        assert len(ops[0].query_params) == 1
        assert ops[0].path_params[0].name == "id"
        assert ops[0].query_params[0].name == "expand"

    def test_no_params_returns_empty_lists(self):
        spec = _minimal_spec({
            "/status": {
                "get": {
                    "responses": {"200": {"description": "ok"}},
                }
            }
        })
        ops = parse_operations(spec)
        assert ops[0].path_params == []
        assert ops[0].query_params == []

    def test_param_description_defaults_to_empty_string(self):
        spec = _minimal_spec({
            "/x/{id}": {
                "get": {
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
        })
        ops = parse_operations(spec)
        assert ops[0].path_params[0].description == ""

    def test_param_required_defaults_to_false(self):
        spec = _minimal_spec({
            "/search": {
                "get": {
                    "parameters": [
                        {
                            "name": "q",
                            "in": "query",
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {"200": {"description": "ok"}},
                }
            }
        })
        ops = parse_operations(spec)
        assert ops[0].query_params[0].required is False

    def test_param_schema_defaults_to_empty_dict(self):
        spec = _minimal_spec({
            "/search": {
                "get": {
                    "parameters": [
                        {
                            "name": "q",
                            "in": "query",
                        }
                    ],
                    "responses": {"200": {"description": "ok"}},
                }
            }
        })
        ops = parse_operations(spec)
        assert ops[0].query_params[0].schema == {}


# ---------------------------------------------------------------------------
# parse_operations — request body
# ---------------------------------------------------------------------------

class TestRequestBodyExtraction:
    def test_body_schema_extracted_from_application_json(self):
        spec = _minimal_spec({
            "/users": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "email": {"type": "string"},
                                    },
                                    "required": ["name", "email"],
                                }
                            }
                        }
                    },
                    "responses": {"201": {"description": "created"}},
                }
            }
        })
        ops = parse_operations(spec)
        assert ops[0].body_schema is not None
        assert ops[0].body_schema["type"] == "object"
        assert "name" in ops[0].body_schema["properties"]

    def test_body_schema_none_when_no_request_body(self):
        spec = _minimal_spec({
            "/users": {
                "get": {
                    "responses": {"200": {"description": "ok"}},
                }
            }
        })
        ops = parse_operations(spec)
        assert ops[0].body_schema is None

    def test_body_schema_none_when_no_application_json(self):
        spec = _minimal_spec({
            "/upload": {
                "post": {
                    "requestBody": {
                        "content": {
                            "multipart/form-data": {
                                "schema": {"type": "object"}
                            }
                        }
                    },
                    "responses": {"200": {"description": "ok"}},
                }
            }
        })
        ops = parse_operations(spec)
        assert ops[0].body_schema is None

    def test_body_schema_passed_as_is_with_refs(self):
        spec = _minimal_spec({
            "/users": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/User"}
                            }
                        }
                    },
                    "responses": {"201": {"description": "created"}},
                }
            }
        })
        ops = parse_operations(spec)
        # $ref is passed as-is, not resolved
        assert ops[0].body_schema == {"$ref": "#/components/schemas/User"}


# ---------------------------------------------------------------------------
# parse_operations — multiple operations and paths
# ---------------------------------------------------------------------------

class TestMultipleOperations:
    def test_multiple_methods_on_same_path(self):
        spec = _minimal_spec({
            "/users": {
                "get": {
                    "operationId": "listUsers",
                    "responses": {"200": {"description": "ok"}},
                },
                "post": {
                    "operationId": "createUser",
                    "responses": {"201": {"description": "created"}},
                },
            }
        })
        ops = parse_operations(spec)
        assert len(ops) == 2
        tool_names = {op.tool_name for op in ops}
        assert tool_names == {"listUsers", "createUser"}

    def test_multiple_paths(self):
        spec = _minimal_spec({
            "/users": {
                "get": {"operationId": "listUsers", "responses": {"200": {"description": "ok"}}},
            },
            "/orders": {
                "get": {"operationId": "listOrders", "responses": {"200": {"description": "ok"}}},
                "post": {"operationId": "createOrder", "responses": {"201": {"description": "created"}}},
            },
        })
        ops = parse_operations(spec)
        assert len(ops) == 3
        tool_names = {op.tool_name for op in ops}
        assert tool_names == {"listUsers", "listOrders", "createOrder"}

    def test_non_method_keys_at_path_level_skipped(self):
        """Keys like 'parameters', 'summary', 'description' at path level are not parsed as ops."""
        spec = _minimal_spec({
            "/users/{id}": {
                "summary": "User operations",
                "description": "Manage a single user",
                "parameters": [
                    {
                        "name": "id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                    }
                ],
                "get": {
                    "operationId": "getUser",
                    "responses": {"200": {"description": "ok"}},
                },
            }
        })
        ops = parse_operations(spec)
        assert len(ops) == 1
        assert ops[0].tool_name == "getUser"

    def test_all_valid_http_methods_parsed(self):
        spec = _minimal_spec({
            "/resource": {
                "get": {"responses": {"200": {"description": "ok"}}},
                "post": {"responses": {"201": {"description": "created"}}},
                "put": {"responses": {"200": {"description": "ok"}}},
                "patch": {"responses": {"200": {"description": "ok"}}},
                "delete": {"responses": {"204": {"description": "no content"}}},
                "head": {"responses": {"200": {"description": "ok"}}},
                "options": {"responses": {"200": {"description": "ok"}}},
            }
        })
        ops = parse_operations(spec)
        assert len(ops) == 7
        methods = {op.method for op in ops}
        assert methods == {"get", "post", "put", "patch", "delete", "head", "options"}


# ---------------------------------------------------------------------------
# parse_operations — graceful error handling
# ---------------------------------------------------------------------------

class TestGracefulHandling:
    def test_empty_paths_returns_empty_list(self):
        spec = _minimal_spec({})
        ops = parse_operations(spec)
        assert ops == []

    def test_missing_paths_key_returns_empty_list(self):
        spec = {"openapi": "3.0.0", "info": {"title": "Test", "version": "0.1.0"}}
        ops = parse_operations(spec)
        assert ops == []

    def test_malformed_path_entry_skipped(self):
        """A path value that is not a dict should be silently skipped."""
        spec = _minimal_spec({
            "/good": {
                "get": {
                    "operationId": "goodOp",
                    "responses": {"200": {"description": "ok"}},
                }
            },
            "/bad": "this is not a dict",
        })
        ops = parse_operations(spec)
        assert len(ops) == 1
        assert ops[0].tool_name == "goodOp"

    def test_malformed_operation_entry_skipped(self):
        """An operation value that is not a dict should be silently skipped."""
        spec = _minimal_spec({
            "/users": {
                "get": "not a dict",
                "post": {
                    "operationId": "createUser",
                    "responses": {"201": {"description": "created"}},
                },
            }
        })
        ops = parse_operations(spec)
        assert len(ops) == 1
        assert ops[0].tool_name == "createUser"

    def test_none_paths_value_returns_empty_list(self):
        spec = {"openapi": "3.0.0", "info": {}, "paths": None}
        ops = parse_operations(spec)
        assert ops == []


# ---------------------------------------------------------------------------
# fetch_spec — with respx mocking
# ---------------------------------------------------------------------------

class TestFetchSpec:
    @respx.mock
    def test_json_response_returns_parsed_dict(self):
        url = "http://example.com/openapi.json"
        spec_data = {"openapi": "3.0.0", "info": {"title": "Test", "version": "1.0.0"}, "paths": {}}
        respx.get(url).mock(return_value=httpx.Response(200, json=spec_data))

        result = fetch_spec(url)

        assert isinstance(result, dict)
        assert result["openapi"] == "3.0.0"
        assert result["info"]["title"] == "Test"

    @respx.mock
    def test_yaml_response_returns_parsed_dict(self):
        url = "http://example.com/openapi.yaml"
        yaml_content = (
            "openapi: '3.0.0'\n"
            "info:\n"
            "  title: YAML Test\n"
            "  version: '1.0.0'\n"
            "paths: {}\n"
        )
        respx.get(url).mock(
            return_value=httpx.Response(
                200,
                content=yaml_content.encode(),
                headers={"content-type": "application/yaml"},
            )
        )

        result = fetch_spec(url)

        assert isinstance(result, dict)
        assert result["openapi"] == "3.0.0"
        assert result["info"]["title"] == "YAML Test"

    @respx.mock
    def test_yaml_content_type_text_yaml(self):
        url = "http://example.com/spec.yaml"
        yaml_content = (
            "openapi: '3.1.0'\n"
            "info:\n"
            "  title: Text YAML\n"
            "  version: '2.0.0'\n"
            "paths: {}\n"
        )
        respx.get(url).mock(
            return_value=httpx.Response(
                200,
                content=yaml_content.encode(),
                headers={"content-type": "text/yaml"},
            )
        )

        result = fetch_spec(url)

        assert result["info"]["title"] == "Text YAML"

    @respx.mock
    def test_http_error_raises_httpx_http_status_error(self):
        url = "http://example.com/openapi.json"
        respx.get(url).mock(return_value=httpx.Response(404))

        with pytest.raises(httpx.HTTPStatusError):
            fetch_spec(url)

    @respx.mock
    def test_invalid_response_body_raises_value_error(self):
        url = "http://example.com/openapi.json"
        # Return something that's neither valid JSON nor valid YAML structure
        respx.get(url).mock(
            return_value=httpx.Response(
                200,
                content=b"\x00\x01\x02\x03",  # binary garbage
                headers={"content-type": "application/json"},
            )
        )

        with pytest.raises(ValueError):
            fetch_spec(url)

    @respx.mock
    def test_json_content_type_used_for_json_detection(self):
        url = "http://example.com/api-docs"
        spec_data = {"openapi": "3.0.0", "paths": {}}
        respx.get(url).mock(
            return_value=httpx.Response(
                200,
                json=spec_data,
                headers={"content-type": "application/json"},
            )
        )

        result = fetch_spec(url)
        assert result["openapi"] == "3.0.0"

    @respx.mock
    def test_fallback_yaml_parse_when_no_clear_content_type(self):
        url = "http://example.com/openapi"
        yaml_content = (
            "openapi: '3.0.0'\n"
            "info:\n"
            "  title: Fallback\n"
            "  version: '1.0.0'\n"
            "paths: {}\n"
        )
        respx.get(url).mock(
            return_value=httpx.Response(
                200,
                content=yaml_content.encode(),
                headers={"content-type": "text/plain"},
            )
        )

        result = fetch_spec(url)
        assert result["info"]["title"] == "Fallback"
