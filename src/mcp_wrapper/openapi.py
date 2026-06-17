"""OpenAPI spec fetching and parsing for mcp-wrapper.

Provides two public functions:
- fetch_spec(url) — download an OpenAPI JSON or YAML spec and return it as a dict
- parse_operations(spec) — extract OperationDef objects from an OpenAPI spec dict

These run synchronously at startup (inside a lifespan handler), not during
request handling, so the sync httpx call is intentional.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx
import yaml


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ParamDef:
    """A single parameter (path or query) extracted from an OpenAPI operation."""

    name: str
    required: bool
    schema: dict  # JSON Schema fragment; may contain $ref, passed as-is
    description: str  # empty string if absent


@dataclass
class OperationDef:
    """An HTTP operation extracted from an OpenAPI spec, ready for MCP tool registration."""

    tool_name: str          # operationId if present, else "{method}_{path_snake}"
    method: str             # lowercase HTTP method: "get", "post", etc.
    path: str               # raw path template, e.g. "/users/{id}"
    description: str        # summary > description > ""
    path_params: list[ParamDef]
    query_params: list[ParamDef]
    body_schema: dict | None  # JSON Schema for application/json request body, or None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_METHODS = frozenset({"get", "post", "put", "patch", "delete", "head", "options"})


# ---------------------------------------------------------------------------
# fetch_spec
# ---------------------------------------------------------------------------


def fetch_spec(url: str) -> dict:
    """Fetch an OpenAPI spec from *url* and return it as a Python dict.

    Supports both JSON and YAML responses. Detection uses the Content-Type
    header first; if the content type contains "yaml" the body is parsed as
    YAML. Otherwise JSON is tried first, with YAML as a fallback.

    Args:
        url: Full URL to the OpenAPI spec (JSON or YAML).

    Returns:
        The parsed spec as a plain Python dict.

    Raises:
        httpx.HTTPError: On any network-level error or non-2xx HTTP status.
        ValueError: If the response body cannot be parsed as JSON or YAML,
            or if the parsed result is not a dict.
    """
    response = httpx.get(url, follow_redirects=True, timeout=30)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    raw_bytes = response.content

    if "yaml" in content_type:
        return _parse_yaml_bytes(raw_bytes)

    # Try JSON first, then fall back to YAML
    try:
        parsed = response.json()
        if not isinstance(parsed, dict):
            raise ValueError(
                f"Expected a JSON object at the top level, got {type(parsed).__name__}"
            )
        return parsed
    except Exception as json_exc:  # noqa: BLE001
        # JSON parse failed — try YAML
        try:
            return _parse_yaml_bytes(raw_bytes)
        except ValueError:
            # Re-raise the original JSON error for clarity
            raise ValueError(
                f"Response from {url!r} is not valid JSON or YAML: {json_exc}"
            ) from json_exc


def _parse_yaml_bytes(raw: bytes) -> dict:
    """Parse *raw* bytes as YAML and return a dict, or raise ValueError."""
    try:
        text = raw.decode("utf-8", errors="replace")
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse response as YAML: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Expected a YAML mapping at the top level, got {type(parsed).__name__}"
        )
    return parsed


# ---------------------------------------------------------------------------
# parse_operations
# ---------------------------------------------------------------------------


def parse_operations(spec: dict) -> list[OperationDef]:
    """Extract all HTTP operations from an OpenAPI spec dict.

    Iterates ``spec["paths"]``, parses each operation, and returns a flat list
    of OperationDef objects.  Malformed path or operation entries are silently
    skipped so a single bad spec entry cannot break the entire startup.

    Parameters can be defined at path level and/or operation level.  Operation-
    level parameters shadow path-level parameters with the same (name, in) key.

    Args:
        spec: A raw OpenAPI 3.x spec as a Python dict (refs are NOT resolved).

    Returns:
        A list of OperationDef, one per valid HTTP operation found in the spec.
    """
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return []

    operations: list[OperationDef] = []

    for path_str, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue  # malformed path entry — skip

        path_level_params: list[dict] = path_item.get("parameters") or []

        for raw_method, operation in path_item.items():
            method = raw_method.lower()
            if method not in _VALID_METHODS:
                continue  # skip non-method keys ("parameters", "summary", etc.)

            if not isinstance(operation, dict):
                continue  # malformed operation entry — skip

            try:
                op_def = _build_operation(path_str, method, operation, path_level_params)
            except Exception:  # noqa: BLE001
                continue  # defensive: skip on any unexpected error

            operations.append(op_def)

    return operations


def _build_operation(
    path_str: str,
    method: str,
    operation: dict,
    path_level_params: list[dict],
) -> OperationDef:
    """Build a single OperationDef from a parsed OpenAPI operation object."""
    tool_name = _derive_tool_name(method, path_str, operation.get("operationId"))
    description = operation.get("summary") or operation.get("description") or ""

    # Merge parameters: path-level first, operation-level overrides by (name, in)
    merged_params = _merge_params(path_level_params, operation.get("parameters") or [])

    path_params: list[ParamDef] = []
    query_params: list[ParamDef] = []

    for param in merged_params:
        if not isinstance(param, dict):
            continue
        location = param.get("in", "")
        param_def = ParamDef(
            name=param.get("name", ""),
            required=bool(param.get("required", False)),
            schema=param.get("schema") or {},
            description=param.get("description") or "",
        )
        if location == "path":
            path_params.append(param_def)
        elif location == "query":
            query_params.append(param_def)

    body_schema = _extract_body_schema(operation)

    return OperationDef(
        tool_name=tool_name,
        method=method,
        path=path_str,
        description=description,
        path_params=path_params,
        query_params=query_params,
        body_schema=body_schema,
    )


def _derive_tool_name(method: str, path: str, operation_id: str | None) -> str:
    """Return the MCP tool name for an operation.

    Uses *operation_id* directly if present.  Otherwise derives a name from
    *method* and *path* by replacing special characters with underscores,
    collapsing consecutive underscores, and stripping leading underscores.
    """
    if operation_id:
        return operation_id

    # Replace /, {, }, - with _
    path_part = re.sub(r"[/{}\\-]", "_", path)
    # Collapse multiple consecutive underscores into one
    path_part = re.sub(r"_+", "_", path_part)
    # Strip leading/trailing underscores
    path_part = path_part.strip("_")

    if path_part:
        return f"{method}_{path_part}"
    return method


def _merge_params(
    path_level: list[dict],
    op_level: list[dict],
) -> list[dict]:
    """Return a merged parameter list where op-level overrides path-level.

    The key for deduplication is (name, in).  Operation-level params take
    precedence; all remaining path-level params are appended.
    """
    result: dict[tuple[str, str], dict] = {}

    for param in path_level:
        if isinstance(param, dict):
            key = (param.get("name", ""), param.get("in", ""))
            result[key] = param

    for param in op_level:
        if isinstance(param, dict):
            key = (param.get("name", ""), param.get("in", ""))
            result[key] = param  # overrides path-level entry if same key

    return list(result.values())


def _extract_body_schema(operation: dict) -> dict | None:
    """Return the JSON Schema for the application/json request body, or None."""
    request_body = operation.get("requestBody")
    if not isinstance(request_body, dict):
        return None

    content = request_body.get("content")
    if not isinstance(content, dict):
        return None

    json_content = content.get("application/json")
    if not isinstance(json_content, dict):
        return None

    schema = json_content.get("schema")
    return schema if isinstance(schema, dict) else None
