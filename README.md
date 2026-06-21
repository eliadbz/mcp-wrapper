# mcp-wrapper

mcp-wrapper proxies third-party REST APIs as MCP tools over Streamable HTTP, letting any MCP client call your existing HTTP APIs without modification. It reads an OpenAPI spec from each configured server, generates one MCP tool per operation, and forwards calls through an authenticated httpx client.

## Quick start

**Install dependencies**

```
uv sync
```

**Configure servers**

```
cp config.example.yaml config.yaml
# edit config.yaml with your server URLs and credentials
```

**Run**

```
uv run uvicorn mcp_wrapper.main:app --host 0.0.0.0 --port 8000
```

## Client connection

Connect your MCP client to:

```
http://localhost:8000/servers/{server_id}/mcp
```

Replace `{server_id}` with the key used for the server in `config.yaml`. Uses the MCP Streamable HTTP transport (POST to `/mcp`).

## Config schema

The config file is a YAML document with a top-level `servers` map. Each key becomes the `{server_id}` used in the connection URL.

| Field | Required | Description |
|---|---|---|
| `servers.<id>.openapi_url` | Yes | URL to fetch the OpenAPI JSON spec from |
| `servers.<id>.base_url` | Yes | Base URL for all API requests |
| `servers.<id>.auth` | No | Auth config block (see auth types below); omit for public APIs |
| `servers.<id>.auth.type` | Yes (if auth set) | One of `bearer`, `api_key`, or `basic` |
| `servers.<id>.readonly` | No | If `true`, filters out all mutating operations (POST/PUT/PATCH/DELETE) and annotates remaining tools with `readOnlyHint` |
| `servers.<id>.readonly_overrides` | No | List of exceptions to include despite `readonly: true` — each entry is either an operationId/tool name or `"METHOD /path"` |

### Auth types

**bearer** — sends `Authorization: Bearer <token>` on every request

| Field | Description |
|---|---|
| `token` | The bearer token value |

**api_key** — sends a custom header with a fixed value

| Field | Description |
|---|---|
| `header` | Header name (e.g. `X-API-Key`) |
| `value` | Header value |

**basic** — HTTP Basic authentication

| Field | Description |
|---|---|
| `username` | Basic auth username |
| `password` | Basic auth password |

### Readonly mode

Set `readonly: true` to expose a server as read-only. mcp-wrapper will:

- Drop all POST, PUT, PATCH, and DELETE operations (they won't appear as tools)
- Annotate every registered tool with `readOnlyHint: true`

Use `readonly_overrides` to include specific mutating endpoints anyway (e.g. a POST search endpoint):

```yaml
readonly-api:
  openapi_url: "http://search-api/openapi.json"
  base_url: "http://search-api"
  auth:
    type: api_key
    header: "X-API-Key"
    value: "your-api-key-here"
  readonly: true
  readonly_overrides:
    - search_users          # by operationId / tool name
    - "POST /api/search"    # by HTTP method + path (method must be uppercase)
```

## Environment variable

Set `MCP_WRAPPER_CONFIG` to override the config file path (default: `config.yaml` relative to the working directory):

```
MCP_WRAPPER_CONFIG=/etc/mcp-wrapper/config.yaml uv run uvicorn mcp_wrapper.main:app --host 0.0.0.0 --port 8000
```
