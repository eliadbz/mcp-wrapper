import uvicorn

uvicorn.run("mcp_wrapper.main:app", host="0.0.0.0", port=8000)
