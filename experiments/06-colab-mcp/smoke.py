"""Start the pinned colab-mcp server over stdio and list its tools (no browser opened)."""
import asyncio, sys
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
SRC = sys.argv[1]
async def main():
    t = StdioTransport(command="uv", args=["run", "--project", SRC, "colab-mcp"])
    async with Client(t) as c:
        tools = await c.list_tools()
        for x in tools: print(x.name, "-", (x.description or "")[:100])
asyncio.run(main())
