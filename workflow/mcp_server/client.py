import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="mcp-email-server",
    args=["stdio"],
)

async def call_tool(name, args):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments=args)
            return result.content[0].text

def call_tool_sync(name, args):
    return asyncio.run(call_tool(name, args))