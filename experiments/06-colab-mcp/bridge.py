"""Keep one colab-mcp session open. Open the browser once, poll for the Colab tab to connect (up to 6 min),
save the tool list, then execute commands appended to cmds.jsonl ({"id","tool","args"}) and write results to out.jsonl.
A command {"tool":"__quit__"} ends the session."""
import asyncio, json, sys, os, time
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
SRC, D = sys.argv[1], sys.argv[2]
def log(o):
    with open(os.path.join(D, "out.jsonl"), "a", encoding="utf-8") as f: f.write(json.dumps(o, default=str) + "\n")
async def main():
    t = StdioTransport(command="uv", args=["run", "--project", SRC, "colab-mcp"])
    async with Client(t, timeout=600) as c:
        try:
            r = await asyncio.wait_for(c.call_tool("open_colab_browser_connection", {}), 90)
            log({"event": "first_call", "result": r.data})
        except Exception as e:
            log({"event": "first_call_err", "err": repr(e)})
        t0 = time.time()
        while time.time() - t0 < 360:
            names = [x.name for x in await c.list_tools()]
            if len(names) > 1: break
            await asyncio.sleep(5)
        tools = await c.list_tools()
        json.dump([{"name": x.name, "description": x.description, "schema": x.inputSchema} for x in tools],
                  open(os.path.join(D, "tools.json"), "w"), indent=1)
        log({"event": "tools", "names": [x.name for x in tools]})
        if len(tools) <= 1: return
        done, path = set(), os.path.join(D, "cmds.jsonl")
        while True:
            if os.path.exists(path):
                for line in open(path, encoding="utf-8"):
                    if not line.strip(): continue
                    cmd = json.loads(line)
                    if cmd["id"] in done: continue
                    done.add(cmd["id"])
                    if cmd["tool"] == "__quit__": log({"event": "quit"}); return
                    try:
                        r = await c.call_tool(cmd["tool"], cmd.get("args", {}), raise_on_error=False)
                        log({"id": cmd["id"], "is_error": r.is_error,
                             "content": [getattr(b, "text", str(b)) for b in r.content],
                             "data": r.structured_content})
                    except Exception as e:
                        log({"id": cmd["id"], "err": repr(e)})
            await asyncio.sleep(2)
asyncio.run(main())
