# 06 · Driving a Colab GPU from a coding agent with the official Colab MCP server

**Date:** 2026-11-08 · **Track:** Agentic tooling · **Status:** done · **Spend:** a few minutes of Colab Pro+ compute units (A100 runtime for about 5 minutes); no cloud bill

## Goal

Find out whether a local coding agent (Claude Code here) can use a Colab Pro+ GPU programmatically, using
Google's official [colab-mcp](https://github.com/googlecolab/colab-mcp) server (Apache-2.0), and how that
compares with renting a GPU VM directly.

## How it works (from reading the source at commit `b9ab389`)

- The server is about 450 lines of Python. It starts a WebSocket server on `localhost` (random port, random
  16-byte token) that only accepts connections whose Origin is `colab.research.google.com` or `colab.google.com`.
- Before connecting, it exposes a single tool, `open_colab_browser_connection`. Calling it opens
  `colab.research.google.com/notebooks/empty.ipynb` in your default browser, with the port and token in the URL
  fragment (fragments are not sent to Google's servers).
- The Colab page connects back to the local server and becomes the MCP server for the notebook tools. The local
  process just proxies them to the agent and sends `tools/list_changed` when the connection changes. That is why
  the client must support `notifications/tools/list_changed` and must run on the same machine as the browser.
- No Google credentials pass through the server. The browser tab is already signed in, and that is the only auth.

## What I ran

| Step | Result |
|-|-|
| `uv sync` (Python 3.13 is required; uv fetched it) and list tools over stdio | 1 tool before connecting |
| Call `open_colab_browser_connection`, user allows the connection in the Colab tab | Connected; 7 notebook tools appeared (below) |
| Add, update and run a system-check cell on the default runtime | CPU only: 2 vCPU, 12.7 GB RAM, Python 3.13 |
| User switches to a GPU runtime by hand (Runtime → Change runtime type) | The bridge stayed connected across the runtime change |
| Rerun the same cell | NVIDIA A100-SXM4-40GB, driver 580.82.07, 12 vCPU, 83.5 GB RAM, 236 GB disk free |
| 8192² bf16 matmul benchmark plus a 45 s `sleep` in one cell | 239 TFLOPS (about 77% of the A100's 312 peak); the 52 s call returned normally |
| `google.colab.runtime.unassign()` from a cell | Runtime released, so it stops using compute units |

Exposed tools: `add_code_cell`, `add_text_cell`, `update_cell`, `delete_cell`, `move_cell`, `get_cells`,
`run_code_cell`. Outputs come back as Jupyter output JSON (streams, errors with ANSI tracebacks). See
`results/tools.json` and `results/session_outputs.json`.

## Problems and fixes

- **The 60-second connect window is tight.** The tool waits 60 s for the browser tab to connect. My first attempt
  connected after about 2 min 50 s, once the agent had already given up. `bridge.py` calls the tool once and then
  polls `list_tools` for up to 6 minutes, so it never opens a second tab.
- **One MCP session per run.** A new server process means a new token and a new browser tab. `bridge.py` keeps one
  session open and runs commands from a file, so an agent can drive the notebook over many steps.
- **No runtime control.** No tool selects or changes the runtime type. A person has to pick the GPU in the
  Colab UI. Releasing it can be done from a cell (`runtime.unassign()`).
- **No file transfer or terminal tools.** Data goes in with code (`!wget`, `!pip`, Drive mounts), not through MCP.

## Colab Pro+ vs a rented GPU VM, for one person learning

| | Colab Pro+ via colab-mcp | GPU VM on a cloud (Spot) |
|-|-|-|
| Setup | Minutes. No quota request, no drivers, no images | Quota request, image choice, driver fixes, guardrail scripts |
| GPU choice | Whatever the UI offers (A100 40 GB here); picked by hand | Any type you have quota for (80 GB A100, L4, H100), picked in code |
| Automation | Cell-level only, and needs a signed-in browser tab on your machine | Fully scriptable (SSH, containers, cron), no browser needed |
| Long jobs | Session limits and idle disconnects apply | Bounded only by your own max-run-duration |
| Docker / custom drivers | No (so no Isaac Sim container) | Yes |
| Cost model | Prepaid subscription and compute units | Per second, billed to your project |

**Verdict:** good for interactive, short GPU work from an agent, such as notebooks, small fine-tunes and model
smoke tests that fit 40 GB. It doesn't replace a VM for unattended runs, 80 GB models, or containers.

## Reproduce

```bash
git clone https://github.com/googlecolab/colab-mcp src && git -C src checkout b9ab389
uv sync --project src --python 3.13
uv run --project src python -I bridge.py "$PWD/src" "$PWD"     # then allow the connection in the Colab tab
echo '{"id":"1","tool":"get_cells","args":{"cellIndexStart":0,"cellIndexEnd":5,"includeOutputs":true}}' >> cmds.jsonl
echo '{"id":"q","tool":"__quit__"}' >> cmds.jsonl
```

To use it straight from Claude Code instead, add a project `.mcp.json` with
`"command": "uvx", "args": ["git+https://github.com/googlecolab/colab-mcp@b9ab389"]`.
