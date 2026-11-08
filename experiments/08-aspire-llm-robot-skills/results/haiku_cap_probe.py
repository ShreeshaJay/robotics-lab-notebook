"""CPU-only probe: ask Claude Haiku 5.5 for one ASPIRE CaP program, then gate it
with ASPIRE's own sandbox validator and tool allowlist. No simulator, no GPU.

Usage: python haiku_cap_probe.py <aspire_repo> <env_file> <out_dir>
The API key is read from env_file into the process env and never printed.
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
import time
from pathlib import Path

repo, env_file, out_dir = (Path(p) for p in sys.argv[1:4])
out_dir.mkdir(parents=True, exist_ok=True)

for line in env_file.read_text(encoding="utf-8").splitlines():
    if line.startswith("ANTHROPIC_API_KEY="):
        os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
        break

import anthropic  # noqa: E402
from cap_harness.registry import LIBERO_PUBLIC_TOOL_NAMES, SHARED_PUBLIC_TOOL_NAMES  # noqa: E402
from cap_harness.runtime import _ProgramValidator  # noqa: E402

MODEL = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT, PRICE_CACHE_READ = 0.10, 0.50, 0.01  # $/MTok, <=100K-token prompts
BUDGET_USD = 3.00

api_ref = (repo / "docs" / "api-reference.md").read_text(encoding="utf-8")
api_ref = api_ref.split("\n## Real YAM", 1)[0]  # simulation surface only

TASK = (
    "Benchmark: libero-pro, suite libero_goal_swap, task 0. "
    "Instruction: 'open the middle drawer of the cabinet'. "
    "Franka Panda, tabletop kitchen scene, 20 Hz control, RGB-D cameras."
)
system = (
    "You write Code-as-Policy (CaP) programs for the cap-harness API. A program is plain "
    "Python executed in a sandbox: only `import math` is allowed, no classes, no `with`, "
    "no names or attributes starting with '_'. All tools and contract constructors are "
    "injected as globals exactly as documented. Use perception (segment_text, "
    "mask_to_point_cloud, ...) to localize objects; never hard-code scene coordinates. "
    "Assign a JSON-serializable summary to a global named `result`. "
    "Reply with exactly one ```python fenced block and nothing else."
)
user = f"API reference:\n\n{api_ref}\n\nTask:\n{TASK}\n\nWrite the program."

client = anthropic.Anthropic()
t0 = time.time()
msg = client.messages.create(
    model=MODEL,
    max_tokens=16000,  # worst case ~ $0.01 at Haiku 5.5 list price
    system=system,
    messages=[{"role": "user", "content": user}],
    output_config={"effort": "medium"},
)
latency = time.time() - t0
if msg.stop_reason == "refusal":
    sys.exit("refusal")
text = "".join(b.text for b in msg.content if b.type == "text")
match = re.search(r"```python\n(.*?)```", text, re.S)
code = match.group(1) if match else text
(out_dir / "haiku_program.py").write_text(code, encoding="utf-8")

u = msg.usage
cost = (
    (u.input_tokens or 0) * PRICE_IN
    + (getattr(u, "cache_read_input_tokens", 0) or 0) * PRICE_CACHE_READ
    + (u.output_tokens or 0) * PRICE_OUT
) / 1e6
assert cost < BUDGET_USD


def gate(src: str) -> dict:
    allowed = set(SHARED_PUBLIC_TOOL_NAMES) | set(LIBERO_PUBLIC_TOOL_NAMES)
    report: dict = {"lines": len(src.splitlines())}
    try:
        tree = ast.parse(src)
        _ProgramValidator().visit(tree)
        report["sandbox_validator"] = "pass"
    except (SyntaxError, ValueError) as exc:
        report["sandbox_validator"] = f"fail: {exc}"
        return report
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                called.add(f.id)
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                called.add(f"{f.value.id}.{f.attr}")
    tool_calls = sorted(c for c in called if c in allowed)
    builtins_ok = {"len", "float", "int", "abs", "min", "max", "range", "sorted", "sum",
                   "list", "dict", "tuple", "str", "bool", "enumerate", "zip", "round",
                   "print", "any", "all", "isinstance", "reversed", "set"}
    unknown = sorted(
        c for c in called
        if c not in allowed and c not in defined and c.split(".")[0] not in builtins_ok
        and not c.startswith("math.") and not c[:1].isupper()
        and "." not in c
    )
    report.update(tool_calls=tool_calls, unknown_bare_calls=unknown,
                  uses_perception=any(t in tool_calls for t in
                                      ("segment_text", "mask_to_point_cloud", "localize_object")),
                  assigns_result=any(isinstance(n, ast.Assign) and any(
                      isinstance(t, ast.Name) and t.id == "result" for t in n.targets)
                      for n in ast.walk(tree)))
    return report


reference = (repo / "examples" / "libero" / "open_middle_drawer.py").read_text(encoding="utf-8")
summary = {
    "model": MODEL,
    "stop_reason": msg.stop_reason,
    "usage": {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
              "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0)},
    "cost_usd_list_price": round(cost, 5),
    "latency_s": round(latency, 1),
    "haiku_program": gate(code),
    "reference_program_open_middle_drawer": gate(reference),
    "note": "Static gate only (sandbox AST validator + tool allowlist). No simulator rollout.",
}
(out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
