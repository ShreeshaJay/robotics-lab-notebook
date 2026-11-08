# 02 · Inspect Robots: running an open robot-eval harness with a frontier LLM

**Date:** 2026-11-07 · **Track:** Evals · **Status:** done (CPU mock only) · **Spend:** ~$0.007 of API usage

## Goal

Learn how an open robot-evaluation harness is put together (tasks, scenes, scoring, grading, logs) by
reading the paper behind it and running the smallest eval I can on a laptop CPU with a cheap LLM as the
robot policy.

- Paper: *The Bitter Lesson Comes to Robotics*, Leet et al., arXiv:2610.25287 (v2, Oct 2026; CoRL 2026 TAG workshop).
- Harness: [Inspect Robots](https://github.com/robocurve/inspect-robots) by Robocurve, MIT license. Its
  own paper is cited as arXiv:2610.06306 (not read here).

## What the paper does

- Nine frontier multimodal LLMs (15 model/effort/service-tier pairs) drive a **physical** bimanual
  I2RT YAM arm directly, by calling a `move_to` tool with a 7-number end-effector pose per arm
  (x, y, z, yaw, pitch, roll, gripper). IK turns poses into joint angles. Two safety guards clip joint
  limits and per-action motion; pitch is capped at ±0.6 rad. The model also has `done` and `give_up`.
- Observations: proprioception plus three cameras (two wrist, one overhead). **Budget: 40 actions per trial.**
- Tasks: six RoboDojo tabletop tasks (Stack Bowls, Store in Safe, Sort Objects, Cap Pen, Pack and Pour
  Fruit, Stand Up Bottles), 20 trials each per pair, so about 1,800 real-robot trials in total.
- **Scoring:** each task is split into cumulative subtasks (Store in Safe: 25/50/75/100%). A trial gets
  the highest subtask reached at any point, even if later undone.
- **Grading: an LLM judge** (GPT-6 Astra, high effort) watches the three camera videos and grades every
  trial. That judge is also one of the evaluated models, and it is not validated against human labels.
- **Headline numbers:**
  - Best mean scores: about 48% (Gemini 4 Argon 48.4%, Opus 5.5 at xhigh about 48.3%).
  - Score doubling time about 54 days (95% CI 36 to 60 days) over July to October 2026.
  - Fastest: 54.4 subtasks/hour (GPT-6 Astra with priority tier, about 1.1 min per subtask), 4.4x better than the earlier best.
  - Cheapest: $0.38 per subtask (Gemini 3.8 Flash), 6.2x better than the earlier best. Others: $0.97 to $4.51 per subtask.
  - Cost comes from token counts at list prices. Wall-clock time includes robot motion. Trials take "several minutes".
- **Failure modes:** grasps that close on nothing (32% to 64% of grasp attempts, by model), unrecoverable
  errors (knocking things off the table, launching a pen cap). Stand Up Bottles was almost never solved
  (6 of 180 trials uprighted one bottle).
- **Limitations (from the text; there is no limitations section):**
  - The 40-action cap likely understates what the models can do.
  - Only one robot platform.
  - The trend fits rest on 3 or 4 record-setting points.
  - The judge is not validated, and it overlaps with the models it grades.

## What the harness is

- It is "Inspect AI for robotics". You swap two inputs: a **Policy** (a VLA or an LLM agent) and an
  **Embodiment** (a real robot or a sim). Both are scored against a **Task** = scenes + rubrics/scorers.
  `eval()` runs the closed loop and writes an immutable JSON `EvalLog`. It also writes per-trial
  records, actions, the agent transcript and the raw LLM traffic ("wire" capture: request bodies, no
  headers), and has an HTML/Rerun viewer.
- The core depends only on NumPy. Optional plugins:
  - `inspect-robots-agent`: LLMs as policies, using raw HTTP to Anthropic, OpenAI, Gemini, OpenRouter
    and others.
  - Isaac Lab sim (**needs an NVIDIA GPU + Omniverse**).
  - ROS bridge.
  - XPolicyLab: about 40 served VLAs.
  - CaP-X: code as policy.
  - Voice.
- **CPU backend:** only `cubepick`, a deterministic 2D toy world. A point must reach a cube, the action
  is (dx, dy), and the camera is a 32x32 image. There is no CPU physics sim of homes or tabletops.
- **Grading options:**
  - Programmatic scorers that read the recorded trajectory: `success_at_end`, `min_distance_to_goal`,
    `episode_length`, and others.
  - `operator`: a human types a verdict at the terminal.
  - `vlm`: an **LLM judge** that sees the first and last camera frames plus a rubric, and writes its
    verdict where the human one would go.
  - Verdicts can be corrected after the run (`inspect-robots edit` / `score`) without rewriting the log.
- Tasks are Python: `Task(name, scenes=[Scene(id, instruction, init_seed, metadata, rubrics)], rubrics=[Rubric(name, scorer, kind=capability|safety, text)], max_steps | max_seconds, epochs=Epochs(count, reducer))`.
  Tasks are grouped into a versioned `Benchmark`. Reducers include `pass_at_k`.

## Steps

1. Read the arXiv abstract and HTML. Read the harness README, `AGENTS.md`, `docs/guide/*.md`, and the agent plugin README and source.
2. Cloned `robocurve/inspect-robots` (main at `772019598e`; latest tag `v1.0.0`) into a scratch folder.
   Read the CLI, the `.env` loader and the agent plugin before running anything. Checks:
   - The only outbound hosts are LLM provider APIs.
   - There is no telemetry.
   - The wire capture stores request bodies only, not auth headers.
   - The CLI auto-loads a `.env` from the current directory, so I ran from a directory without one.
3. Fresh venv (`uv venv --python 3.11`) and pinned PyPI installs: see [`requirements.txt`](requirements.txt).
4. [`run_cubepick_agent.py`](run_cubepick_agent.py):
   - Loads `ANTHROPIC_API_KEY` from a dotenv path into the process only (never printed).
   - Runs the free scripted baseline, then the LLM agent on 3 seeded scenes, one trial each, with a
     $0.50 spend guard between trials.
   - Checks that the key string appears in no log file.
5. Sketched a household benchmark in the task format: [`household_turnover_task.py`](household_turnover_task.py).
   It builds the objects with the installed library, but there is no embodiment to run it on.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -I run_cubepick_agent.py --dotenv /path/to/.env --scenes 3 --log-dir logs
```

Settings: `model=anthropic/claude-haiku-5-5`, `wire=messages` (native Anthropic API), `effort=low`,
`images=always`, `image_horizon=none`, `max_llm_calls=25`, `max_steps=80`.

## Results

Task: `cubepick-reach` (reach the cube). Scorers: `success_at_end`, `episode_length`, `min_distance_to_goal`.
Raw numbers are in [`results/summary.json`](results/summary.json).

| Policy | Scene | Success | Steps | Min dist | LLM calls | Out tok | Cache-read tok | Wall s | Cost $ |
|-|-|-|-|-|-|-|-|-|-|
| scripted | 0 | 1 | 10 | 0.037 | 0 | 0 | 0 | 0.05 | 0 |
| scripted | 1 | 1 | 8 | 0.000 | 0 | 0 | 0 | 0.01 | 0 |
| scripted | 2 | 1 | 8 | 0.000 | 0 | 0 | 0 | 0.01 | 0 |
| Haiku 5.5 | 0 | 1 | 15 | 0.0002 | 8 | 1,977 | 24,226 | 11.5 | 0.0016 |
| Haiku 5.5 | 1 | 1 | 11 | 0.0006 | 6 | 1,689 | 15,960 | 10.1 | 0.0013 |
| Haiku 5.5 | 2 | 1 | 11 | 0.0008 | 6 | 1,699 | 16,543 | 9.5 | 0.0013 |

- Agent: **3/3 success**. It used 11 to 15 env steps against 8 to 10 for the scripted policy, but ended much
  closer to the cube. Each run took about 10 s and used 6 to 8 LLM calls.
- Tokens for the 3 trials: 80 uncached input, 7,632 cache-write, 56,729 cache-read and 5,365 output.
  **Cost $0.0042.**
- **Gotcha:** the first smoke run failed on call 2 with HTTP 400, "Invalid `signature` in `thinking`
  block ... bound to a different conversation". The agent's default `image_horizon=2` rewrites older
  messages to stub out old camera frames. That edit breaks the API's preserved-thinking check on newer
  Claude models. The fix was `image_horizon=none`, which sends the full image history (cheap here with
  32x32 frames, but costly with real cameras). Another fix would be thinking `prefix_mismatch_behavior: drop_block`
  under the `thinking-binding-controls-2026-08-01` beta, which the harness does not expose. This is
  worth reporting upstream.
- The transcript shows the agent **ignored the camera** ("the top camera image is too dark to use") and
  solved the task by arithmetic on the privileged `cube_pos` in the state readout: five 0.1 diagonal
  steps, then one fine correction. So this run tests the plumbing, not perception or manipulation.

## Cost

| Item | Amount |
|-|-|
| LLM API (Haiku 5.5, 2 smoke runs + 3-trial run) | about $0.0065 (cap was $10) |
| Compute | laptop CPU, no GPU, no cloud |
| My time | about 1 hour (reading about 30 min, setup + runs about 15 min, write-up) |

Before the run I estimated under $0.05 for 3 trials: at most 25 calls × about 5k tokens at $0.10/$0.50 per
MTok. Cache-read and cache-write prices are **assumed** at 0.1x and 1.25x of the input price.

## What this teaches about robot evals

1. **The two swappable inputs make it work.** Keeping the policy, the embodiment and the task separate
   means the same task runs a VLA, an LLM agent or a scripted baseline on a sim or a real rig, and the
   logs are comparable. Always run a free scripted or random baseline next to the expensive policy.
2. **Grading is the weak point, not running the robot.** Real robots have no success oracle, so grading
   falls to a human or a VLM judge. The paper relies on an unvalidated LLM judge that is also one of
   the contestants. The harness handles this well by putting human and VLM verdicts in the same slot,
   recording judge config in the log, and allowing post-hoc corrections. Anyone publishing
   numbers should measure judge-vs-human agreement on a sample.
3. **Tasks need partial credit and safety rubrics.** Cumulative subtasks ("max reached") and separate
   `safety` rubrics show far more than one pass/fail bit. A safety rubric only makes sense if the
   policy can `give_up`.
4. **Privileged state leaks make tasks trivial.** If the observation includes the goal position, an LLM
   skips perception entirely. Real evals should strip privileged state from the policy's view and keep it
   for the scorer only.
5. **Report cost and time, not just success.** The paper's subtasks/hour and $/subtask measures are the
   useful ones for deployment. The per-trial token accounting in the logs makes them cheap to compute.
6. **Harness and API details can break runs.** A context-trimming setting broke the run on a current
   model. Pin versions, record the model and effort, and do a 1-trial smoke run first.

## Needed for anything beyond the toy

- A physics sim of homes or tabletops: the Isaac Lab plugin needs an NVIDIA GPU, Omniverse and Isaac Lab,
  which means accepting NVIDIA's terms. I did not try it.
- Or a real rig with its plugin (for example `inspect-robots-yam` for I2RT YAM arms).
- A household task (make bed, restock toiletries, tidy) fits the format as-is. See the sketch: one Task per
  chore, scenes varied by mess level, capability and safety rubrics graded by an operator or a VLM. Partial
  credit per subtask would need a small custom scorer.

## Next steps

- Run `--grader vlm` on cubepick with a cheap judge and compare it against the privileged `success_at_end`.
  That is a cheap first judge-agreement measurement.
- Hide `cube_pos` from the agent (custom embodiment wrapper) to force vision-based control on the toy.
- Try a MuJoCo or ManiSkill CPU tabletop as a custom embodiment. The adapter API looks small.
- Report the `image_horizon` / preserved-thinking issue upstream.
