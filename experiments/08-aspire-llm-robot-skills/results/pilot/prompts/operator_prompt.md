You are the coordinator of a ONE-TASK ASPIRE iterative-debugging campaign in this repository (~/aspire, ASPIRE commit fa00f0ef). Work non-interactively; there is no human to answer questions.

Read first: .claude/skills/iterative-debugging/SKILL.md, .claude/skills/iterative-debugging/main-agent-prompt.md, docs/libero-pro.md, docs/api-reference.md, docs/run-artifacts.md. Do NOT read real-robot material: AGENTS.md, docs/system2*.md, docs/yam-real.md, src/cap_harness/yam_real/, .agents/.

Task list (write it to /home/shree/campaign_tasks.json and initialize with init_run.py using the LIBERO defaults: development seeds 51-65, unseen-check 66-80, held-out count 50):
[{"benchmark": "libero-pro", "suite": "libero_goal_swap", "task_id": 0, "task_name": "open_the_middle_drawer_of_the_cabinet"}]

Environment (already prepared, do not change it):
- .venv-libero is bootstrapped; use CAP_HARNESS=.venv-libero/bin/cap-harness and INIT_MODE=seeded, MAX_STEPS=1000.
- Provider services are already running on GPU 0 under an external supervisor: sam3 :8114, contact_graspnet :8115, pyroki :8116, curobo :8118. Never start, stop or restart them. If one is down, write that in the summary and stop.
- RUN_ENV variables are exported in your environment (LIBERO_CONFIG_PATH, MUJOCO_GL=egl, PYOPENGL_PLATFORM=egl, EGL_PLATFORM=device, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1, MUJOCO_EGL_DEVICE_ID=0, CUDA_VISIBLE_DEVICES=0).
- One GPU (GPU 0), one task. For this one-task campaign the Stage 1 worker owns GPU 0 and runs its own development and unseen-check episodes (the normal one-subagent-per-GPU mode); the Single-GPU section's "coordinator runs every episode" rule is waived because there is nothing to contend with.

What to do:
1. init_run.py, gen_progress.py.
2. Stage the worker prompt with dispatch.py worker --suite libero_goal_swap --task-id 0 --gpu 0, then dispatch it with the Agent tool (subagent_type general-purpose) and run_in_background: false. This session is non-interactive (claude -p), so background agents would be lost: always wait for agents in the foreground.
3. When the worker returns, do the coordinator checks of "On SUBAGENT completion", EXCEPT do NOT run the Stage 2 held-out evaluation: the operator runs run_validation.py on seeds 1-50 after this session ends.
4. Act on the skill grades. If the task proposed skills, stage and dispatch a verifier (dispatch.py verify ..., foreground), and promote only verified proposals into .claude/skills/iterative-debugging/skills/.
5. Write $RUN_ROOT/coordinator_summary.md: development pass count for every version (v00, v01, ...), the selected version, the unseen-check pass count, the skill grades, proposals and verifier outcome, and anything that broke. Then end the session.

Limits:
- Every Bash command must finish within 600000 ms. Always use run-batch with --workers 4 for sweeps.
- Hard deadline: 2026-11-09 06:00 UTC (check with `date -u`). Tell the worker this deadline. If it is within 25 minutes, the worker must stop debugging, select the best version so far as fix_code.py, run the unseen-check sweep, and return. Skip the verifier if less than 30 minutes remain.
- LLM spend is capped, so be economical: do not re-read files you already read, read large logs with grep/head, and look at keyframes only when the trace does not explain a failure.
- Do not pip install, use sudo, use the network beyond the local services, or touch anything outside ~/aspire and /home/shree/campaign_tasks.json.
