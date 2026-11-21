# Tutorial 1: Four VLAs, one honest comparison

Four robot policies on the **same 50 LIBERO-10 episodes** (10 tasks x LIBERO's fixed initial states 0-4), then six
extensions that ask what a single success rate hides. Results: [RESULTS.md](RESULTS.md). Spend: [COSTS.md](COSTS.md).

| Model | What it is | Checkpoint (pinned) | License |
|-|-|-|-|
| G0.5 | Galaxea VLA on a Qwen3.5-2B backbone, LIBERO fine-tune (LeRobot port) | `lerobot/g05_libero@9b7ee7d7` | G0.5 Community License (non-commercial; covers outputs) |
| SmolVLA | LeRobot's 0.45B VLA, LIBERO fine-tune | `lerobot/smolvla_libero@a24f1f38` | Apache-2.0 |
| GR00T N1.7 | NVIDIA's 3B VLA (Cosmos-Reason2 / Qwen3-VL backbone + flow-matching DiT), LIBERO-10 fine-tune | `nvidia/GR00T-N1.7-LIBERO@65854032`, folder `libero_10` | Weights: NVIDIA Open Model License. Code (Isaac-GR00T@708617f5): Apache-2.0 |
| PEAR | MIT Han Lab's 27B reasoning LLM (Qwen3.8-27B) RL-trained in simulation to output end-effector targets as text | `mit-han-lab/Qwen3.8-27B-PEAR@65176e43` (BF16) | Apache-2.0 (weights and code, pear@2a81b2ee) |

Other assets: LIBERO (MIT; `Lifelong-Robot-Learning/LIBERO@8f1084e3` for GR00T and PEAR, `hf-libero 0.1.4` inside
lerobot 0.7.0 for G0.5 and SmolVLA), `lerobot/libero-assets@0b3ea86b`, `Qwen/Qwen3-VL-2B-Instruct@89644892`
(Apache-2.0, see "GR00T backbone" below), robosuite 1.4.0 (MIT), MuJoCo (Apache-2.0).

## What "the same episode" means here

Episode (task t, i) = LIBERO's fixed initial state `init_states[i]` of task t, numpy seed i, ten no-op settle steps
with the gripper open, then at most **520 control steps at 20 Hz**. That is exactly what the baseline did with
`lerobot-eval --env.init_states=true --seed=0 --eval.batch_size=1 --eval.n_episodes=5` (lerobot 0.7.0), and
`harness/libero_common.py` reproduces it for GR00T and PEAR, which do not run through lerobot. Every model is scored
by LIBERO's own BDDL goal check. G0.5 and SmolVLA's main numbers are the **saved** per-episode results of
`robotics-lab-notebook/experiments/05-g05-lerobot/results/rerun50` (2026-11-08); they were re-run only where an
extension needs new conditions or simulator states.

What differs between models, on purpose, because each runs with its own documented inputs:

| | G0.5 | SmolVLA | GR00T N1.7 | PEAR |
|-|-|-|-|-|
| Harness | `lerobot-eval` 0.7.0 | `lerobot-eval` 0.7.0 | own loop, NVIDIA's obs/action code | own loop, PEAR's prompts and controller |
| Cameras | agentview + wrist, 512 px | agentview + wrist, 360 px | agentview + wrist, 256 px | agentview + wrist, 512 px, PNG |
| Output per call | 32 actions, 10 executed | 50 actions, 50 executed | 16 actions, 8 executed | one end-effector target as JSON; PEAR's MotionController drives the arm there in closed loop (up to 100 native steps) |
| Language | instruction | instruction | instruction | instruction with left/right swapped (PEAR's convention: its frames are rotated 180 degrees) |

## Deviations and things to know

- **GR00T backbone.** The GR00T checkpoint names `nvidia/Cosmos-Reason2-2B` as its backbone, a gated repo our token
  cannot open, and we do not accept gates. The checkpoint itself carries all backbone weights (493 tensors), so only
  the backbone's *config and processor* were needed. We used `Qwen/Qwen3-VL-2B-Instruct` (ungated, Apache-2.0)
  for those: its config is identical to LeRobot's hard-coded copy of the Cosmos-Reason2-2B config
  (`groot_n1_7.py::_cosmos_reason2_qwen3_vl_config`). The tokenizer and chat template are Qwen3-VL's; if
  Cosmos-Reason2 changed them, GR00T's score here is a lower bound. The score (45/50, 90%) is within our interval of NVIDIA's
  reported 94.35% on LIBERO-10, which suggests no large mismatch. `modal_groot.py::fetch` records the edit
  (`config.json.orig` is kept).
- **GR00T settings vs NVIDIA's script:** our init states, seeds, settle steps and 520-step budget (NVIDIA's
  `rollout_policy.py` uses `env.reset()` and 720 steps).
- **PEAR checkpoint and the 39.0% claim.** The paper's 39.0% "zero-shot on LIBERO-Long" is the **stage-1 checkpoint
  (step 148), trained on LIBERO Goal/Spatial/Object only**, which is not released. The released
  `Qwen3.8-27B-PEAR` continued training on all four LIBERO suites (stage 2, which adds LIBERO-Long and its 50 initial
  states per task) and RoboDojo (stage 3). So on our episodes it is **not zero-shot**: it has trained on these exact
  initial states with RL. The paper reports 53.2% on LIBERO-Long for its LIBERO-only stage-2 model (xhigh effort) and
  no per-suite LIBERO number for the released final model. Our run therefore checks neither published number
  directly; see RESULTS.md.
- **PEAR settings:** xhigh reasoning effort (the chat template's and the paper's default), temperature 1.0,
  top-p 0.95, top-k off (`pear/configs/embodied.yaml`), the full conversation with its reasoning kept in context, a
  256,000-token response budget per episode (`examples/eval/eval_libero.sh`). Served by vLLM 0.28.0 (PEAR's
  server pin) on one B200 instead of PEAR's verl/Ray cluster; the agent loop is re-implemented in
  `harness/pear_runner.py` from PEAR's files, the controller and prompts are vendored verbatim
  (`harness/pear_vendor/`). Two vLLM settings differ from defaults to avoid FlashInfer kernels that need a CUDA
  compiler at start-up on Blackwell: `--attention-backend FLASH_ATTN` and `VLLM_USE_FLASHINFER_SAMPLER=0`.
  PEAR episodes ran several at a time on one GPU, so its per-turn latency includes batching. The simulator
  processes render on the CPU (OSMesa) after the pilot: at 23 concurrent episodes their GPU EGL contexts took the
  memory vLLM had reserved and crashed the engine.
- **PEAR coverage:** 35 of the 50 main episodes, 11 of 18 counterfactual and 4 of 10 perturbed episodes finished.
  At 16 concurrent episodes the KV cache filled and throughput collapsed, so two containers were stopped and the
  unfinished episodes re-run at 11 concurrent; that container was stopped at the budget line (COSTS.md). Unfinished
  episodes are listed in `results/raw/pear/ext_stopped.meta.json` and `resume.meta.json`, and RESULTS.md gives
  bounds for them.
- **GPU types:** the baseline ran on a GCP L4. Modal had no L4 capacity for part of the day, so the GR00T and
  G0.5/SmolVLA re-runs requested `["L4", "A10"]` and the GPU actually used is recorded per run. Latency is only
  compared on the same GPU type.
- **G0.5 re-runs** use the license decision recorded in the baseline (accepted 2026-11-08 for personal,
  non-commercial research). Nothing here uses G0.5 outputs commercially.
- **lerobot re-runs** add `--eval.use_async_envs=false` (the env patches run in-process); physics, seeds and init
  states are unchanged. The agreement of each re-run with the saved run is reported.

## Files

| Path | What |
|-|-|
| `harness/libero_common.py` | Episode definition, BDDL step tracker (ext. 3), counterfactual instructions (ext. 2), perturbation (ext. 4) |
| `harness/groot_runner.py` | GR00T loop (NVIDIA's observation/action conversion) |
| `harness/pear_runner.py`, `harness/pear_vendor/` | PEAR agent loop against vLLM; PEAR's controller and prompts (Apache-2.0) |
| `harness/lerobot_tut_eval.py` | The baseline's `timed_eval.py` plus env hooks for G0.5 / SmolVLA |
| `modal_groot.py`, `modal_pear.py`, `modal_lerobot.py` | Modal apps `t3-tut01-groot`, `t3-tut01-pear`, `t3-tut01-lerobot` |
| `analysis/stats.py`, `analysis/analyze.py` | Wilson, exact McNemar, paired bootstrap, power; all tables and figures |
| `results/raw/` | Raw per-episode JSON(L) per run and per-run metadata (PEAR: `pilot`, `main_a`, `main_b` = main episodes; `cf`, `perturb` = extensions) |
| `results/summary.json`, `results/*.csv`, `results/figures/*.png` | Derived numbers and figures |

## Reproduce

Modal client 1.6.4. `HF_TOKEN` (or the `HF_TOKEN_ASPIRE` line of the dotenv file named by `T3_DOTENV`) is passed to
the containers as an ephemeral Modal Secret, never as an argument. Keep at most 4 GPUs running at once.

```bash
modal run modal_groot.py::fetch && modal run modal_pear.py::fetch && modal run modal_lerobot.py::fetch   # CPU only
modal run modal_groot.py::main --tag main --plan-name main --shards 5
modal run modal_pear.py::main --tag pilot --plan-name pilot --conc 5 --gpu B200
modal run modal_pear.py::main --tag main --plan-name main_rest --conc 11 --gpu B200     # what we ran: 16, then "resume" at 11
modal run modal_pear.py::main --tag ext --plan-name "ext:2" --conc 11 --gpu B200        # counterfactual + perturbation
modal run modal_groot.py::main --tag latency_l4 --plan-name latency --gpu L4
modal run modal_groot.py::main --tag horizon --plan-name "horizon:1,4,16" --shards 3
modal run modal_groot.py::main --tag cf --plan-name "cf:2" --shards 2
modal run modal_groot.py::main --tag perturb --plan-name perturb --shards 2
modal run modal_lerobot.py::main --jobs "smolvla:states,g05:states"
modal run modal_lerobot.py::main --jobs "g05:horizon:1,g05:horizon:4,g05:horizon:8,g05:horizon:32"
modal run modal_lerobot.py::main --jobs "smolvla:horizon:1,smolvla:horizon:4,smolvla:horizon:8"
modal run modal_lerobot.py::main --jobs "g05:cf:2,smolvla:cf:2,g05:perturb,smolvla:perturb"
python analysis/analyze.py && python analysis/costs.py
python analysis/stats.py 0.5 0.6 0.05 0.8 0.2      # power calculator
```
