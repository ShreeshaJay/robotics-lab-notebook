# Costs: Tutorial 1 (Four VLAs)

**Total: about $52.1 of the $60 ceiling** ($49.6 metered from the runs' own clocks + about $2.4 estimated for runs
without a clock: fetches, the probe, failed starts). Compute: Modal, apps `t3-tut01-*`, at most 4 GPUs at once
after the 2026-11-20 coordinator cap. Time spent waiting in Modal's queue is not billed and is not counted.

Prices (modal.com/pricing, read 2026-11-20, per second, billed while a container runs):

| Resource | $/s | $/h |
|-|-|-|
| B200 | 0.001736 | 6.25 |
| H200 | 0.001261 | 4.54 |
| H100 | 0.001097 | 3.95 |
| A100 80 GB | 0.000694 | 2.50 |
| A10 (A10G) | 0.000306 | 1.10 |
| L4 | 0.000222 | 0.80 |
| T4 | 0.000164 | 0.59 |
| CPU (per physical core) | 0.0000131 | 0.047 |
| Memory (per GiB) | 0.00000222 | 0.008 |

Each row is the container's own clock (seconds from the run's `meta.json`, or the `nvidia-smi` sampler for runs
stopped by hand) times GPU + requested cores + requested memory (GR00T 4 cores / 24 GiB, lerobot 8 / 32,
PEAR 16 / 64, so a PEAR B200 second costs $0.002088). `python analysis/costs.py` prints the per-shard rows.

## Why the PEAR run used a B200

PEAR is 27B parameters in BF16: 51.1 GiB of weights loaded, plus about 64 KiB of KV cache per token (16 full-attention
layers x 2 x 4 KV heads x 256 x 2 bytes). One 260k-token episode therefore needs about 16 GiB of KV. An 80 GB
A100/H100 holds the weights and roughly one long episode at a time, so 50 episodes of ~57k tokens each (pilot
average) would run nearly one at a time: about 10+ GPU-hours on an A100 (~$25+) or ~$40 on an H100 at the
~35-50 tokens/s a single stream gets. A B200 (180 GB) held 1.65-1.77M tokens of KV, i.e. 11-16 concurrent episodes,
for about 500 tokens/s total, which made it the cheapest GPU per episode. H200 was the fallback; it was not needed.

## Pilot and projection (PEAR, before the full run)

- Pilot: 5 episodes (tasks 0, 2, 4, 6, 8, init state 0) at 5 concurrent, 1,878 s of B200 ($3.92), 3/5 successes.
- Time per episode: 505-1,630 s (mean 1,025 s) with 5 sharing the GPU; 18-40 turns; 22k-98k generated tokens
  (mean 57k); 245 tokens/s total.
- Projection for the other 45: 45 x 57k = 2.6M tokens at ~550 tokens/s with 16 concurrent = ~1.3 h, **~$12-15**,
  total tutorial ~$35-40. Under the ceiling, so the full run went ahead.
- What actually happened: failing episodes used far more tokens than the pilot's (mean 123k vs 33k for successes,
  since a failure runs the full 520 steps), and at 16 concurrent the KV cache filled (92-99%), so vLLM preempted
  and recomputed requests and generation fell to 5-280 tokens/s. I stopped two containers with episodes in flight
  (their in-flight work was lost), re-ran the unfinished episodes at 11 concurrent, and stopped that container at
  $52 total to stay under the ceiling. 35 of the 50 PEAR main episodes finished; see RESULTS.md.

## Log

| App | Run | GPU | Container s | Est. $ | Note |
|-|-|-|-|-|-|
| t3-tut01-probe | driver / EGL probe | T4 | ~60 | 0.01 | not metered, estimate |
| t3-tut01-groot, -pear, -lerobot | `fetch` (weights, assets) | CPU | ~1,600 | 0.17 | not metered, estimate |
| t3-tut01-pear | `dry` (CPU check of the episode loop) | CPU | ~240 | 0.01 | not metered, estimate |
| t3-tut01-groot | smoke (failed: init-state unpickling), smoke2 | L4 | 109 | 0.03 | |
| t3-tut01-groot | main: 50 episodes, 5 shards | A10 | 754 | 0.31 | |
| t3-tut01-groot | latency_l4: 5 episodes on L4 | L4 | 136 | 0.04 | same-GPU latency vs the baseline |
| t3-tut01-groot | horizon 1/4/16, 60 episodes, 6 shards | A10/L4 | 1,743 | 0.68 | before the 4-GPU cap |
| t3-tut01-groot | cf: 18 episodes | A10 | 354 | 0.14 | |
| t3-tut01-groot | perturb: 10 episodes | A10 | 186 | 0.08 | |
| t3-tut01-lerobot | g05 / smolvla states: 2 x 50 episodes | A10 | 1,832 | 0.88 | |
| t3-tut01-lerobot | horizon: g05 1/4/8/32, smolvla 1/4/8, 7 x 20 episodes | A10 | 8,756 | 4.21 | |
| t3-tut01-lerobot | cf: 2 x 18, perturb: 2 x 10 | A10 | 1,438 | 0.69 | |
| t3-tut01-pear | pilot attempts 1-3 (vLLM failed to start) | B200 | ~535 | 1.11 | not metered; FlashInfer JIT needed nvcc; attempt 2 was also stopped because its base image carried a license-acceptance banner |
| t3-tut01-pear | pilot: 5 episodes | B200 | 1,878 | 3.92 | |
| t3-tut01-pear | main, attempt 1 (engine OOM) | B200 | 215 | 0.45 | simulators' EGL contexts took GPU memory; fixed with CPU rendering |
| t3-tut01-pear | ext, attempt 1 (stopped with the OOM fix pending) | B200 | ~330 | 0.69 | not metered, estimate |
| t3-tut01-pear | main_a: 45 episodes at 16 concurrent, stopped | B200 | 5,191 | 10.84 | 18 finished; KV thrash |
| t3-tut01-pear | ext: 28 episodes at 16 concurrent, stopped | B200 | 4,564 | 9.53 | 15 finished (11 cf, 4 perturb) |
| t3-tut01-pear | resume: 40 episodes at 11 concurrent, stopped | B200 | 8,738 | 18.24 | 12 main finished; stopped at the budget line |
| **Total** | | | | **~52.1** | **metered 49.6 + estimated 2.4** |

Lessons for the next PEAR run: cap concurrency by KV, not by sessions (about 1.6M tokens / ~150k tokens per failing
episode = ~10 on a B200), and budget by tokens: a failing LIBERO-10 episode costs ~120k generated tokens, about
$0.50 on a B200 at full batch, against ~$0.004 for a GR00T episode on an A10.
