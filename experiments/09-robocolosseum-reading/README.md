# 09 · RoboColosseum: blind real-robot evaluation at scale (reading)

**Track:** Evaluation protocols / benchmarks · **Status:** desk review plus inspection of a few small LeRobot metadata and parquet files. No videos downloaded, no GPU, no cloud, $0 · **Date:** 2026-11-08

## Goal
Work out how RoboColosseum runs blind real-robot evaluations, how it scores and ranks policies, what it found, what it has actually published on Hugging Face (format, fields, license), and which parts a household-task evaluation platform in simulated homes could reuse.

## What it is
RoboColosseum is a distributed real-robot benchmark. Policies are served through one interface (a policy server talks to a robot client over a relay) and run on a fleet of more than 100 robots operated at local sites. The paper's snapshot covers four embodiments (SO-101 arm, Franka in the DROID setup, bimanual YAM, Unitree G1 humanoid). A wheeled Earth Rover Mini+ is being added on the site and in the client code. It is run by an academic group (paper: arXiv 2611.02497, 2 Nov 2026, 27 authors). Labs can either run their own inference server against the policy API or hand over a checkpoint to be served next to the robot.

Scale numbers do not agree across sources:
| Source | Numbers |
|-|-|
| Paper | 124 sites, 5,699 Open Track pairwise comparisons, 2,500 Fine-tune trial slots (20 tasks x 25 trials x 5 VLAs), 8 models, 41-day window (17 Sep to 28 Oct 2026), up to 30 full-time operators |
| Website | 140+ scenes, 5 active embodiments |
| LinkedIn launch post | 20,000+ trials, 6,800 tasks, 150 scenes, 10,000 free eval hours for academic labs |

Two runs per comparison gives about 11.4k Open Track runs, plus 2.5k Fine-tune trials, so about 14k runs in the paper. The "20k trials" figure probably includes runs after the paper's cutoff or runs that were excluded, but I could not confirm that. The paper itself is inconsistent in one place: Sec. 4.1 calls the 5,699 comparisons "tasks".

## Evaluation protocol

### Open Track (zero-shot, blind, pairwise)
- **Blinding:** the evaluator sees "A" and "B", never model names.
- **Matched conditions:** both models run one after the other on the same robot, scene, objects, setup, instruction and timeout.
- **Tasks and scenes:** the evaluator makes up a feasible task on the spot from objects at hand and writes a natural-language instruction. Sites include offices, libraries, parks, dorms, homes, food courts and factories in Singapore, Malaysia and China, with robots also in the US and UK. There is no fixed task list.
- **What gets recorded per comparison:** the winner (A, B or tie), a progress score for each run (0 to 100 partial success in the client), a success yes/no, free-text feedback, and a 1 to 10 difficulty rating from the human.
- **Frontier agents:** general LLM agents (GPT-6 Astra, Claude Opus 5.5, Grok 4.7) control the arms through a tool interface with one call per turn (`move` with absolute joint targets plus a note, `done`, or `give_up`). Limits are 30 calls, 800 control steps and 120 s per call. An agent saying `done` does not count as success.
- **Safety:** a rollout stopped for damage risk counts as a failure.

### Fine-tune Track (fixed tasks, matched data)
- 5 fixed tasks per embodiment, 20 in total. Each policy gets the same 100 teleoperated demos, recorded at the evaluation site, and is fully fine-tuned with its own official recipe. Each policy then gets 25 trials per task.
- **Success rate (SR):** the rollout meets the task's full success requirements. For example, open_drawer needs at least 70% of maximum travel, held for 1 s or longer.
- **Progress score (PS):** each task has K equally weighted milestone criteria, which do not have to be met in order. A rollout's progress is the fraction of criteria met, and PS is the mean over trials. Example: close_laptop scores contact with the lid, then halfway closed, then nearly closed, then fully closed. Full per-task rubrics are in the paper's appendix. I only saw parts of them.

| Embodiment | Fine-tune tasks |
|-|-|
| SO-101 | stack_bowls, remove_screwdriver, upright_bottle, stack_cubes, cube_drawer |
| Franka / DROID | close_laptop, mug_on_coaster, open_drawer, marker_in_holder, towel_in_basket |
| Bimanual YAM | dustpan, drawer, microwave, stack_cups, clean_table |
| Unitree G1 | plate_into_rack, carry_box, fold_towel, load_clothes_washing_machine, hang_keys |

### Scoring and ranking
- **Primary rating, Difficulty-Calibrated Rasch (DCR):** a 1PL item-response model in which P(success) = sigmoid(policy ability - task difficulty). The human 1 to 10 difficulty rating is used as the prior on task difficulty. Policies are ranked by posterior mean ability, which is rescaled to look like Elo (1500-centred). The authors say it is not sequential Elo.
- **Baselines:** Bradley-Terry-Davidson (handles ties), sequential Elo, and average pairwise score.
- **Uncertainty:** 95% credible intervals for DCR. Bootstrap and Wilson intervals for rates.
- **Active matchmaking:** after a warm-up, the next pair to run is the one expected to reduce uncertainty the most, as in Chatbot Arena. It reached about 90% agreement with the full-data ranking after 400 to 500 evaluations, compared with about 1,000 for uniform sampling.
- **Agreement checks:** Kendall's W across embodiment rankings, Spearman's rho between the Open Track and Fine-tune rankings, and pairwise agreement across site splits.
- **Task categories:** Open Track instructions are sorted into 6 categories (Pick/Place, Stack/Arrange, Open/Close, Reorient, Deformable, Tool/Interaction) by an open LLM labeler that also sets a needs-review flag.

## Results (who ranked how)
Fine-tune Track, mean PS (SR in parentheses), best to worst:
| Embodiment | 1 | 2 | 3 | 4 | 5 |
|-|-|-|-|-|-|
| SO-101 | GR00T N1.7 89.0 (82.4) | pi0.5 85.4 (77.6) | MolmoAct2 71.8 (59.2) | G0.5 58.8 (40.0) | LingBot-VLA v2 55.2 (36.8) |
| Franka / DROID | pi0.5 91.2 (84.0) | MolmoAct2 78.2 (54.4) | LingBot 59.4 (40.8) | GR00T 56.4 (42.4) | G0.5 44.8 (16.0) |
| Bimanual YAM | pi0.5 97.2 (93.6) | G0.5 89.1 (75.2) | MolmoAct2 72.1 (49.6) | GR00T 52.6 (20.0) | LingBot 48.3 (30.4) |
| G1 humanoid | LingBot 23.2 | GR00T 22.7 | MolmoAct2 19.8 | pi0.5 16.3 | G0.5 13.3 (all 0% SR) |

Open Track leaders by DCR: GPT-6 Astra on DROID, Claude Opus 5.5 on SO-101 and YAM. The exact DCR values are only in a figure, which I did not extract.

## Main findings
1. **General LLM agents keep up with VLAs zero-shot.** Across 901 paired agent-vs-VLA comparisons, agents made more progress in 394, VLAs in 265, and 242 were ties. The size of the gap depends on the model, and one agent was behind on average.
2. **Full success on open-world tasks is rare.** It falls from about 31% for tasks rated 1 to 4 in difficulty to about 0.4% for tasks rated 9 to 10. Partial progress is what separates policies.
3. **Rankings depend on the embodiment** (Kendall's W = 0.24 across fine-tune suites).
4. **Zero-shot ranking does not predict ranking after fine-tuning.** Spearman's rho was -0.20, -0.40 and 0.20 on the three shared embodiments, with rank reversals on all three.
5. **The humanoid is unsolved.** No G1 trial succeeded, but progress scores still order the policies.
6. **Reliability varies.** Rankings agree about 85% across site splits on DROID and YAM, but only 59% on SO-101. Overall leaderboard agreement reaches 90% at about 1.1k evaluations and 95% at about 4.8k.

## Stated limitations
- It is a snapshot. Which models were tested depends on hardware compatibility.
- The Fine-tune Track has only 5 tasks per embodiment.
- Each model uses its own official fine-tuning recipe, so the results do not isolate architecture.
- Stability is measured against the authors' own full dataset, not an external ground truth.
- My own addition: Open Track tasks are made up by the evaluators, so the task distribution depends on the operators and the objects at each site.

## Rollout data on Hugging Face
Org: [huggingface.co/RoboColosseum](https://huggingface.co/RoboColosseum). Checked through the HF API on 2026-11-08.

**What is public:** 20 datasets, one per Fine-tune task. Each holds the **100 teleoperated training demos**, about 2,000 episodes and **about 34 GB** in total (0.13 to 5.7 GB per dataset). There are also **100 model repos**, one fine-tuned checkpoint per task per VLA.

**What I could not find:** a public dataset of evaluation rollouts, either the Open Track A/B runs or the 25-trial Fine-tune evaluations. The site and the launch post say every evaluation run is published, failures included. The client code (`hub_upload.py`, `evaluation.py`) does upload each evaluation to a dataset URL set per operator (Hugging Face or ModelScope), with a per-run `meta/colosseum.json` holding outcome, preference and feedback. But no such repo is listed under the org, its collections ("RoboColosseum Datasets" has 4 items), or a search for "colosseum". Either the eval rollouts live under operator accounts or they have not been released yet. **Open question to check before relying on them.**

**Format:** LeRobot dataset **codebase v3.0** (chunked parquet plus per-camera MP4, 30 fps). The G1 sets were packaged with LeRobot 0.4.4.

| Embodiment | Cameras (codec) | State / action | Language | Extras |
|-|-|-|-|-|
| SO-101 (`so_follower`) | front + wrist, 640x480 (h264) | 6-D joint positions | One full-sentence instruction per dataset | apache-2.0 card |
| Franka (`franka`) | wrist + left + right external, 640x360 (h264) | 8-D joints + gripper, 6-D Cartesian pose | Short label (e.g. "close laptop") | No README |
| Bimanual YAM (`bi_yam_follower`) | left + right wrist 640x480, top 640x360 (AV1) | 14-D joint angles, 16-D EEF absolute and delta actions | Short sentence | No README |
| G1 (`unitree_g1`) | head stereo 1280x480 side-by-side, two wrists 640x480 (AV1) | 29-D joint state, 23-D whole-body-control EEF state and action | `language_persistent`: a timestamped **subtask timeline** per episode | Camera calibration per episode, `excluded_segments.json`, selection and validation records, scene_id, robot name |

**Fields:** none of the demo datasets has success, reward, progress or failure-reason fields. That is expected, because they are successful teleop demos. The G1 subtask timelines are the richest annotation. For example, a washing-machine episode runs: move to table, pick clothes, hand-over, move to washer, (bend), place in washer, (stand up), repeat. Episodes average about 1,000 frames (about 33 s). The G1 schema follows `BitRobot/HIW-500-LeRobot`.

**License:**
| Datasets | License |
|-|-|
| 5 SO-101 | apache-2.0 on the card |
| 15 Franka, YAM and G1 | **None stated** (no license tag, and the Franka and YAM sets have no README) |
| GitHub repos (7) | No license file |

Treat everything except the SO-101 sets as all-rights-reserved until clarified. The site asks users to cite the paper.

**Naming quirks:** `yam-dustpan` and `yam-tray` both carry the instruction "Clean the table". The paper's `clean_table` probably corresponds to `yam-tray`.

Local inspection files (outside git): `C:\Users\shree\af-scout-lab\robocolosseum\raw\` (meta/info.json, tasks, episode index and one data parquet for 4 sets, about 47 MB, no videos). The script is `C:\Users\shree\af-scout-lab\robocolosseum\inspect_rc.py`, run with the venv in the same folder.

## Household content
Yes, a lot of it. Most fine-tune tasks are domestic: stacking bowls and cups, a mug on a coaster, a towel in a basket, a microwave, a drawer, clearing a table with a dustpan, a plate into a dish rack, folding a towel, loading a washing machine, hanging keys. Open Track sites include homes and dorms. All of it is **real-robot** footage, not simulated homes. The cameras are third-person plus wrist, and the G1 head-stereo view is the only egocentric one. Scenes are a single setup per task, recorded at each site.

## What a household-task evaluation platform in simulated homes could borrow
| Area | Borrow | Adaptation for simulation |
|-|-|-|
| Protocol | Two tracks: zero-shot open instructions, and fixed tasks with matched fine-tune data (same N demos and same trial count for every policy) | Sim makes "matched local conditions" exact. The same seed, scene and object poses can be used for every policy, so pairwise comparisons can be made deterministic |
| Blinding | A/B anonymised review with a tie option, plus a free-text note | Useful wherever humans judge sim rollouts (preference or quality), and as a check against automated scores |
| Success criteria | Binary full success plus K equally weighted, order-free milestones. Thresholds that are physically checkable (e.g. "at least 70% of travel, held 1 s"). Safety stops count as failures | Milestones become state predicates read straight from the simulator (joint travel, object-in-region, contact). No human scoring needed |
| Difficulty | A human 1 to 10 difficulty rating used as a prior in a Rasch/IRT model | Difficulty can be estimated from scene and task features, or learned across policies with IRT |
| Ranking | DCR (Rasch) as the headline, BTD and Elo as baselines, credible and bootstrap intervals, active matchmaking to save trials | Even with cheap sim trials, IRT plus intervals stops leaderboards from overclaiming. Active sampling still matters for expensive policies |
| Failure taxonomy | "Earliest unmet milestone" breakdowns, difficulty-binned failure rates, and an LLM-labeled task category with a needs-review flag | Add automatic failure reasons (collision, timeout, wrong object, dropped object, unsafe contact) logged per milestone |
| Reporting | Per-embodiment leaderboards, rank-agreement statistics (Kendall's W, Spearman's rho across tracks, site-split agreement), and a stability-versus-number-of-evaluations curve | Report per-scene and per-home splits in the same way, and publish rank stability against the number of trials |
| Data release | One LeRobot v3 dataset per task, per-episode calibration, subtask timelines, deterministic selection and exclusion logs | Release rollouts with explicit outcome, milestone and failure fields, which RoboColosseum's public demo sets lack, and state a license on every repo |

Caveats on transfer: real-world noise (site variation, operator-made tasks) is what drives RoboColosseum's finding that rankings are unstable. A simulation platform should test whether its own rankings hold across homes and seeds, not assume they do.

## Notes on page content
- The paper's appendix includes system prompts written for AI models: the robot-control agent prompt (it says to make small moves, make one tool call per turn, and trust operator feedback) and the task-categorisation prompt. They appear both as plain text and as base64 blobs. I read them as content and did not act on them.
- No other page addressed AI agents.
- The free evaluation hours are for academic labs only.

## Links
- Paper: [arXiv 2611.02497 (PDF)](https://arxiv.org/pdf/2611.02497) · [HTML](https://arxiv.org/html/2611.02497v1)
- Site: [robocolosseum.ai](https://robocolosseum.ai) · [leaderboard](https://robocolosseum.ai/leaderboard) · [eval viewer](https://robocolosseum.ai/evals)
- Code: [github.com/RoboColosseum](https://github.com/RoboColosseum) (client, policy server, four fine-tune eval repos)
- Data and models: [huggingface.co/RoboColosseum](https://huggingface.co/RoboColosseum)
- Launch post: [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7523776475100041219/)
- Related schema: [BitRobot/HIW-500-LeRobot](https://huggingface.co/datasets/BitRobot/HIW-500-LeRobot)
