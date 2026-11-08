# 01: Inspecting the FineART step-labeled robot dataset (LeRobot format)

Date: 2026-11-07. Status: done. Cost: $0 (CPU laptop, about 50 MB downloaded).

## Goal
Learn what a well-labeled, step-annotated robot manipulation dataset looks like on disk: its schema, how the step (subtask) labels are stored and timed, and what the label text looks like. I did not train anything.

FineART was released by Scale AI and Hugging Face LeRobot. It has 40k+ teleoperated bimanual demos with dense, human-written subtask labels, and it was used to mid-train FineART-VLA (a π0.5-based policy in LeRobot).

## Sources
- LeRobot docs: [FineART-VLA](https://huggingface.co/docs/lerobot/main/en/fineart_vla) and [Language columns and recipes](https://huggingface.co/docs/lerobot/main/en/language_and_recipes)
- Dataset: [`lerobot/FineART`](https://huggingface.co/datasets/lerobot/FineART), inspected at commit `15746b76ddc21ffd38d73888fefe2e7a1496385b`
- Model card: [`lerobot/FineART-VLA-subtask-midtrained`](https://huggingface.co/lerobot/FineART-VLA-subtask-midtrained)
- Paper: [arXiv 2609.36416](https://arxiv.org/abs/2609.36416)
- Code: [huggingface/lerobot](https://github.com/huggingface/lerobot)

## Steps
1. Read the docs, the dataset card and the model card. The FineART-VLA docs page does not give the dataset id. I found `lerobot/FineART` with a Hub search. It is **not gated**, so I did not need a login or a terms click-through.
2. Created a venv outside any git repo and installed only `huggingface_hub`, `pandas` and `pyarrow`, pinned in `requirements.txt`. I did not install LeRobot itself.
3. Ran `inspect_fineart.py`. It does three things:
   - downloads only `meta/info.json`, `meta/stats.json`, `meta/tasks.parquet` and `meta/episodes/chunk-000/file-000.parquet` (about 33 MB, most of it the episode table);
   - reads episodes 0–2 from `data/chunk-000/file-000.parquet` with HTTP range reads through `HfFileSystem`, fetching only the row groups it needs;
   - reads one row group per sampled episode (6 episodes from different task families) to see label lists outside the first file.

   It downloads no videos. The full dataset is 211 data parquet files (about 25 GB) plus about 100k MP4s.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
# -I keeps the script's folder off sys.path; required because this file is named inspect_fineart.py,
# which would otherwise shadow the stdlib `inspect` module that pandas imports.
.venv/Scripts/python inspect_fineart.py --cache-dir C:/path/outside/repo/hf-cache --episodes 3
```

Pinned versions: Python 3.11.9, huggingface_hub 2.4.0, pandas 3.1.1, pyarrow 26.0.1. The full freeze is in `requirements.txt`.

## Results

### Dataset at a glance
| Item | Value |
|-|-|
| Repo id | `lerobot/FineART` (public, not gated) |
| License | CC BY 4.0 (dataset). LeRobot code is Apache 2.0. The FineART-VLA checkpoint is under the Gemma license (the model card says so, but the docs page says Apache 2.0) |
| LeRobot format | `codebase_version: v3.0` (many episodes packed per parquet/MP4 file, located through `meta/episodes` offsets) |
| Size | 40,543 episodes, 185,534,500 frames, about 1,718 h; one `train` split `0:40543` |
| Tasks | 151 task strings in `meta/tasks.parquet` |
| Subtask labels | 533,913 in total (`subtask_count` per episode: mean 13.2, median 11, max 158; 156 episodes have 0) |
| Episode length | mean 152.5 s, median 123.7 s, min 7.3 s, max 2,935 s |
| Robot | `robot_type: aloha` (Trossen Stationary AI, ALOHA-style bimanual) |
| fps | 30 (data and all videos) |
| State / action | `observation.state` and `action`, both float32 [14]: `left_{shoulder_pan, shoulder_lift, elbow_flex, wrist_angle, wrist_rotate, gripper, gripper_finger}` then the same seven for `right_`. Actions are absolute joint targets |
| Cameras | `observation.images.cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist`; h264 video, 360×640×3 |
| Other columns | `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` |

### How the step labels are stored
- **Two optional per-frame columns** in the data parquet:
  - `language_persistent`: a list of structs with fields `role`, `content`, `style`, `timestamp`, `camera`, `tool_calls`;
  - `language_events`: the same struct without `timestamp`.

  In `info.json` their dtype is `language`.
- **Segment labels, stored per frame.** Each episode's full subtask list is repeated on every frame of that episode. I checked this on sampled frames of episodes 0–2. Each row has a start `timestamp` (episode-relative seconds, on the same clock as the frame `timestamp`). A subtask stays active until the next row starts. Segments are contiguous and non-overlapping. The last one runs to the end of the episode.
- **Every row looks the same:** `role="assistant"`, `style="subtask"`, `camera=null`, `tool_calls=null`. `language_events` was empty on every frame I read. The schema has room for `memory`, `plan`, `vqa`, `interjection` and speech, but FineART uses only `subtask`.
- **The first ~1–4 s of each episode has no active subtask** (the first label starts later).
- **The episode-level task** comes from `task_index` → `meta/tasks.parquet`. In the actual file the task string is the pandas *index* (column `task_index`, index name `task`), not two columns as the dataset card's snippet assumes, so `tasks["task"]` raises `KeyError` unless you call `reset_index()` first.
- **Training use:** LeRobot "recipes" (YAML) bind `${task}` and `${subtask}` (`active_at(t, style=subtask)`) into chat-style turns. FineART-VLA's default recipe is 30% task → subtask text prediction and 70% subtask-conditioned action training.

### Example labels
Episode 0, task "Pick & Place candle stick holders":
```text
   1.90s  Put the transparent glass candle holder on the shelf
  12.50s  Put the top right pink glass candle holder on the shelf aligned with the other glass candle holder
  43.30s  Put the right black candle holder on the shelf
  77.80s  Return to Home Position
```
Other task families I sampled:
- "Arrange a breakfast setting": `Pour the cereal from the pink cereal box into the blue bowl`
- "Insert USB": `Insert the blue charger cable into the top black outlet extension`
- "Fold Laundry from Bin": `Stack the folded gray pants on top of the folded black shirt`
- "Pick & Place Garden tools in Gardening bag": `Adjust the white gardening bag`

### Label taxonomy
I looked at 83 labels across 9 episodes (27 from the first 3 episodes plus 56 from 6 sampled episodes):
- **Template.** Labels are imperative, one action per label: verb + attribute-rich object + destination or relation. Median 8 words, max 18.
- **Verbs (first word, 56 sampled labels).** `put` (23), `insert` (11), `fold` (8), `return` (6), `stack` (4), `move` (2), `pour` (1), `adjust` (1).
- **Objects.** Object references carry color, material and position ("top right pink glass candle holder").
- **Ending.** Every episode ends with `Return to Home Position`.
- **Segment length.** In the first file the median segment was 9.0 s. The paper reports 8.6 s across the whole dataset.
- **Repeated labels.** The same label often appears several times in a row, for example `Fold the black shirt` ×3 over about 430 s, and `Put the yellow glass candle holder on the shelf` ×3. This looks like re-attempts, or one object split across segments. No field marks a retry, a failure or a success.

### Inconsistencies I noticed
- **Missing metadata.** The dataset card mentions a whole-episode text summary and execution-quality/success metadata. I could not find either in `info.json` features or in the `meta/episodes` columns, which are `tasks`, `length`, `subtask_count`, file/offset pointers and `stats/*`.
- **Messy task strings.** Some task names are near-duplicates or typos ("Coil cords" / "Coiling cords", "Fold laundry" / "Fold Laundry from Bin", "vace", "relplace"). The tasks are heavily imbalanced: the top task has 1,323 episodes and three tasks have only 1 each.
- **Data files.** Some data files have no parquet row-group statistics, so `inspect_fineart.py` locates rows from `meta/episodes` offsets instead.
- **Docs.** The docs refer to `lerobot/fineart_vla_base`. That name now redirects to `lerobot/FineART-VLA-subtask-midtrained`. The dataset card still has TODO placeholders for the model link.
- **No embedded instructions.** I saw nothing in the pages I read that tried to instruct the reader or an agent.

### Eval caveat: what "16% → 76%" means
The 16% → 76% headline is for an unseen long-horizon task ("prepare breakfast"), 25 rollouts per policy. In **both** conditions, *a human oracle supplied every subtask*. The human typed the next subtask only after judging the previous one complete. The comparison is between a policy mid-trained with subtask labels (76%) and one without (16%), both under this human guidance. It shows the policy became better at *following* step instructions. It does not show the policy can plan the steps itself.

When the policies ran **autonomously** (self-predicted subtasks, or the task label only), **both scored 0% success**, with about 55% stage progress. The paper says "one misstep is enough to stall the whole rollout". The separate spatial-instruction result (32% → 100%, "put the donut in the left/right/middle bin") is single-step. Source: paper §4.2.3 and the dataset card ("with human subtask guidance").

## What this teaches about labeling robot data
1. **Store step labels as timestamped segments on the same clock as the frames.** A start-time-only list (end = next start) automatically keeps segments contiguous and non-overlapping, and resolving "the active step at frame t" is a one-liner.
2. **Make labels specific enough to act on.** Each label should be one imperative action per segment of about 5–15 s, with color, position and relation qualifiers. This specificity is what drove the spatial-instruction gain.
3. **Use a closed style vocabulary plus free text.** Here `style` is the controlled field and `content` is free text. The same schema could later hold memory, VQA (with `camera`) or interjections without a format change.
4. **Keep data columns separate from QA metadata.** The success/quality fields the card mentions are not in the public files. Retries currently show up only as repeated labels. Explicit `attempt`/`outcome` fields would make these datasets easier to filter.
5. **Normalize the task vocabulary.** Near-duplicate task names and long-tail tasks with a single episode hurt sampling by task, which the card itself recommends.
6. **Check headline evals for who chose the steps.** Dense step labels clearly help instruction-following. Autonomous long-horizon planning remained unsolved in this release.

## Next steps
- Decode one short episode's `cam_high` clip (a single MP4 range plus `from_timestamp`/`to_timestamp` in `meta/episodes`) and check how the label boundaries line up with the video.
- Compute dataset-wide label statistics with the Hub's streaming or Dataset Viewer API: verb distribution, repeat rate and segment-length distribution per task family.
- Try LeRobot's `lerobot-annotate` validator and the `active_at` resolver on this data, and compare against my plain-Python resolution.
- Read the rest of the paper's annotation-QA section (auditor sampling and inter-annotator checks).
