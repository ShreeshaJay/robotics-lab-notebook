# Skills Under Test

Proposals from `libero_goal_swap/task_0`. **They are staged here and are deliberately NOT in the shared library** — whether they get promoted is what your verdict decides. Do not go looking for them under `skills/`; this file is their only home. Re-solve the task from the library PLUS this file, without seeing the author's program.

No held-out score exists for the task's own program at this stage. Do not draw a conclusion from a missing score. On a task whose residual failure is kinematic, a low score of your own is expected and is **not** by itself a refutation. Judge each proposal on whether it transmits and whether its mechanism reproduces.

---

## 1. → `localize.md` — Middle of stacked identical handles (height-ordered)

**Trigger.** Several identical handles or knobs stacked vertically on one face, and the target is the middle layer

```python
handle_seg = segment_text("agentview", "black drawer handle")
if not handle_seg.ok or len(handle_seg.segmentations) == 0:
    raise ValueError("no handle segmented")
top = max(float(c.score) for c in handle_seg.segmentations)
entries = []
for candidate in handle_seg.segmentations[:10]:
    if float(candidate.score) < 0.5 * top:
        continue
    cloud = mask_to_point_cloud(candidate, "agentview", target_frame=get_robot_state().base_frame)
    if len(cloud.points) < 10:
        continue
    zs = sorted(float(p[2]) for p in cloud.points)
    xs = sorted(float(p[0]) for p in cloud.points)
    ys = sorted(float(p[1]) for p in cloud.points)
    k = len(zs) // 2
    entries.append((zs[k], len(entries), xs[k], ys[k], candidate))
entries = sorted(entries)
middle = entries[len(entries) // 2]
hz, hx, hy, handle_mask = middle[0], middle[2], middle[3], middle[4]
```

**Evidence.** Development sweeps v00-v06 (handle selected on all 15 development runs); the selected handle was grasped in 4 of 15 development runs in v04-v05

---

## 2. → `grasp.md` — Grasp verification by jaw width after close

**Trigger.** After close_gripper() on a small part, to tell a real hold from a miss

```python
close_gripper()
width = float(get_robot_state().gripper_positions["primary"])
if width < 0.1:
    # jaws reached ~0.0: they closed on nothing, so the grasp missed the target.
    # Do not pull. Reopen, shift the approach height, and re-approach.
    open_gripper()
else:
    # held: a contact stall leaves the width at the object's thickness
    grasp_held = True
```

**Evidence.** v05: closed-empty at dz=0 and +0.02 (width 0.0) on several development runs; held at dz=-0.02 (width 0.23) on 3 of 15; held at dz=0 (width 0.13) on 1 run

---

## Edits

### `grasp.md` § Standard Pick-and-Place Template

**Change.** Before using the generate_grasps backend, check the returned grasp: its position must lie within about 3 cm of the localized target center. If it does not, fall back to a handwritten grasp pose on the localized center. Note that contact-graspnet on a small handle mask (about 2 cm) returned an off-target candidate.

*Why:* a development-run probe: the single candidate was about 0.3 m from the handle, and the v03 sweep using it failed pre-grasp on 15/15 seeds

### `grasp.md` § Key Conventions: Backend choice

**Change.** Limit trajectory_planner='curobo' to transits whose goal is clear of the target object. A pre-grasp about 0.10 m from a drawer handle failed with planning_failed after about 79 s per call; keep interpolation for the final approach and any goal that is close to the object.

*Why:* v00 sweep: 0/15 with planning_failed after about 79 s per call on seeds 51-65

### `manipulation.md`, file header pointer

**Change.** Either add a Drawer Interaction section or remove the reference to it. Until a verified drawer pull exists, the header should say the drawer section is empty.

*Why:* the header points to a section that does not exist, so a drawer task finds no guidance

### `localize.md` § Prompt Registry

**Change.** Add a row: object 'drawer handle, stacked identical handles' | prompt 'black drawer handle' | libero_goal_swap task 0 | three handles at scores 0.53/0.52/0.51 on one development run; a 0.20-score false positive on a metal stove edge; filter candidates to score >= 0.5 x top before use.

*Why:* a development-run probe gave these scores and boxes


---

## Standing disputes — report on any your scene touches

- **Identity selectors.** `localize.md` orders them: the relation the instruction states, else the
  sign of the support gap, else the candidate's own height, else proximity. Tasks measured these
  head to head and got different answers on different scenes. **Compute all of them in-program and
  report the column.**
- **The near-base descent.** Two regimes with opposite remedies — settle-time (more ticks fix it)
  and a servo wall (they do not). The `>= ~50 ticks` duration bar is RETIRED; so is the
  outward-drift test, which one task measured backwards. What survives is **where it froze**
  (achieved z against commanded z) and **which axis is short**. Report both.
- **The jaw as a slip test.** Five numeric bars have been tried and all five failed; a jaw that
  *rises* is a distinct signal (something is prying it). If you deliberately spoil a grasp, report
  the jaw at close / after lift / before place for both it and a good one.
- **Release height.** Measure your band and grade on the landed pose, not the pass count.
- **The first `move_to_pose`.** `tries=3` and stage only if all three fail. Report what yours does.

Verifications have refuted claims outright while their symptoms held, and found helper functions
that entries CALL and that were defined nowhere. Treat "it passed" as the beginning of the question.
