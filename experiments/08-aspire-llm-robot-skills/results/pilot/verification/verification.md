# Verification: libero-pro / libero_goal_swap task 0

Verdict: refuted (attempt budget exhausted). Seeds passed: 0/5 on attempt 1 and 0/5 on attempt 2.

## Reading rules

Read: the shared skill library (README, task-exploration, grasp, localize, transport, manipulation), `proposed_skills.md`, `docs/api-reference.md`, `docs/run-artifacts.md`, `docs/libero-pro.md`, and `.claude/skills/iterative-debugging/scripts/scene_snapshot.py` (copied to `code/probe_scene.py`). Also read my own run artifacts and one keyframe image. `docs/libero-pro.md` has a table naming an example program for this task. I did not open that program or anything under `examples/`. The table's one-line behavior description ("localize and pull the middle drawer") was visible to me, but the snapshot's task language ("Open the middle layer of the drawer") already said the same. I did not list or read any campaign task directory, sibling workspace, `progress.md`, or `curriculum.md`. I did not read any real-robot material.

## Prediction (step 1)

I expected to need: the localize proposal (middle of stacked handles) and its prompt registry row, the grasp width check, and the grasp template for the pick. I expected the drawer pull to need guidance the library lacks, because `manipulation.md` has no drawer section. The first part held: the localization reproduced. The second part held in the worst way: the grasp never completed, and the library had nothing for the pull.

## Scene

Task language: "Open the middle layer of the drawer." Cabinet on the left of the agentview frame, three stacked black handles on one face. Detected handle centres in robot_base: about (0.417, -0.157) at z about 0.041, 0.110 and 0.183. The selected middle handle sits at z 0.1097 on every seed. Pull direction (principal axis of the handle footprint, sign set toward +x) came out near (1.0, 0.05) on all seeds.

## Attempt 1 (v01, literal proposals)

`code/v01_from_skills.py` applies the localize proposal verbatim, then a grasp loop at dz in (-0.02, -0.03, -0.01) with pre-grasp at +0.12 and `move_to_pose` at tolerance 0.02. Every seed failed at the pre-grasp `move_to_pose`: status `timeout`, steps_executed 120 to 126, while `plan_motion` and `pyroki.solve_ik` returned ok. Recorded joint states show the arm stopping about 2 cm short in y and about 0.6 to 0.9 cm high in z of the commanded pre-grasp (commanded z 0.210, measured 0.216 to 0.219). Run: `debug/v01/`.

## Attempt 2 (v02, closed-loop re-issue)

`code/v02_closed_loop_reach.py` re-issues each target with the measured error added to the command, up to 4 times, aborting if the error exceeds 5 cm. The position error falls to about 0.0064 m by the third re-issue, yet status stays `timeout`, and the fourth re-issue returns `controller_failed`. No seed reached a close. The end-effector quaternion ends about 0.02 to 0.03 off the downward quaternion, which may be the real convergence blocker. That is a hypothesis and was not tested. Run: `debug/v02/`.

## Proposals

- localize.md middle-of-stack: sound. Detection and selection reproduced on all five seeds. Fragile to extra candidates at or above 0.5 x top, as seed 53 showed with a low false positive that happened to sit below the middle.
- localize.md prompt registry row: sound. Scores 0.53/0.52/0.51 reproduced on seeds 51 and 55, and the 0.20 stove-edge false positive was filtered on every seed.
- grasp.md width check after close: untested. No close ever ran on a handle.
- grasp.md generate_grasps 3 cm check: untested. The backend was never called.
- grasp.md curobo backend edit: untested. No curobo call was made.
- manipulation.md header pointer: sound. The Drawer Interaction section it points to does not exist.

## Gaps

The skills say nothing about: steady-state offset of the joint servo at the handle; what a `move_to_pose` timeout with small measured error means; whether move_to_pose's convergence test includes orientation; a drawer pull pattern (direction, grasp-then-pull, pull length); and the gripper readback drifting from open toward about 0.6 during transit. Full text is in `verification_report.json`.
