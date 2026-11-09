import json
import sys

LIB_SHA = "91d636a706623f4e"

HANDLE_SNIPPET = '''handle_seg = segment_text("agentview", "black drawer handle")
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
hz, hx, hy, handle_mask = middle[0], middle[2], middle[3], middle[4]'''

JAW_SNIPPET = '''close_gripper()
width = float(get_robot_state().gripper_positions["primary"])
if width < 0.1:
    # jaws reached ~0.0: they closed on nothing, so the grasp missed the target.
    # Do not pull. Reopen, shift the approach height, and re-approach.
    open_gripper()
else:
    # held: a contact stall leaves the width at the object's thickness
    grasp_held = True'''

report = {
    "schema_version": 1,
    "benchmark": "libero-pro",
    "suite": "libero_goal_swap",
    "task_id": 0,
    "skill_library_sha": LIB_SHA,
    "consulted": [
        {"skill": "task-exploration.md", "section": "Initial Scene Analysis (scene_snapshot probe)",
         "verdict": "useful",
         "evidence": "scene_snapshot on seed 51 gave the authoritative language and both keyframes; probe copy with PROBE_PROMPTS filled gave prompt scores and overlays for seed 51"},
        {"skill": "task-exploration.md", "section": "Sandbox constraints when writing probes",
         "verdict": "useful",
         "evidence": "no-underscore and no-getattr rules were followed in every program; no sandbox errors across 15-seed sweeps v00-v06"},
        {"skill": "localize.md", "section": "Key Signals",
         "verdict": "useful",
         "evidence": "score threshold and minimum-point guard applied to handle masks; the 0.5*top score filter removed a 0.20-score false positive on seed 51"},
        {"skill": "localize.md", "section": "Prompt Registry",
         "verdict": "useful",
         "evidence": "prompt 'black drawer handle' returned three handle masks at 0.53/0.52/0.51 on seed 51"},
        {"skill": "localize.md", "section": "Disambiguation: Two Similar Objects in Scene",
         "verdict": "not-applicable",
         "evidence": "selection was by height among stacked identical handles, not by bbox area; the smallest_box_candidate helper was not needed"},
        {"skill": "localize.md", "section": "Standard Localization Helper",
         "verdict": "not-applicable",
         "evidence": "needed the per-candidate masks for mask_to_point_cloud and generate_grasps, so localize_object was not used"},
        {"skill": "grasp.md", "section": "Standard Pick-and-Place Template (grasp backend preferred for irregular objects)",
         "verdict": "misleading",
         "evidence": "contact-graspnet on the middle handle mask (seed 56 probe) returned one candidate at (0.173,-0.278,-0.036) while the handle center was at (0.403,-0.170,0.109), about 0.3 m away; v03 used that backend and failed pre-grasp on all 15 seeds 51-65"},
        {"skill": "grasp.md", "section": "Key Conventions: Backend choice (curobo for free-space transit)",
         "verdict": "misleading",
         "evidence": "v00 used trajectory_planner='curobo' for a pre-grasp transit whose goal was 0.10 m from the handle: planning_failed after ~79 s per call on all 15 seeds 51-65 (0/15); interpolation was used afterwards"},
        {"skill": "grasp.md", "section": "Key Conventions: close_gripper and never infer grasp success from a completed call",
         "verdict": "useful",
         "evidence": "close_gripper used; v04 seed 56 closed to 0.0 in ~9 steps with no object, which showed the completed call was a miss; v05 width check used on seeds 51-65 (held at dz=-0.02 on 51, 60, 61)"},
        {"skill": "grasp.md", "section": "Object Geometry Utilities (quantile center)",
         "verdict": "useful",
         "evidence": "median-quantile center of the handle cloud was used for every sweep; grasps at dz=-0.02 held on seeds 51, 60, 61 (v05) and 52, 53, 63 (v06), while dz=0 and +0.02 closed on nothing; the median center is probably about 2 cm high for this handle"},
        {"skill": "transport.md", "section": "Pre-Probe IK Conditioning",
         "verdict": "not-applicable",
         "evidence": "no placement target; the task is an articulated pull"},
        {"skill": "transport.md", "section": "Physics Settling After Release / Basket Container Placement",
         "verdict": "not-applicable",
         "evidence": "no release onto a target; the program ends after opening the gripper"},
        {"skill": "manipulation.md", "section": "Knob / Switch Turning (guard every motion; Z disambiguation)",
         "verdict": "useful",
         "evidence": "the .ok guard on motions and the height-ordered choice among identical stacked handles were applied; middle-by-height selected the same handle on every seed"},
        {"skill": "manipulation.md", "section": "Hinged Door / Pushing / General Notes",
         "verdict": "not-applicable",
         "evidence": "no hinge, push, or knob in this task"},
        {"skill": "manipulation.md", "section": "Header reference to '*Drawer Interaction* principles below'",
         "verdict": "misleading",
         "evidence": "the header promises Drawer Interaction principles, but no such section exists in manipulation.md; I found no drawer-pull guidance and had to derive the approach from probes on seeds 51, 52 and 56"},
    ],
    "proposed_edits": [
        {"skill": "grasp.md", "section": "Standard Pick-and-Place Template",
         "change": "Before using the generate_grasps backend, check the returned grasp: its position must lie within about 3 cm of the localized target center. If it does not, fall back to a handwritten grasp pose on the localized center. Note that contact-graspnet on a small handle mask (about 2 cm) returned an off-target candidate.",
         "why": "seed 56 probe: the single candidate was about 0.3 m from the handle, and the v03 sweep using it failed pre-grasp on 15/15 seeds"},
        {"skill": "grasp.md", "section": "Key Conventions: Backend choice",
         "change": "Limit trajectory_planner='curobo' to transits whose goal is clear of the target object. A pre-grasp about 0.10 m from a drawer handle failed with planning_failed after about 79 s per call; keep interpolation for the final approach and any goal that is close to the object.",
         "why": "v00 sweep: 0/15 with planning_failed after about 79 s per call on seeds 51-65"},
        {"skill": "manipulation.md", "section": "Header",
         "change": "Either add a Drawer Interaction section or remove the reference to it. Until a verified drawer pull exists, the header should say the drawer section is empty.",
         "why": "the header points to a section that does not exist, so a drawer task finds no guidance"},
        {"skill": "localize.md", "section": "Prompt Registry",
         "change": "Add a row: object 'drawer handle, stacked identical handles' | prompt 'black drawer handle' | libero_goal_swap task 0 | three handles at scores 0.53/0.52/0.51 on seed 51; a 0.20-score false positive on a metal stove edge; filter candidates to score >= 0.5 x top before use.",
         "why": "the seed-51 probe gave these scores and boxes"},
    ],
    "proposed_new": [
        {"skill": "localize.md", "title": "Middle of stacked identical handles (height-ordered)",
         "trigger": "Several identical handles or knobs stacked vertically on one face, and the target is the middle layer",
         "code": HANDLE_SNIPPET,
         "evidence": "seeds 51-65 sweeps v00-v06 (handle chosen on all seeds); the selected handle was grasped on seeds 51, 56, 60, 61 in v04-v05"},
        {"skill": "grasp.md", "title": "Grasp verification by jaw width after close",
         "trigger": "After close_gripper() on a small part, to tell a real hold from a miss",
         "code": JAW_SNIPPET,
         "evidence": "v05, seeds 51, 60, 61: closed-empty at dz=0 and +0.02 (width 0.0), held at dz=-0.02 (width 0.23); seed 56 held at dz=0 (width 0.13)"},
    ],
}

with open(sys.argv[1], "w") as fh:
    json.dump(report, fh, indent=2)
print("wrote", sys.argv[1])
