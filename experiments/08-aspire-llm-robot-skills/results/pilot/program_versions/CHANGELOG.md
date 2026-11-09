modes: planning(curobo transit to pre-grasp, planning_failed after 79s, 15/15 program_error) → fix: default interpolation for transit
v00 — initial program from scene analysis → 0/15 (all: pre-grasp curobo planning_failed)
v01 — ik/planning: drop curobo for pre-grasp transit (plain move_to_pose) → pending
v02 — ik/reach: horizontal approach along -u (fingers close on the bar along Z) instead of current top-down orientation; v01 timed out at z=0.154 vs target 0.110 on seed 51 (reached on 57, 65) → pending
v03 — grasp: contact-graspnet grasp on the middle handle mask (top_down) instead of hand-built poses; v01 pre-grasp at 0.10 standoff stalled at z 0.154 (hand/handle contact), v02 horizontal stalled; 0.18 standoff at handle height reached (probe) → pending
v04 — ik/planning: horizontal approach (v02 orientation) at 0.14 standoff; timeouts were under-converged orientation with position reached (probe seed 51: pre within 5 mm), so success judged by measured EE distance, not status → pending
v05 — grasp: retry on missed/empty close (gripper width check after close_gripper), grasp at handle center with 0.01 tolerance, z-shift retries 0/+0.02/-0.02; v04 closed to 0.0 on empty (missed handle by ~2 cm) → pending
v06 — grasp: contact height offset order (-0.02, 0.0, -0.04): v05 misses at 0.0/+0.02 and holds at -0.02 (seeds 51, 60, 61) — handle center estimate biased high; pull 0.22 m (v05 stalled at ~0.10 m on seed 51 with handle still held) → pending
SELECTED: v05 — 0/15 development seeds (tie-break: v05 has 4 program_ok:true vs v06 3, v04 2)
