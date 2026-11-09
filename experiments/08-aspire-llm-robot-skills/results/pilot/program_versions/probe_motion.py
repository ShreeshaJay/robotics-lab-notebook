# v00 — initial program from scene analysis.
# Task: "Open the middle layer of the drawer" (libero_goal_swap task 0).
# Scene: dark cabinet on the left with three stacked black drawer handles on its front face.
# Strategy: localize handles with SAM3, pick the middle one by height, approach it, grasp, pull outward
# along the cabinet-center -> handle direction, release.

base_frame = get_robot_state().base_frame
camera = "agentview"


def center_of(point_cloud):
    pts = point_cloud.points
    xs = sorted(float(p[0]) for p in pts)
    ys = sorted(float(p[1]) for p in pts)
    zs = sorted(float(p[2]) for p in pts)
    n = len(xs)
    return (xs[n // 2], ys[n // 2], zs[n // 2])


# --- Handle candidates ---
handle_seg = segment_text(camera, "black drawer handle")
if not handle_seg.ok or len(handle_seg.segmentations) == 0:
    raise ValueError("no drawer handle segmented")
top_handle_score = max(float(c.score) for c in handle_seg.segmentations)
handle_entries = []
for candidate in handle_seg.segmentations[:10]:
    if float(candidate.score) < 0.5 * top_handle_score:
        continue
    handle_cloud = mask_to_point_cloud(candidate, camera, target_frame=base_frame)
    if len(handle_cloud.points) < 10:
        continue
    hc = center_of(handle_cloud)
    handle_entries.append((hc[2], hc[0], hc[1]))
if len(handle_entries) == 0:
    raise ValueError("no handle candidate with enough points")

# Middle layer: order handles by height and take the middle one.
handle_entries = sorted(handle_entries)
mid = handle_entries[len(handle_entries) // 2]
hz = mid[0]
hx = mid[1]
hy = mid[2]

# --- Cabinet body center, to get the outward pull direction ---
cab_seg = segment_text(camera, "black cabinet")
if not cab_seg.ok or len(cab_seg.segmentations) == 0:
    raise ValueError("cabinet not segmented")
best_cab = cab_seg.segmentations[0]
for candidate in cab_seg.segmentations[:5]:
    if float(candidate.score) > float(best_cab.score):
        best_cab = candidate
cab_cloud = mask_to_point_cloud(best_cab, camera, target_frame=base_frame)
if len(cab_cloud.points) < 20:
    raise ValueError("cabinet cloud too small")
cab = center_of(cab_cloud)
dx = hx - cab[0]
dy = hy - cab[1]
norm = (dx * dx + dy * dy) ** 0.5
if norm < 1e-6:
    raise ValueError("degenerate pull direction")
ux = dx / norm
uy = dy / norm

# --- Poses (end-effector orientation kept from the current state) ---
quat = get_robot_state().end_effector_poses["primary"].quaternion_wxyz
standoff = 0.10
pre_grasp = Pose((hx + ux * standoff, hy + uy * standoff, hz), quat, base_frame)
grasp_pose = Pose((hx, hy, hz), quat, base_frame)
pulled = Pose((hx + ux * 0.15, hy + uy * 0.15, hz), quat, base_frame)

report = {"task": get_task_context().language, "handle": [hx, hy, hz], "pull_dir": [ux, uy]}

open_gripper()
r1 = move_to_pose(pre_grasp, tolerance=0.03, max_steps=400)
report["r1"] = [str(r1.status), int(r1.steps_executed), str(r1.final_errors)[:300]]
report["ee1"] = list(get_robot_state().end_effector_poses["primary"].position)
r2 = move_to_joints(get_robot_state().joint_positions["primary"], tolerance=0.01, max_steps=10)
report["r2"] = [str(r2.status), int(r2.steps_executed)]
result = report
