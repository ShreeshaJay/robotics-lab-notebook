import math

def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    return ordered[int((len(ordered) - 1) * fraction)]

base_frame = get_robot_state().base_frame
downward = get_robot_state().end_effector_poses["primary"].quaternion_wxyz
report = {"task": get_task_context().language}

# --- localize.md proposal: middle of stacked identical handles (height-ordered) ---
handle_seg = segment_text("agentview", "black drawer handle")
if not handle_seg.ok or len(handle_seg.segmentations) == 0:
    raise ValueError("no handle segmented")
top = max(float(c.score) for c in handle_seg.segmentations)
entries = []
for candidate in handle_seg.segmentations[:10]:
    if float(candidate.score) < 0.5 * top:
        continue
    cloud = mask_to_point_cloud(candidate, "agentview", target_frame=base_frame)
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
report["handle_scores"] = [round(float(c.score), 3) for c in handle_seg.segmentations[:10]]
report["handle_candidates_kept"] = [[round(e[0], 4), round(e[2], 4), round(e[3], 4)] for e in entries]
report["selected_handle"] = [round(hx, 4), round(hy, 4), round(hz, 4)]

# Pull direction: principal axis of the selected handle's XY footprint (handle protrudes
# along the pull direction); sign chosen toward +x of robot_base (proposal silent: assumption).
handle_cloud = mask_to_point_cloud(handle_mask, "agentview", target_frame=base_frame)
pts = [(float(p[0]), float(p[1])) for p in handle_cloud.points]
mx = sum(p[0] for p in pts) / len(pts)
my = sum(p[1] for p in pts) / len(pts)
sxx = sum((p[0] - mx) ** 2 for p in pts)
syy = sum((p[1] - my) ** 2 for p in pts)
sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
dx = math.cos(theta)
dy = math.sin(theta)
if dx < 0:
    dx = -dx
    dy = -dy
report["pull_dir"] = [round(dx, 4), round(dy, 4)]
report["handle_points"] = len(pts)

# --- grasp.md proposal: grasp verification by jaw width after close ---
# Grasp heights: dz candidates from the proposal (-0.02 held; 0 and +0.02 closed-empty).
held = False
grasp_dz = None
attempts = []
for dz in (-0.02, -0.03, -0.01):
    if held:
        break
    open_gripper()
    pre_grasp = Pose((hx, hy, hz + dz + 0.12), downward, base_frame)
    if not move_to_pose(pre_grasp, tolerance=0.02, max_steps=250).ok:
        attempts.append([dz, "pre_grasp_failed"])
        continue
    grasp_pose = Pose((hx, hy, hz + dz), downward, base_frame)
    if not move_to_pose(grasp_pose, tolerance=0.015, max_steps=250).ok:
        attempts.append([dz, "grasp_approach_failed"])
        continue
    close_gripper()
    width = float(get_robot_state().gripper_positions["primary"])
    attempts.append([dz, round(width, 4)])
    if width < 0.1:
        # closed on nothing: reopen and shift the approach height
        open_gripper()
        continue
    held = True
    grasp_dz = dz
report["grasp_attempts"] = attempts
report["held"] = held

if held:
    gx, gy, gz = hx, hy, hz + grasp_dz
    pull_len = 0.12
    pull_target = Pose((gx + dx * pull_len, gy + dy * pull_len, gz), downward, base_frame)
    pull = move_to_pose(pull_target, tolerance=0.02, max_steps=300)
    report["pull_ok"] = bool(pull.ok)
    report["pull_status"] = str(pull.status)
    report["pull_steps"] = int(pull.steps_executed)
    open_gripper()
    retreat = Pose((gx + dx * pull_len, gy + dy * pull_len, gz + 0.10), downward, base_frame)
    move_to_pose(retreat, tolerance=0.02, max_steps=250)

state_end = get_robot_state()
report["end_ee"] = [round(float(v), 4) for v in state_end.end_effector_poses["primary"].position]
result = report
