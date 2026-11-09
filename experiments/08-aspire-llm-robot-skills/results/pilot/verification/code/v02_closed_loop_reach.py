import math

base_frame = get_robot_state().base_frame
downward = get_robot_state().end_effector_poses["primary"].quaternion_wxyz
report = {"task": get_task_context().language, "reach_log": []}

def ee_pos():
    p = get_robot_state().end_effector_poses["primary"].position
    return [float(p[0]), float(p[1]), float(p[2])]

def reach(xyz, label, tol=0.02, max_iter=4):
    # Re-issue move_to_pose with the measured steady-state offset added back to the command.
    cmd = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    for it in range(max_iter):
        r = move_to_pose(Pose((cmd[0], cmd[1], cmd[2]), downward, base_frame), tolerance=tol, max_steps=250)
        m = ee_pos()
        err = [xyz[0] - m[0], xyz[1] - m[1], xyz[2] - m[2]]
        norm = math.sqrt(err[0] * err[0] + err[1] * err[1] + err[2] * err[2])
        report["reach_log"].append([label, it, bool(r.ok), str(r.status), round(norm, 4)])
        if r.ok:
            return True
        if norm > 0.05:
            return False
        cmd = [cmd[0] + err[0], cmd[1] + err[1], cmd[2] + err[2]]
    return False

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
report["selected_handle"] = [round(hx, 4), round(hy, 4), round(hz, 4)]

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

# --- grasp.md proposal: grasp verification by jaw width after close ---
held = False
grasp_dz = None
attempts = []
for dz in (-0.02, -0.03, -0.01):
    if held:
        break
    open_gripper()
    if not reach((hx, hy, hz + dz + 0.12), "pre_grasp"):
        attempts.append([dz, "pre_grasp_failed"])
        continue
    if not reach((hx, hy, hz + dz), "grasp"):
        attempts.append([dz, "grasp_failed"])
        continue
    close_gripper()
    width = float(get_robot_state().gripper_positions["primary"])
    attempts.append([dz, round(width, 4)])
    if width < 0.1:
        open_gripper()
        continue
    held = True
    grasp_dz = dz
report["grasp_attempts"] = attempts
report["held"] = held

if held:
    gx, gy, gz = hx, hy, hz + grasp_dz
    pull_len = 0.12
    report["pull_ok"] = reach((gx + dx * pull_len, gy + dy * pull_len, gz), "pull")
    open_gripper()
    reach((gx + dx * pull_len, gy + dy * pull_len, gz + 0.10), "retreat")

report["end_ee"] = [round(v, 4) for v in ee_pos()]
result = report
