# v05 — horizontal approach to the middle drawer handle with measured-position checks and
# a grasp-verification retry (close, read gripper width, reopen and shift height on a miss).
# Task: "Open the middle layer of the drawer".

base_frame = get_robot_state().base_frame
camera = "agentview"


def center_of(point_cloud):
    pts = point_cloud.points
    xs = sorted(float(p[0]) for p in pts)
    ys = sorted(float(p[1]) for p in pts)
    zs = sorted(float(p[2]) for p in pts)
    n = len(xs)
    return (xs[n // 2], ys[n // 2], zs[n // 2])


def ee_pos():
    return get_robot_state().end_effector_poses["primary"].position


def dist_to(p):
    e = ee_pos()
    return ((float(e[0]) - float(p[0])) ** 2 + (float(e[1]) - float(p[1])) ** 2 + (float(e[2]) - float(p[2])) ** 2) ** 0.5


# --- Middle handle (by height among the three black handles) ---
handle_seg = segment_text(camera, "black drawer handle")
if not handle_seg.ok or len(handle_seg.segmentations) == 0:
    raise ValueError("no drawer handle segmented")
top_handle_score = max(float(c.score) for c in handle_seg.segmentations)
entries = []
for candidate in handle_seg.segmentations[:10]:
    if float(candidate.score) < 0.5 * top_handle_score:
        continue
    handle_cloud = mask_to_point_cloud(candidate, camera, target_frame=base_frame)
    if len(handle_cloud.points) < 10:
        continue
    hc = center_of(handle_cloud)
    entries.append((hc[2], hc[0], hc[1]))
if len(entries) == 0:
    raise ValueError("no handle candidate with enough points")
entries = sorted(entries)
mid = entries[len(entries) // 2]
hz = mid[0]
hx = mid[1]
hy = mid[2]

# --- Outward pull direction: cabinet body center -> handle ---
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

# --- Horizontal gripper orientation: approach axis = -u, fingers separated along Z ---
ax = (uy, -ux, 0.0)
ay = (0.0, 0.0, 1.0)
az = (-ux, -uy, 0.0)
m = [[ax[0], ay[0], az[0]], [ax[1], ay[1], az[1]], [ax[2], ay[2], az[2]]]
trace = m[0][0] + m[1][1] + m[2][2]
if trace > 0:
    s = (trace + 1.0) ** 0.5 * 2.0
    qw = 0.25 * s
    qx = (m[2][1] - m[1][2]) / s
    qy = (m[0][2] - m[2][0]) / s
    qz = (m[1][0] - m[0][1]) / s
elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
    s = (1.0 + m[0][0] - m[1][1] - m[2][2]) ** 0.5 * 2.0
    qw = (m[2][1] - m[1][2]) / s
    qx = 0.25 * s
    qy = (m[0][1] + m[1][0]) / s
    qz = (m[0][2] + m[2][0]) / s
elif m[1][1] > m[2][2]:
    s = (1.0 + m[1][1] - m[0][0] - m[2][2]) ** 0.5 * 2.0
    qw = (m[0][2] - m[2][0]) / s
    qx = (m[0][1] + m[1][0]) / s
    qy = 0.25 * s
    qz = (m[1][2] + m[2][1]) / s
else:
    s = (1.0 + m[2][2] - m[0][0] - m[1][1]) ** 0.5 * 2.0
    qw = (m[1][0] - m[0][1]) / s
    qx = (m[0][2] + m[2][0]) / s
    qy = (m[1][2] + m[2][1]) / s
    qz = 0.25 * s
quat = (qw, qx, qy, qz)

report = {"task": get_task_context().language, "handle": [hx, hy, hz], "pull_dir": [ux, uy], "attempts": []}

standoff = 0.14
pull_len = 0.22
grasp_ok = False
open_gripper()
for dz in (-0.02, 0.0, -0.04):
    zc = hz + dz
    pre_xyz = (hx + ux * standoff, hy + uy * standoff, zc)
    grasp_xyz = (hx, hy, zc)
    move_to_pose(Pose(pre_xyz, quat, base_frame), tolerance=0.02, max_steps=200)
    pre_d = dist_to(pre_xyz)
    if pre_d > 0.025:
        report["attempts"].append([dz, "pre-miss", round(pre_d, 4)])
        continue
    move_to_pose(Pose(grasp_xyz, quat, base_frame), tolerance=0.01, max_steps=200)
    grasp_d = dist_to(grasp_xyz)
    if grasp_d > 0.02:
        report["attempts"].append([dz, "grasp-miss", round(grasp_d, 4)])
        continue
    close_gripper()
    width = get_robot_state().gripper_positions["primary"]
    if float(width) < 0.1:
        report["attempts"].append([dz, "closed-empty", round(float(width), 3)])
        open_gripper()
        continue
    report["attempts"].append([dz, "held", round(float(width), 3)])
    grasp_ok = True
    break

report["grasp_ok"] = grasp_ok
report["width_before_pull"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
if not grasp_ok:
    raise ValueError("no handle grasp verified")

pulled_xyz = (hx + ux * pull_len, hy + uy * pull_len, hz + dz)
move_to_pose(Pose(pulled_xyz, quat, base_frame), tolerance=0.02, max_steps=300)
report["pull_dist"] = round(dist_to(pulled_xyz), 4)
open_gripper()

result = report
