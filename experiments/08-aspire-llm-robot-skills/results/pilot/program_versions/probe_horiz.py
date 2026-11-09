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
# Horizontal approach along -u (gripper z = approach dir), fingers close along world Z (gripper y = +Z).
# Columns of R are the gripper axes: x = y cross z, y = +Z, z = -u.
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
quat_h = quat
report = {}
quat_down = get_robot_state().end_effector_poses["primary"].quaternion_wxyz
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2)
rows = []
for name, q in [("H_v02", quat_h), ("H_rot90", qmul(quat_h, (0.70710678, 0.0, 0.0, 0.70710678)))]:
    pre = Pose((hx + ux * 0.14, hy + uy * 0.14, hz), q, base_frame)
    r1 = move_to_pose(pre, tolerance=0.03, max_steps=200)
    ee1 = get_robot_state().end_effector_poses["primary"].position
    rows.append([name, "pre", str(r1.status), int(r1.steps_executed), round(float(ee1[0]),3), round(float(ee1[1]),3), round(float(ee1[2]),3)])
    if r1.ok:
        gp_ = Pose((hx + ux * 0.02, hy + uy * 0.02, hz), q, base_frame)
        r2 = move_to_pose(gp_, tolerance=0.02, max_steps=200)
        ee2 = get_robot_state().end_effector_poses["primary"].position
        rows.append([name, "grasp", str(r2.status), int(r2.steps_executed), round(float(ee2[0]),3), round(float(ee2[1]),3), round(float(ee2[2]),3)])
    gh = go_home()
    rows.append([name, "home", str(gh.status)])
report["rows"] = rows
result = report
