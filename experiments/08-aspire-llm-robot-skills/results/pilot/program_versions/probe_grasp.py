# v03 — grasp backend on the middle drawer handle, then pull outward along the cabinet normal.
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


def approach_axis(q):
    w = float(q[0])
    x = float(q[1])
    y = float(q[2])
    z = float(q[3])
    return (2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y))


# --- Middle handle: keep the candidate mask so the grasp backend can use it ---
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
    entries.append((hc[2], len(entries), hc[0], hc[1], candidate))
if len(entries) == 0:
    raise ValueError("no handle candidate with enough points")
entries = sorted(entries)
mid = entries[len(entries) // 2]
hz = mid[0]
hx = mid[2]
hy = mid[3]
handle_mask = mid[4]

# --- Pull direction: from cabinet body center toward the handle (outward) ---
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

rows = []
report = {}
gs = generate_grasps(camera, handle_mask, backend="contact-graspnet", max_candidates=5)
rows.append(["ok", str(gs.ok)])
if gs.ok:
    for g in gs.grasps[:5]:
        p = g.pose.position
        q = g.pose.quaternion_wxyz
        a = approach_axis(q)
        rows.append([round(float(p[0]),3), round(float(p[1]),3), round(float(p[2]),3), [round(float(v),2) for v in a], round(float(g.score),3)])
rows.append(["handle", round(hx,3), round(hy,3), round(hz,3), "u", round(ux,3), round(uy,3)])
for strat in ["top_down", "highest_score"]:
    sel = select_grasp(gs, strategy=strat) if gs.ok else None
    rows.append([strat, None if sel is None else [round(float(v),3) for v in sel.pose.position]])
report["rows"] = rows
result = report
