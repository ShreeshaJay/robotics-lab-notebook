import math

ctx = get_task_context()
state = get_robot_state()
frame = state.base_frame
cam = "agentview"

result = {"ok": False, "stage": "start", "instruction": ctx.language, "frame": frame}

seg = segment_text(cam, "drawer handle on the cabinet")
if not seg.ok:
    result = {"ok": False, "stage": "segment", "instruction": ctx.language, "frame": frame}
else:
    cands = []
    for m in seg.masks:
        pc = mask_to_point_cloud(m, cam, target_frame=frame)
        geom = estimate_geometry(pc)
        ext = geom.extents
        size = max(float(ext[0]), float(ext[1]), float(ext[2]))
        if size < 0.25:
            c = geom.center
            cands.append({"x": float(c[0]), "y": float(c[1]), "z": float(c[2])})

    if len(cands) == 0:
        result = {"ok": False, "stage": "no_handle_candidates", "instruction": ctx.language, "frame": frame}
    else:
        cands.sort(key=lambda d: d["z"])
        mid = cands[len(cands) // 2]
        hx, hy, hz = mid["x"], mid["y"], mid["z"]

        # Pull direction: from handle toward robot base (base assumed at origin of base frame).
        ux, uy = -hx, -hy
        norm = math.sqrt(ux * ux + uy * uy)
        if norm < 1e-6:
            ux, uy = -1.0, 0.0
            norm = 1.0
        ux, uy = ux / norm, uy / norm

        # Gripper z-axis points into the cabinet (-u); yaw phi about world z.
        phi = math.atan2(-uy, -ux)
        w1 = math.cos(phi / 2.0)
        s1 = math.sin(phi / 2.0)
        c = math.sqrt(0.5)
        quat = [c * w1, -s1 * c, w1 * c, c * s1]
        qn = math.sqrt(sum(q * q for q in quat))
        quat = [q / qn for q in quat]

        pre_pos = [hx + 0.10 * ux, hy + 0.10 * uy, hz + 0.02]
        grasp_pos = [hx, hy, hz]
        pull_pos = [hx + 0.12 * ux, hy + 0.12 * uy, hz]

        steps = []

        r = open_gripper()
        steps.append({"stage": "open_pre", "ok": bool(r.ok)})

        r = move_to_pose(Pose(pre_pos, quat, frame), max_steps=300)
        steps.append({"stage": "pre_grasp", "ok": bool(r.ok)})
        if not r.ok:
            result = {"ok": False, "stage": "pre_grasp", "steps": steps, "handle": [hx, hy, hz]}
        else:
            r = move_to_pose(Pose(grasp_pos, quat, frame), max_steps=300)
            steps.append({"stage": "grasp", "ok": bool(r.ok)})
            if not r.ok:
                result = {"ok": False, "stage": "grasp", "steps": steps, "handle": [hx, hy, hz]}
            else:
                r = close_gripper()
                steps.append({"stage": "close", "ok": bool(r.ok)})

                r = move_to_pose(Pose(pull_pos, quat, frame), max_steps=300)
                steps.append({"stage": "pull", "ok": bool(r.ok)})

                r = open_gripper()
                steps.append({"stage": "release", "ok": bool(r.ok)})

                result = {
                    "ok": all(s["ok"] for s in steps),
                    "stage": "done",
                    "instruction": ctx.language,
                    "frame": frame,
                    "handle_xyz": [hx, hy, hz],
                    "pull_direction_xy": [ux, uy],
                    "num_handle_candidates": len(cands),
                    "steps": steps,
                }
