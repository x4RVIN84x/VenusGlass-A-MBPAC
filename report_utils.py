import os
import json
from collections import Counter, defaultdict
from datetime import datetime

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def _save_json(path, data):
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def _cause_list(result):
    causes = []
    if not result.get("pass_x", True): causes.append("x")
    if not result.get("pass_y", True): causes.append("y")
    if not result.get("pass_angle", True): causes.append("angle")
    if result.get("not_found", False): causes.append("not_found")
    return causes

def update_reports(*, now: datetime, output_root: str, run_json: dict):
    """
    Append a single glass result into daily and monthly summaries.
    `run_json` must include:
      - "result": the dict from compare_to_golden()
      - "meta": { "file_name", "glass_id", "batch_id", "timestamp" }
    """
    result = run_json.get("result", {})
    meta   = run_json.get("meta", {})

    day_str = now.strftime("%Y-%m-%d")
    mon_str = now.strftime("%Y-%m")

    # daily
    day_dir = os.path.join(output_root, day_str)
    day_path = os.path.join(day_dir, "day_summary.json")
    day = _load_json(day_path, {
        "date": day_str,
        "total": 0, "pass": 0, "fail": 0,
        "fail_causes": {"x":0,"y":0,"angle":0,"not_found":0},
        "samples": []  # optional: keep a short tail of samples
    })

    day["total"] += 1
    if result.get("overall", False):
        day["pass"] += 1
    else:
        day["fail"] += 1
        for c in _cause_list(result):
            day["fail_causes"][c] = day["fail_causes"].get(c, 0) + 1

    # keep last 50 samples in the summary (lightweight preview)
    day["samples"].append({
        "glass_id": meta.get("glass_id"),
        "batch_id": meta.get("batch_id"),
        "file_name": meta.get("file_name"),
        "overall": result.get("overall"),
        "dx": result.get("dx"),
        "dy": result.get("dy"),
        "dtheta": result.get("dtheta"),
        "causes": _cause_list(result),
        "ts": meta.get("timestamp")
    })
    if len(day["samples"]) > 50:
        day["samples"] = day["samples"][-50:]

    _save_json(day_path, day)

    # monthly
    mon_dir = os.path.join(output_root, "monthly")
    mon_path = os.path.join(mon_dir, f"{mon_str}.json")
    mon = _load_json(mon_path, {
        "month": mon_str,
        "days": {},
        "totals": {"total":0,"pass":0,"fail":0,"fail_causes":{"x":0,"y":0,"angle":0,"not_found":0}}
    })

    md = mon["days"].get(day_str, {"total":0,"pass":0,"fail":0,"fail_causes":{"x":0,"y":0,"angle":0,"not_found":0}})
    md["total"] += 1
    if result.get("overall", False):
        md["pass"] += 1
    else:
        md["fail"] += 1
        for c in _cause_list(result):
            md["fail_causes"][c] = md["fail_causes"].get(c, 0) + 1
    mon["days"][day_str] = md

    # roll up totals
    mon["totals"]["total"] += 1
    if result.get("overall", False):
        mon["totals"]["pass"] += 1
    else:
        mon["totals"]["fail"] += 1
        for c in _cause_list(result):
            mon["totals"]["fail_causes"][c] = mon["totals"]["fail_causes"].get(c, 0) + 1

    _save_json(mon_path, mon)
