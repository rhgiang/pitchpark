"""Read-only queries over the M4 viewer JSON (web/public/data/<slug>.json), for the MCP server.

Nothing here simulates or fetches: it aggregates what export.export_pitcher already wrote
(held-out games only, Statcast-fit "real" paths, M1+M3 "sim" paths), so every number is the
same one the viewer shows. Each function takes the loaded JSON dict, so it is testable without MCP.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "web" / "public" / "data"


def list_pitchers(data_dir: Path = DATA_DIR) -> list[dict]:
    """[{slug, name, throws, n_at_bats}] for every exported pitcher."""
    return json.loads((data_dir / "index.json").read_text(encoding="utf-8"))


def load(slug: str, data_dir: Path = DATA_DIR) -> dict:
    """One pitcher's viewer JSON. The slug is a filename, so reject anything but [a-z0-9-]."""
    if not re.fullmatch(r"[a-z0-9-]+", slug):
        raise ValueError(f"bad slug {slug!r}; use a slug from list_pitchers")
    return json.loads((data_dir / f"{slug}.json").read_text(encoding="utf-8"))


def _med(x) -> float | None:
    x = [v for v in x if v is not None]
    return round(float(np.median(x)), 2) if x else None


def _pitches(d: dict):
    """(previous pitch in the same at-bat or None, pitch) for every pitch."""
    for ab in d["at_bats"]:
        prev = None
        for p in ab["pitches"]:
            yield prev, p
            prev = p


def arsenal(d: dict) -> dict:
    """Per pitch type: count and median speed, spin, approach angles, perceived speed, and break
    vs a no-spin ball (in; + is toward catcher's right / up). Pitches with no tracking count in n
    but have no numbers."""
    by: dict[str, list[dict]] = {}
    for _, p in _pitches(d):
        if p["pitch_type"]:
            by.setdefault(p["pitch_type"], []).append(p)
    return {t: {"n": len(ps),
                "median_mph": _med(p["speed_mph"] for p in ps),
                "median_rpm": _med(p["spin_rpm"] for p in ps),
                "median_vaa_deg": _med(p["tunnel"]["vaa_deg"] for p in ps),
                "median_haa_deg": _med(p["tunnel"]["haa_deg"] for p in ps),
                "median_perceived_mph": _med(p["tunnel"]["perceived_mph"] for p in ps),
                "median_break_x_in": _med(p["tunnel"]["break_x_in"] for p in ps),
                "median_break_z_in": _med(p["tunnel"]["break_z_in"] for p in ps)}
            for t, ps in by.items()}


def tunnel(d: dict, a: str, b: str) -> dict:
    """Separation (in) at the decision point (~25 ft from plate) between a pitch of type `a` and
    the very next pitch of type `b` in the same at-bat: what the hitter actually sees.
    Smaller = harder to tell apart that early."""
    seps = [p["tunnel"]["sep_prev_in"] for prev, p in _pitches(d)
            if prev and prev["pitch_type"] == a and p["pitch_type"] == b
            and p["tunnel"]["sep_prev_in"] is not None]
    return {"from": a, "to": b, "n": len(seps), "median_sep_in": _med(seps),
            "min_sep_in": round(min(seps), 2) if seps else None,
            "max_sep_in": round(max(seps), 2) if seps else None,
            "decision_y_ft": d["meta"].get("decision_y_ft", 25.0)}


def plate_error(d: dict) -> dict:
    """Per pitch type: median distance (in) between the M1+M3 simulated and the Statcast-fit plate
    position, on held-out pitches. This is the model's miss, not a measurement error."""
    miss: dict[str, list[float]] = {}
    for _, p in _pitches(d):
        if p["status"] == "ok":
            r, s = p["plate_real"], p["plate_sim"]
            miss.setdefault(p["pitch_type"], []).append(
                float(np.hypot(r[0] - s[0], r[1] - s[1])) * 12.0)
    return {t: {"n": len(m), "median_miss_in": _med(m)} for t, m in miss.items()}


def compare(a: dict, b: dict) -> dict:
    """Pitch types both pitchers throw, side by side. Each field is {a, b, diff}, diff = b - a
    (None if either side has no number). Types only one throws are listed under only_a / only_b.
    Compare n before trusting a diff: a type thrown a handful of times is noise."""
    ar_a, ar_b = arsenal(a), arsenal(b)
    er_a, er_b = plate_error(a), plate_error(b)

    def pair(x, y):
        return {"a": x, "b": y, "diff": None if x is None or y is None else round(y - x, 2)}

    shared = {}
    for t in ar_a.keys() & ar_b.keys():
        row = {k: pair(v, ar_b[t][k]) for k, v in ar_a[t].items()}   # includes n
        row["median_miss_in"] = pair(er_a.get(t, {}).get("median_miss_in"),
                                     er_b.get(t, {}).get("median_miss_in"))
        shared[t] = row
    return {"a": a["pitcher"]["name"], "b": b["pitcher"]["name"], "shared": shared,
            "only_a": sorted(ar_a.keys() - ar_b.keys()), "only_b": sorted(ar_b.keys() - ar_a.keys())}


def sequence(d: dict, stand: str | None = None, balls: int | None = None,
             strikes: int | None = None) -> dict:
    """What he throws next, given the previous pitch type in the at-bat ("start" = first pitch).
    Per row: n transitions, and for each next type its count n, share p, and median separation
    (in) at the decision point from the previous pitch (None for "start"). Filters: batter side
    `stand` ("L"/"R"), and the count the NEXT pitch is thrown in. Observed frequencies, unsmoothed:
    read n before trusting p."""
    cells: dict[str, dict[str, list]] = {}
    for ab in d["at_bats"]:
        if stand and ab["stand"] != stand:
            continue
        prev = None
        for p in ab["pitches"]:
            ok = (p["pitch_type"] and (balls is None or p["balls"] == balls)
                  and (strikes is None or p["strikes"] == strikes))
            if ok:
                cells.setdefault(prev["pitch_type"] if prev else "start", {}) \
                     .setdefault(p["pitch_type"], []).append(p["tunnel"]["sep_prev_in"] if prev else None)
            prev = p if p["pitch_type"] else None   # untyped pitch breaks the chain
    out = {}
    for a, nxt in cells.items():
        n = sum(len(v) for v in nxt.values())
        out[a] = {"n": n, "next": {b: {"n": len(v), "p": round(len(v) / n, 3), "median_sep_in": _med(v)}
                                   for b, v in sorted(nxt.items(), key=lambda kv: -len(kv[1]))}}
    return out
