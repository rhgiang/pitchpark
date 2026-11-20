"""Read-only queries over the M4 viewer JSON (web/public/data/<slug>.json), for the MCP server.

Nothing here simulates or fetches: it aggregates what export.export_pitcher already wrote
(held-out games only, Statcast-fit "real" paths, M1+M3 "sim" paths), so every number is the
same one the viewer shows. Each function takes the loaded JSON dict, so it is testable without MCP.
"""
from __future__ import annotations

import functools
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "web" / "public" / "data"
M6_SUMMARY = Path(__file__).resolve().parent / "sequence" / "m6_held_out.json"   # tracked copy of M6 numbers
M2_DIR = Path(__file__).resolve().parents[2] / "results" / "m2"   # gitignored: needs train_generative.py run locally


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


def generator_check(slug: str, pitch_type: str | None = None, m2_dir: Path = M2_DIR) -> list[dict]:
    """M2 evaluation rows (summary.csv) for one pitcher, optionally one pitch type: AUC of real
    held-out vs generated pitches, the real-vs-real floor, gap, margin, and pass. `pass` means gap <=
    margin (see results/M2.md); graded=False rows (under 30 held-out pitches) have no verdict."""
    if not re.fullmatch(r"[a-z0-9-]+", slug):
        raise ValueError(f"bad slug {slug!r}; use a slug from list_pitchers")
    f = m2_dir / slug / "summary.csv"
    if not f.exists():
        raise FileNotFoundError(f"no M2 results for {slug!r}; run scripts/train_generative.py on its CSV")
    df = pd.read_csv(f)
    if pitch_type:
        df = df[df["pitch_type"] == pitch_type]
    return json.loads(df.to_json(orient="records"))   # NaN -> null


def sequence_model_check(slug: str, summary: Path = M6_SUMMARY) -> dict:
    """M6 held-out log loss per model (mix, count, logistic, logistic+tto, boosted, transformer;
    lower is better, 5 game folds) and the largest calibration gap, for pitchers M6 evaluated.
    Others get evaluated=False, not an error: next_pitch_probs still works for them."""
    data = json.loads(summary.read_text(encoding="utf-8"))
    name = next((k for k in data if slug.startswith(_slugify(k) + "-")), None)   # keys are surnames
    if name is None:
        return {"evaluated": False, "message": f"M6 did not evaluate {slug!r}; its probabilities are untested"}
    return {"evaluated": True, **data[name]}


def _slugify(name: str) -> str:
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")


RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
_C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3]   # same grid M6 tunes over


@functools.lru_cache(maxsize=8)
def _fit_logistic_tto(slug: str, raw_dir: Path = RAW_DIR):
    """Logistic+tto fit on ALL of this pitcher's cached games, with C tuned on a 30% game slice
    (as M6 does per fold). Returns (model, types, prev_types). Needs data/raw/*.csv (gitignored)."""
    from pitchsim.data import load_seasons
    from pitchsim.sequence.baseline import log_loss
    from pitchsim.sequence.context import add_context
    from pitchsim.sequence.logistic import FEATURES, LogisticSeq
    from pitchsim.validation import split_by_game

    files = [f for f in sorted(raw_dir.glob("*.csv"))
             if f.stem[0].isdigit() and _slugify(str(pd.read_csv(f, nrows=1)["player_name"].iloc[0])) == slug]
    if not files:
        raise FileNotFoundError(f"no raw Statcast CSV for {slug!r} in data/raw/")
    df = add_context(load_seasons(files)).dropna(subset=["pitch_type"])   # after add_context: untyped pitches break chains
    if len(df) < 300:
        raise ValueError(f"only {len(df)} pitches for {slug!r}; too few to fit")
    types = sorted(df["pitch_type"].unique())
    feats = FEATURES + ["tto"]
    fit, val = split_by_game(df, seed=0)
    y = val["pitch_type"].map({t: i for i, t in enumerate(types)}).to_numpy()
    c = min(_C_GRID, key=lambda c: log_loss(LogisticSeq(types, C=c, features=feats).fit(fit).predict(val), y))
    return LogisticSeq(types, C=c, features=feats).fit(df), types, set(df["prev_type"])


def next_pitch_probs(slug: str, balls: int, strikes: int, stand: str, prev_type: str = "start",
                     tto: int = 1, sample: bool = False, seed: int = 0) -> dict:
    """P(next pitch type) from M6's logistic+tto model, fit on all this pitcher's cached games.
    prev_type: a pitch code from the pitcher's arsenal, "start" (first pitch) or "unknown". tto: times
    through the order (3 = third or later). With sample=True also draws one type (seeded)."""
    if not re.fullmatch(r"[a-z0-9-]+", slug):
        raise ValueError(f"bad slug {slug!r}; use a slug from list_pitchers")
    if balls not in range(4) or strikes not in range(3) or stand not in ("L", "R") or tto not in (1, 2, 3):
        raise ValueError("need balls 0-3, strikes 0-2, stand L or R, tto 1-3")
    model, types, prevs = _fit_logistic_tto(slug)
    if prev_type not in prevs:
        raise ValueError(f"prev_type {prev_type!r} not in {sorted(prevs)}")
    row = pd.DataFrame({"count": [f"{balls}-{strikes}"], "stand": [stand],
                        "prev_type": [prev_type], "tto": [str(min(tto, 3))]})
    p = model.predict(row)[0]
    out = {"probs": {t: round(float(x), 3) for t, x in sorted(zip(types, p), key=lambda tx: -tx[1])},
           "model": "logistic+tto, fit on all cached games (not scored on held-out games; see sequence_model_check)"}
    if sample:
        out["sampled"] = model.sample(row, np.random.default_rng(seed))[0]
    return out
