"""
Pitch rows -> MLB playId -> cached Baseball Savant broadcast clip (mp4).

Chain: StatsAPI live feed gives each pitch a ``playId``; Savant's ``sporty-videos`` page for
that id links one mp4.  Clips are for personal research only (see CLAUDE.md): never commit them.
"""

import html as htmllib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from pitchsim.data import FEED_URL, RAW_DIR

PAGE_URL = "https://baseballsavant.mlb.com/sporty-videos?playId={pid}"
HEADERS = {"User-Agent": "Mozilla/5.0"}  # generic on purpose; nothing personal is sent
KEY = ["game_pk", "at_bat_number", "pitch_number"]
_MP4 = re.compile(r"https://sporty-clips\.mlb\.com/[^\"'\s<>]+?\.mp4")


def bases_empty(df: pd.DataFrame) -> pd.Series:
    """True where no runner is on 1b/2b/3b (Statcast leaves on_Xb NaN when the base is empty).
    Bases-empty means the windup, not the stretch, so one delivery type to model."""
    return df[["on_1b", "on_2b", "on_3b"]].isna().all(axis=1)


def clip_name(row) -> str:
    """File name for one pitch's clip."""
    return f"{int(row['game_pk'])}_{int(row['at_bat_number'])}_{int(row['pitch_number'])}.mp4"


def play_ids_from_feed(feed: dict) -> pd.DataFrame:
    """(game_pk, at_bat_number, pitch_number, play_id) for every pitch in a live feed.
    Only ``isPitch`` events count: pickoffs/steps-offs sit in playEvents but Statcast's
    pitch_number ignores them."""
    rows = []
    for ab, play in enumerate(feed["liveData"]["plays"]["allPlays"], 1):
        pitches = [e for e in play.get("playEvents", []) if e.get("isPitch")]
        rows += [(int(feed["gamePk"]), ab, n, e.get("playId")) for n, e in enumerate(pitches, 1)]
    return pd.DataFrame(rows, columns=KEY + ["play_id"])


def play_ids(df: pd.DataFrame, cache_path: Path = RAW_DIR / "play_ids.csv") -> pd.Series:
    """play_id per row of df (NaN if none).  One feed per uncached game, appended to the
    cache as it arrives so a failure keeps progress (same pattern as fetch_game_conditions)."""
    cache_path = Path(cache_path)
    have = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=KEY + ["play_id"])
    todo = [int(p) for p in df["game_pk"].unique() if p not in set(have["game_pk"])]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for pk in todo:
        try:
            req = urllib.request.Request(FEED_URL.format(pk=pk), headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                new = play_ids_from_feed(json.load(r))
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"feed {pk} failed: {e}")  # not cached, so a rerun retries it
            continue
        new.to_csv(cache_path, mode="a", header=not cache_path.exists(), index=False)
    have = pd.read_csv(cache_path).drop_duplicates(KEY)
    merged = df[KEY].merge(have, on=KEY, how="left")
    return pd.Series(merged["play_id"].to_numpy(), index=df.index)


def clip_url_from_page(html: str) -> str | None:
    """First sporty-clips mp4 link in a Savant sporty-videos page, else None."""
    m = _MP4.search(html)
    # Some pages HTML-escape the link's base64 padding ('=' as '&#x3D;').
    return htmllib.unescape(m.group(0)) if m else None


def _get_page(pid: str) -> str:
    req = urllib.request.Request(PAGE_URL.format(pid=pid), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def _download(url: str, path: Path) -> None:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
        f.write(r.read())


def fetch_clips(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Download one clip per row into out_dir (sequential, skips existing files).
    Never raises on a per-pitch failure; status is ok | cached | no_play_id | no_mp4 | http_error."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = play_ids(df)
    rows = []
    for (_, r), pid in zip(df.iterrows(), ids):
        path = out_dir / clip_name(r)
        if path.exists():   # a saved clip counts even if its feed lookup failed
            status = "cached"
        elif pd.isna(pid):
            status = "no_play_id"
        else:
            try:
                url = clip_url_from_page(_get_page(pid))
                if url is None:
                    status = "no_mp4"
                else:
                    _download(url, path)
                    status = "ok"
            except (urllib.error.URLError, TimeoutError, OSError):
                path.unlink(missing_ok=True)  # no half-written clip
                status = "http_error"
        rows.append({**{k: int(r[k]) for k in KEY}, "play_id": pid,
                     "clip_path": str(path) if status in ("ok", "cached") else None,
                     "status": status})
    return pd.DataFrame(rows)
