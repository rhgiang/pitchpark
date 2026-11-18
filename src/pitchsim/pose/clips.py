"""
Pitch rows -> MLB playId -> cached Baseball Savant broadcast clip (mp4).

Chain: StatsAPI live feed gives each pitch a ``playId``; Savant's ``sporty-videos`` page for
that id links one mp4.  Clips are for personal research only (see CLAUDE.md): never commit them.
"""

import html as htmllib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from pitchsim.data import FEED_URL, RAW_DIR

PAGE_URL = "https://baseballsavant.mlb.com/sporty-videos?playId={pid}"
HEADERS = {"User-Agent": "Mozilla/5.0"}  # generic on purpose; nothing personal is sent
KEY = ["game_pk", "at_bat_number", "pitch_number"]
_MP4 = re.compile(r"https://sporty-clips\.mlb\.com/[^\"'\s<>]+?\.mp4")


LABELS_DIR = Path(__file__).parent / "labels"


def label_path(slug: str, kind: str) -> Path:
    """Hand-label CSV for one pitcher, e.g. label_path("ohtani-shohei", "release_frames").
    `slug` is the clips directory's name; kinds: release_frames, foot_strike_frames, screen_overrides.
    The file may not exist (a new pitcher has no labels yet): callers decide what that means."""
    return LABELS_DIR / f"{slug}_{kind}.csv"


def read_labels(slug: str, kind: str) -> pd.DataFrame:
    """The labels as a table indexed by clip; empty (no rows) when the pitcher has no such file."""
    p = label_path(slug, kind)
    return pd.read_csv(p).set_index("clip") if p.exists() else pd.DataFrame(index=pd.Index([], name="clip"))


def bases_empty(df: pd.DataFrame) -> pd.Series:
    """True where no runner is on 1b/2b/3b (Statcast leaves on_Xb NaN when the base is empty).
    Bases-empty means the windup, not the stretch, so one delivery type to model."""
    return df[["on_1b", "on_2b", "on_3b"]].isna().all(axis=1)


def clip_name(row) -> str:
    """File name for one pitch's clip."""
    return f"{int(row['game_pk'])}_{int(row['at_bat_number'])}_{int(row['pitch_number'])}.mp4"


def listed_clip_names(clips_dir: Path) -> set[str]:
    """Clip file names already named by any <dir>/download_list*.csv (earlier download rounds), so a new round
    never picks them again even if the file was not saved or was renamed."""
    names: set[str] = set()
    for f in Path(clips_dir).glob("download_list*.csv"):
        names |= {clip_name(r) for _, r in pd.read_csv(f).iterrows()}
    return names


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


def fetch_clips(df: pd.DataFrame, out_dir: Path, download: bool = True) -> pd.DataFrame:
    """Download one clip per row into out_dir (sequential, skips existing files).
    Never raises on a per-pitch failure; status is ok | cached | no_play_id | no_mp4 | http_error.
    download=False only resolves play ids: no clip is requested, status is cached | pending | no_play_id
    (the mp4 host refuses scripts; clips are saved by hand and matched by m5_match_clips.py)."""
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
        elif not download:
            status = "pending"
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


def round_robin(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """n rows, one game at a time in a random game order (then a second pass, ...), so clips
    spread across as many games as possible.  Rows inside a game are shuffled too."""
    rng = np.random.default_rng(seed)
    by_game = df.groupby("game_pk").indices
    queues = [list(rng.permutation(by_game[g])) for g in rng.permutation(list(by_game))]
    pos = []
    while len(pos) < n and any(queues):
        for q in queues:
            if q and len(pos) < n:
                pos.append(q.pop())
    return df.iloc[pos]


def interleave(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """a0, b0, a1, b1, ...; the longer one's tail follows."""
    rows = []
    for i in range(max(len(a), len(b))):
        rows += [x.iloc[i] for x in (a, b) if i < len(x)]
    return pd.DataFrame(rows).reset_index(drop=True)


def mp4_basename(url: str) -> str:
    """Savant's own (percent-decoded) file name for a clip link: what the browser saves it as."""
    return urllib.parse.unquote(url.rsplit("/", 1)[-1])


_DUP = re.compile(r" \(\d+\)\.mp4$")


def match_downloads(expected: dict[str, str | None], files: list[str]) -> dict:
    """Pure part of the matcher.  expected: wanted clip name -> Savant basename (None if no link);
    files: names in the folder.  Returns renames {saved: wanted}, have (already named), missing,
    leftovers (unmatched .mp4), duplicates ("x (1).mp4"), partial (.crdownload)."""
    present = set(files)
    have = [n for n in expected if n in present]
    renames = {b: n for n, b in expected.items() if n not in present and b in present}
    used = set(expected) | set(renames)
    return {
        "have": have, "renames": renames,
        "missing": [n for n in expected if n not in present and n not in renames.values()],
        "duplicates": sorted(f for f in files if _DUP.search(f)),
        "partial": sorted(f for f in files if f.endswith(".crdownload")),
        "leftovers": sorted(f for f in files if f.endswith(".mp4") and f not in used and not _DUP.search(f)),
    }
