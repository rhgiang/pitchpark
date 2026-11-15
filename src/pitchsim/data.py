"""
Fetch Statcast pitches (cached on disk) and attach per-pitch air density.

Air density depends on the park's elevation and the game-time temperature; the
temperature comes from the MLB StatsAPI live game feed, not from Statcast.
"""

import json
import re
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from pitchsim.physics import air_density

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"

# venue_id -> (name, elevation ft).  Ids, names and elevations come from
# https://statsapi.mlb.com/api/v1/venues?sportId=1&hydrate=location
# (location.elevation, whole feet).  30 current parks + Steinbrenner Field (Rays, 2025).
VENUE_ELEVATION_FT = {
    1: ("Angel Stadium", 151),
    2: ("Oriole Park at Camden Yards", 33),
    3: ("Fenway Park", 21),
    4: ("Rate Field", 595),
    5: ("Progressive Field", 653),
    7: ("Kauffman Stadium", 856),
    12: ("Tropicana Field", 15),
    14: ("Rogers Centre", 270),
    15: ("Chase Field", 1086),
    17: ("Wrigley Field", 595),
    19: ("Coors Field", 5190),
    22: ("UNIQLO Field at Dodger Stadium", 515),
    31: ("PNC Park", 780),
    32: ("American Family Field", 597),
    680: ("T-Mobile Park", 10),
    2392: ("Daikin Park", 45),
    2394: ("Comerica Park", 600),
    2395: ("Oracle Park", 0),
    2523: ("George M. Steinbrenner Field", 34),
    2529: ("Sutter Health Park", 24),
    2602: ("Great American Ball Park", 535),
    2680: ("Petco Park", 23),
    2681: ("Citizens Bank Park", 20),
    2889: ("Busch Stadium", 460),
    3289: ("Citi Field", 10),
    3309: ("Nationals Park", 35),
    3312: ("Target Field", 828),
    3313: ("Yankee Stadium", 55),
    4169: ("loanDepot park", 10),
    4705: ("Truist Park", 1001),
    5325: ("Globe Life Field", 545),
    # Mexico City Series. 7,349 ft per FanGraphs (blogs.fangraphs.com/mexico-city-series-provided-
    # an-elevated-run-and-entertainment-environment) and Arizona Sports; one article gives 7,320.
    5340: ("Estadio Alfredo Harp Helu", 7349),
}


# (pitcher id, season, Statcast pitch_type) -> the pitch_type we model it as. Only for a pitch
# Statcast labelled differently in different seasons, so that pooling seasons by label doesn't
# put two different pitches in one model. Evidence is season means from data/raw:
#   Misiorowski 2025 "SL": 94.1 mph, spin axis 181 deg, 2.6 in arm-side / 6.4 in rise. That is his
#   2026 "FC" (95.8 mph, 195 deg, 1.4 / 7.2 in), not his 2026 "SL" (91.9 mph, 124 deg, 3.4 / 1.8 in).
RELABEL = {(694819, 2025, "SL"): "FC"}


def load_seasons(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Raw Statcast CSVs (one or more seasons, any pitchers) joined, with RELABEL applied.

    Rows are only relabelled when the CSVs carry `pitcher` and `game_year` (real Statcast does;
    the synthetic CSV doesn't). A relabelled row also takes the pitch_name its new type has.
    """
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if not {"pitcher", "game_year"} <= set(df):
        return df
    for (pid, year, old), new in RELABEL.items():
        hit = (df["pitcher"] == pid) & (df["game_year"] == year) & (df["pitch_type"] == old)
        if hit.any():
            names = df.loc[(df["pitcher"] == pid) & (df["pitch_type"] == new), "pitch_name"]
            df.loc[hit, "pitch_type"] = new
            if not names.empty:
                df.loc[hit, "pitch_name"] = names.iloc[0]
            print(f"relabelled {int(hit.sum())} pitches: pitcher {pid} {year} {old} -> {new}")
    return df


def mark_cross_season(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df with split = "test" for the newest season's games, "train" for the rest.

    validation.split_by_game uses this column instead of its random split, so every model is
    trained on earlier seasons and scored on the latest one: how well it survives a pitcher's
    year-to-year drift, which a pooled random split of games hides.
    """
    if df["game_year"].nunique() < 2:
        raise SystemExit("--cross-season needs at least two seasons of data")
    return df.assign(split=(df["game_year"] == df["game_year"].max()).map({True: "test", False: "train"}))


def load_pitcher(paths: Iterable[str | Path], cross_season: bool = False) -> pd.DataFrame:
    """One pitcher's season CSVs via load_seasons, marked for the cross-season split if asked."""
    df = load_seasons(paths)
    if df["player_name"].nunique() > 1:
        raise SystemExit("These CSVs hold more than one pitcher; pass one pitcher's seasons")
    return mark_cross_season(df) if cross_season else df


def lookup_id(full_name: str) -> int:
    """MLBAM id for a "First Last" name (most recent player if several match)."""
    if " " not in full_name.strip():
        raise SystemExit(f"Need a first and last name, got '{full_name}'")
    from pybaseball import playerid_lookup  # slow import, only needed for name lookups
    first, last = full_name.strip().split(" ", 1)
    res = playerid_lookup(last, first)
    res = res.dropna(subset=["key_mlbam"])
    if res.empty:
        raise SystemExit(f"No MLB player found for '{full_name}'")
    if len(res) > 1:
        print("Several matches, using the most recent:\n",
              res[["name_first", "name_last", "key_mlbam", "mlb_played_last"]])
        res = res.sort_values("mlb_played_last", ascending=False)
    return int(res.iloc[0]["key_mlbam"])


def fetch_pitcher(name: str | None, mlbam_id: int | None, start: str, end: str) -> Path:
    """Regular-season pitches for one pitcher, cached as data/raw/<id>_<start>_<end>.csv."""
    pid = mlbam_id or lookup_id(name)
    path = RAW_DIR / f"{pid}_{start}_{end}.csv"
    if path.exists():
        print(f"cached: {path}")
        return path
    from pybaseball import statcast_pitcher
    df = statcast_pitcher(start, end, pid)
    if not df.empty:
        df = df[df["game_type"] == "R"]  # drop spring training / postseason
    if df.empty:
        raise SystemExit("No regular-season pitches returned; check the id and date range.")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def parse_game_feed(feed: dict) -> dict:
    """game_pk, venue and temperature (nan if absent) from a StatsAPI live feed."""
    game = feed["gameData"]
    try:
        temp = float(game.get("weather", {}).get("temp"))
    except (TypeError, ValueError):
        temp = float("nan")
    return {"game_pk": int(feed["gamePk"]), "venue_id": int(game["venue"]["id"]),
            "venue_name": game["venue"]["name"], "temp_f": temp}


def fetch_game_conditions(game_pks: Iterable[int],
                          cache_path: Path = RAW_DIR / "game_conditions.csv") -> pd.DataFrame:
    """Venue + temperature per game.  Only uncached games hit the network; each row is
    appended as it arrives, so a failure part-way keeps its progress (just rerun)."""
    pks = [int(p) for p in dict.fromkeys(game_pks)]
    cache_path = Path(cache_path)
    have = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()
    cached = set(have.get("game_pk", []))
    todo = [p for p in pks if p not in cached]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for i, pk in enumerate(todo, 1):
        with urllib.request.urlopen(FEED_URL.format(pk=pk), timeout=30) as r:
            row = pd.DataFrame([parse_game_feed(json.load(r))])
        row.to_csv(cache_path, mode="a", header=not cache_path.exists(), index=False)
        if i % 25 == 0:
            print(f"fetched {i}/{len(todo)} games")
    out = pd.read_csv(cache_path)
    return out[out["game_pk"].isin(pks)].reset_index(drop=True)


def add_air_density(df: pd.DataFrame, conditions: pd.DataFrame) -> pd.DataFrame:
    """Copy of df with rho (lb/ft^3) per pitch; missing temp -> 70 F, unknown venue -> 0 ft."""
    cond = conditions.drop_duplicates("game_pk").set_index("game_pk")
    temp = df["game_pk"].map(cond["temp_f"])
    venue_id = df["game_pk"].map(cond["venue_id"])
    known = venue_id.isin(VENUE_ELEVATION_FT)
    elev = venue_id.map(lambda v: VENUE_ELEVATION_FT.get(v, ("", 0.0))[1])

    n_temp = int(temp.isna().sum())
    print(f"air density: {n_temp} of {len(df)} pitches have no temperature (using 70 F)")
    unknown = cond.loc[~cond["venue_id"].isin(VENUE_ELEVATION_FT), ["venue_id", "venue_name"]]
    unknown = unknown[unknown.index.isin(df["game_pk"])].drop_duplicates()
    print(f"air density: {int((~known).sum())} pitches at unknown venues (using 0 ft)"
          + "".join(f"\n  {r.venue_name} (id {r.venue_id})" for r in unknown.itertuples()))

    out = df.copy()
    out["rho"] = [air_density(t, e) for t, e in zip(temp.fillna(70.0), elev)]
    return out


PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people/{id}"


def parse_height(text: str) -> int:
    """StatsAPI height like ``6' 3"`` -> inches (75)."""
    m = re.fullmatch(r"\s*(\d+)'\s*(\d+)\"?\s*", text)
    if not m:
        raise ValueError(f"unparseable height: {text!r}")
    return int(m[1]) * 12 + int(m[2])


def fetch_heights(ids: Iterable[int], cache_path: Path = RAW_DIR / "people.csv") -> pd.Series:
    """Height in inches per MLBAM id (index = id; NaN where unknown).  Only uncached ids hit the network.

    The request carries a generic User-Agent and nothing else.  A failed request or a record without a
    height gives NaN and a printed warning, and is not cached (a later run retries it), so one bad id
    does not stop an export.  An empty cache file counts as no cache."""
    ids = [int(i) for i in dict.fromkeys(ids)]
    cache_path = Path(cache_path)
    has_cache = cache_path.exists() and cache_path.stat().st_size > 0
    have = pd.read_csv(cache_path) if has_cache else pd.DataFrame(columns=["id", "height_in"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    got = {}
    for i in [i for i in ids if i not in set(have["id"])]:
        req = urllib.request.Request(PEOPLE_URL.format(id=i), headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                h = parse_height(json.load(r)["people"][0]["height"])
        except (OSError, KeyError, IndexError, ValueError) as e:  # URLError/timeout are OSErrors
            print(f"WARNING: no height for MLBAM id {i} ({type(e).__name__}: {e}); using NaN")
            got[i] = float("nan")
            continue
        new = not (cache_path.exists() and cache_path.stat().st_size > 0)
        pd.DataFrame([{"id": i, "height_in": h}]).to_csv(cache_path, mode="w" if new else "a",
                                                          header=new, index=False)
        got[i] = h
    out = have.drop_duplicates("id").set_index("id")["height_in"].astype(float).to_dict() | got
    return pd.Series([out[i] for i in ids], index=pd.Index(ids, name="id"), name="height_in")
