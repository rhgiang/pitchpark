import json
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd

from pitchsim.pose import clips

FIX = Path(__file__).parent / "fixtures"


def test_bases_empty():
    df = pd.DataFrame({"on_1b": [np.nan, 5.0, np.nan], "on_2b": [np.nan] * 3,
                       "on_3b": [np.nan, np.nan, 7.0]})
    assert clips.bases_empty(df).tolist() == [True, False, False]


def test_play_ids_from_feed():
    feed = json.loads((FIX / "feed_min.json").read_text())
    out = clips.play_ids_from_feed(feed)
    got = out.set_index(["at_bat_number", "pitch_number"])["play_id"]
    assert got[(1, 2)] == "aaa-2"  # pickoff between does not shift numbering
    assert got[(2, 1)] == "bbb-1"
    assert pd.isna(got[(2, 2)])


def test_clip_url_from_page():
    assert clips.clip_url_from_page((FIX / "sporty_page.html").read_text()) == \
        "https://sporty-clips.mlb.com/abc123_1280x720_59.mp4"
    assert clips.clip_url_from_page("<html></html>") is None
    # Real Savant pages sometimes escape the base64 padding.
    assert clips.clip_url_from_page("<a href='https://sporty-clips.mlb.com/Zm9v&#x3D;&#x3D;.mp4'>") \
        == "https://sporty-clips.mlb.com/Zm9v==.mp4"


def test_fetch_clips_records_failures(tmp_path, monkeypatch):
    df = pd.DataFrame({"game_pk": [1] * 4, "at_bat_number": [1, 2, 3, 4],
                       "pitch_number": [1] * 4})
    ids = {1: None, 2: "nomp4", 3: "boom", 4: "good"}
    monkeypatch.setattr(clips, "play_ids", lambda d, **k: pd.Series(
        [ids[a] for a in d["at_bat_number"]], index=d.index))

    def page(pid):
        if pid == "boom":
            raise urllib.error.HTTPError("u", 500, "x", None, None)
        return "<html></html>" if pid == "nomp4" else (FIX / "sporty_page.html").read_text()
    monkeypatch.setattr(clips, "_get_page", page)
    monkeypatch.setattr(clips, "_download", lambda url, path: Path(path).write_bytes(b"x"))
    out = clips.fetch_clips(df, tmp_path)
    assert out["status"].tolist() == ["no_play_id", "no_mp4", "http_error", "ok"]
    assert clips.fetch_clips(df, tmp_path)["status"].iloc[3] == "cached"


def test_label_paths_follow_the_pitcher_slug(tmp_path):
    p = clips.label_path("some-other-pitcher", "release_frames")
    assert p.name == "some-other-pitcher_release_frames.csv" and p.parent == clips.LABELS_DIR
    # a pitcher with no hand labels yet gets an empty table, not a crash
    assert clips.read_labels("some-other-pitcher", "release_frames").empty
    # the committed Ohtani labels are found under his clips-directory slug
    assert not clips.read_labels("ohtani-shohei", "release_frames").empty


# --- download-list helpers; all data below is SYNTHETIC ---
def _synthetic_rows(n_games=4, per_game=5):
    return pd.DataFrame([(g, i) for g in range(1, n_games + 1) for i in range(per_game)],
                        columns=["game_pk", "at_bat_number"])


def test_round_robin_spreads_over_games():
    out = clips.round_robin(_synthetic_rows(), 6, seed=0)
    assert len(out) == 6
    assert out["game_pk"].value_counts().max() == 2  # 4 games: one pass of 4, then 2 more
    assert clips.round_robin(_synthetic_rows(), 6, seed=0).equals(out)  # seeded
    assert len(clips.round_robin(_synthetic_rows(1, 3), 10)) == 3


def test_interleave_alternates_then_tail():
    a, b = pd.DataFrame({"x": [1, 2, 3]}), pd.DataFrame({"x": [10]})
    assert clips.interleave(a, b)["x"].tolist() == [1, 10, 2, 3]


def test_mp4_basename_from_page_matches_saved_name():
    html = '<a href="https://sporty-clips.mlb.com/p/abc%2Bdef&#x3D;.mp4">'
    assert clips.mp4_basename(clips.clip_url_from_page(html)) == "abc+def=.mp4"


def test_match_downloads_reports_everything():
    expected = {"1_1_1.mp4": "A.mp4", "1_1_2.mp4": "B.mp4", "1_1_3.mp4": "C.mp4", "1_1_4.mp4": None}
    files = ["A.mp4", "1_1_2.mp4", "A (1).mp4", "Z.mp4", "D.mp4.crdownload", "notes.txt"]
    r = clips.match_downloads(expected, files)
    assert r["renames"] == {"A.mp4": "1_1_1.mp4"}
    assert r["have"] == ["1_1_2.mp4"]
    assert r["missing"] == ["1_1_3.mp4", "1_1_4.mp4"]
    assert r["duplicates"] == ["A (1).mp4"]
    assert r["partial"] == ["D.mp4.crdownload"]
    assert r["leftovers"] == ["Z.mp4"]
    assert clips.match_downloads(expected, ["1_1_1.mp4"])["renames"] == {}  # idempotent


def test_listed_clip_names_reads_every_download_list(tmp_path):
    """SYNTHETIC lists: a second download round must skip pitches any earlier list already named."""
    cols = "game_pk,at_bat_number,pitch_number,play_id\n"
    (tmp_path / "download_list.csv").write_text(cols + "1,2,3,a\n4,5,6,b\n")
    (tmp_path / "download_list_2.csv").write_text(cols + "7,8,9,c\n")
    (tmp_path / "clips.csv").write_text(cols + "9,9,9,z\n")        # not a download list
    assert clips.listed_clip_names(tmp_path) == {"1_2_3.mp4", "4_5_6.mp4", "7_8_9.mp4"}
    assert clips.listed_clip_names(tmp_path / "nope") == set()


def test_venue_yield_table_and_smoothing_synthetic():
    # SYNTHETIC: venue A 4/4 ok, B 0/4, overall prior 0.5
    df = pd.DataFrame({"home_team": list("AAAABBBB"), "game_pk": [1, 1, 2, 2, 3, 3, 4, 4],
                       "ok": [True] * 4 + [False] * 4})
    t = clips.venue_yield_table(df).set_index("home_team")
    assert t.loc["A", "usable_frac"] == 1.0 and t.loc["A", "n_games"] == 2
    assert t.loc["A", "smoothed"] == (4 + 4 * 0.5) / 8 and t.loc["B", "smoothed"] == 0.25
    assert clips.smoothed_yield(0, 0, 0.3) == 0.3  # unseen venue = prior


def test_pick_by_yield_prefers_good_venues_synthetic():
    # SYNTHETIC: games 1-4 at GOOD (0.6), 5-8 at BAD (0.02), 9-10 at UNSEEN
    rows = pd.DataFrame({"game_pk": np.repeat(range(1, 11), 3), "x": range(30)})
    home = pd.Series(["GOOD"] * 4 + ["BAD"] * 4 + ["NEW"] * 2, index=range(1, 11))
    venue = pd.DataFrame({"home_team": ["GOOD", "BAD"], "smoothed": [0.6, 0.02]})
    out = clips.pick_by_yield(rows, 6, home, venue)
    assert set(out["game_pk"].map(home)) <= {"GOOD", "NEW"} and out["game_pk"].nunique() == 6
    assert set(out["game_pk"]) >= {1, 2, 3, 4}  # best-scored games come first (UNSEEN gets mean 0.31)
    short = clips.pick_by_yield(rows, 28, home, venue)  # not enough good games: BAD stays, ranked last
    assert len(short) == 28
