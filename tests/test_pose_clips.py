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
