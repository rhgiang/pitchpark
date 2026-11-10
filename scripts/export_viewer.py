"""M4: export pitchers' held-out at-bats as JSON for the batter's-eye viewer.

    python scripts/export_viewer.py data/raw/<pitcher>.csv [more.csv ...] [--out web/public/data]

Writes <out>/<slug>.json per CSV and <out>/index.json = [{slug, name, throws, n_at_bats}].
CSV paths are explicit (no globbing data/raw/). Synthetic pitchers may only be written to
web/test/fixtures/ so nothing fake lands next to real pitchers in the viewer.
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from pitchsim.data import add_air_density, fetch_game_conditions
from pitchsim.export import export_pitcher, write_json
from pitchsim.physics import air_density

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "web" / "test" / "fixtures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "web" / "public" / "data"))
    ap.add_argument("--no-weather", action="store_true",
                    help="use a constant air density (synthetic data)")
    ap.add_argument("--dt", type=float, default=0.005, help="path sampling step (s)")
    args = ap.parse_args()
    out = Path(args.out)

    index = []
    for csv in args.csv:
        raw = pd.read_csv(csv)
        name = str(raw["player_name"].iloc[0])
        # Strip accents: the slug regex would turn an accented letter into a hyphen.
        plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
        if "Synthetic" in name and FIXTURES.resolve() not in [out.resolve(), *out.resolve().parents]:
            raise SystemExit(f"Synthetic data may only be written to {FIXTURES}")
        slug = re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")

        if args.no_weather:
            rho = pd.Series(air_density(70.0, 0.0), index=raw.index)
        else:
            rho = add_air_density(raw, fetch_game_conditions(raw["game_pk"].unique()))["rho"]
        data = export_pitcher(raw, rho, dt_s=args.dt)
        path = out / f"{slug}.json"
        write_json(data, path)
        counts = pd.Series([p["status"] for a in data["at_bats"] for p in a["pitches"]])
        print(f"{plain}: {len(data['at_bats'])} at-bats, "
              f"{counts.value_counts().to_dict()}, {path.stat().st_size / 1e6:.2f} MB -> {path}")
        index.append({"slug": slug, "name": name, "throws": data["pitcher"]["throws"],
                      "n_at_bats": len(data["at_bats"])})

    (out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"wrote {out / 'index.json'}")


if __name__ == "__main__":
    main()
