"""One HTML page of Savant video links to save by hand (the mp4 host refuses scripts), M5.

usage: python scripts/m5_download_page.py data/video/<slug> [...] --n-train 25 --n-test 15 --out data/video/download.html
Writes <clips_dir>/download_list.csv per pitcher (input to m5_match_clips.py).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from pitchsim.pose.clips import clip_name, interleave, round_robin

LINK = "https://baseballsavant.mlb.com/sporty-videos?playId={}"
PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>M5 clip downloads</title>
<style>
body{background:#14161a;color:#d8dbe0;font:15px system-ui;margin:0 auto;max-width:900px;padding:16px}
h2{margin:28px 0 4px}code,.mono{font-family:Consolas,monospace;color:#9ccfa0}
table{border-collapse:collapse;width:100%}td,th{padding:4px 8px;border-bottom:1px solid #2a2e35;text-align:left}
a{color:#7ab4ff}button{background:#2a2e35;color:#d8dbe0;border:1px solid #444;border-radius:4px;padding:4px 10px;cursor:pointer}
tr.done{opacity:.4}#bar{position:sticky;top:0;background:#14161a;padding:8px 0;border-bottom:1px solid #2a2e35}
</style>
<div id=bar><button id=next>open next (n)</button> <b id=count></b>
 &nbsp;Personal research only: do not commit or share clips.</div><div id=root></div>
<script>
const SECTIONS=__DATA__;
const key=n=>"m5dl:"+n, get=n=>{try{return localStorage.getItem(key(n))==="1"}catch(e){return false}},
 set=(n,v)=>{try{localStorage.setItem(key(n),v?"1":"0")}catch(e){}};
const root=document.getElementById("root"), rows=[];
for(const s of SECTIONS){
  const h=document.createElement("h2");h.textContent=s.name;root.append(h);
  const p=document.createElement("p");p.innerHTML="Save into <code></code> ";p.querySelector("code").textContent=s.folder;
  const b=document.createElement("button");b.textContent="copy folder path";
  b.onclick=()=>navigator.clipboard.writeText(s.folder);p.append(b);root.append(p);
  const t=document.createElement("table");t.innerHTML="<tr><th>#<th>split<th>pitch<th>video<th>expected file<th>done</tr>";
  s.rows.forEach((r,i)=>{
    const tr=t.insertRow();
    tr.innerHTML=`<td>${i+1}<td>${r.split}<td>${r.pitch_type}<td><a target=_blank rel=noopener href="${r.url}">open video</a><td class=mono>${r.name}<td><input type=checkbox>`;
    const cb=tr.querySelector("input");cb.checked=get(r.name);
    const sync=()=>{tr.className=cb.checked?"done":"";count()};
    cb.onchange=()=>{set(r.name,cb.checked);sync()};tr.querySelector("a").onclick=()=>{cb.checked=true;cb.onchange()};
    rows.push({r,cb,sync});sync();
  });root.append(t);
}
function count(){document.getElementById("count").textContent=rows.filter(x=>x.cb.checked).length+" / "+rows.length+" done"}
function next(){const x=rows.find(x=>!x.cb.checked);if(!x)return;window.open(x.r.url,"_blank");x.cb.checked=true;set(x.r.name,true);x.sync()}
document.getElementById("next").onclick=next;
addEventListener("keydown",e=>{if(e.key==="n"&&!e.ctrlKey&&!e.metaKey)next()});count();
</script>"""


def choose(clips_dir: Path, n_train: int, n_test: int, seed: int = 0) -> pd.DataFrame:
    """Rows to save: round-robin over games per split, train/test alternating; skips clips already on disk."""
    df = pd.read_csv(clips_dir / "clips.csv")
    df = df[df["play_id"].notna()].reset_index(drop=True)
    df = df[[not (clips_dir / clip_name(r)).exists() for _, r in df.iterrows()]]
    tr, te = (round_robin(df[df["split"] == s].reset_index(drop=True), n, seed)
              for s, n in (("train", n_train), ("test", n_test)))
    return interleave(tr, te)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips_dirs", nargs="+")
    ap.add_argument("--n-train", type=int, default=25)
    ap.add_argument("--n-test", type=int, default=15)
    ap.add_argument("--out", default="data/video/download.html")
    a = ap.parse_args()
    sections = []
    for d in map(Path, a.clips_dirs):
        sel = choose(d, a.n_train, a.n_test)
        sel.to_csv(d / "download_list.csv", index=False)
        rows = [{"split": r["split"], "pitch_type": r["pitch_type"], "name": clip_name(r),
                 "url": LINK.format(r["play_id"])} for _, r in sel.iterrows()]
        sections.append({"name": d.name, "folder": str(d.resolve()), "rows": rows})
        print(d.name, len(sel), sel["split"].value_counts().to_dict(), "games", sel["game_pk"].nunique(),
              sel["pitch_type"].value_counts().to_dict())
    data = json.dumps(sections).replace("</", "<\\/")
    Path(a.out).write_text(PAGE.replace("__DATA__", data), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
