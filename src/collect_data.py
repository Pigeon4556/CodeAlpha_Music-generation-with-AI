"""
Step 1 - COLLECT MIDI DATA.

Three ways to fill `data/midi/`:

  1. Real MIDI files you already have  -> just copy them into data/midi/real/
  2. Direct MIDI links                 -> put one URL per line in data/sources.txt, then
                                          python -m src.collect_data --urls data/sources.txt
  3. Synthetic raga corpus (bootstrap) -> python -m src.collect_data --synthetic

Use only music you have the right to use (your own transcriptions, public-domain or
permissively licensed files). Then `--report` shows what was collected.
"""
import argparse
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from . import config, raga_corpus

MIDI_EXT = {".mid", ".midi"}


def download(urls_file, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for line in Path(urls_file).read_text().splitlines():
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        name = Path(urlparse(url).path).name or "download.mid"
        if Path(name).suffix.lower() not in MIDI_EXT:
            name += ".mid"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r, open(out_dir / name, "wb") as f:
                f.write(r.read())
            ok += 1
            print(f"  downloaded {name}")
        except Exception as e:  # keep going if one link is dead
            print(f"  FAILED {url}: {e}")
    print(f"Downloaded {ok} file(s) to {out_dir}")


def report():
    files = [p for p in config.MIDI_DIR.rglob("*") if p.suffix.lower() in MIDI_EXT]
    by_folder = {}
    for p in files:
        by_folder.setdefault(p.parent.name, 0)
        by_folder[p.parent.name] += 1
    print(f"MIDI files found under {config.MIDI_DIR}: {len(files)}")
    for k, v in sorted(by_folder.items()):
        print(f"  {k:<12} {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urls", help="text file with one direct .mid URL per line")
    ap.add_argument("--synthetic", action="store_true", help="generate the synthetic raga corpus")
    ap.add_argument("--pieces-per-raga", type=int, default=30)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.urls:
        download(a.urls, config.MIDI_DIR / "real")
    if a.synthetic:
        raga_corpus.build_corpus(config.MIDI_DIR / "synthetic", pieces_per_raga=a.pieces_per_raga)
    report()
