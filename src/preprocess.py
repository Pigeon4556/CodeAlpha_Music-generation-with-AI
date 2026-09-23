"""
Step 2 - PREPROCESS MIDI INTO NOTE SEQUENCES (music21).

Every note becomes a token  "P<midi_pitch>_<duration>"  (e.g. "P62_0.5") and every rest
becomes "R_<duration>". Durations are snapped to config.DURATIONS. Chords keep only their
top note (the melody line). If a MIDI file has several tracks, the track with the most
notes is used (assumed to be the melody) unless --all-parts is given.

Output:
  data/processed/vocab.json      token <-> integer mapping
  data/processed/sequences.npz   sliding windows X (N, SEQ_LEN) -> y (N,) and piece ids
"""
import argparse
import json
from collections import Counter

import numpy as np
from music21 import converter

from . import config

MIDI_EXT = {".mid", ".midi"}


def snap(q):
    return min(config.DURATIONS, key=lambda d: abs(d - q))


def midi_to_tokens(path, all_parts=False):
    score = converter.parse(str(path))
    parts = list(score.parts) if score.parts else []
    if parts and not all_parts and len(parts) > 1:
        source = max(parts, key=lambda p: len(p.flatten().notes))
    else:
        source = score
    tokens = []
    for el in source.flatten().notesAndRests:
        q = float(el.duration.quarterLength)
        if q <= 0:                       # grace notes
            continue
        d = snap(q)
        if el.isRest:
            tokens.append(f"R_{d}")
        elif el.isChord:
            tokens.append(f"P{max(p.midi for p in el.pitches)}_{d}")
        elif el.isNote:
            tokens.append(f"P{el.pitch.midi}_{d}")
    while tokens and tokens[0].startswith("R_"):
        tokens.pop(0)
    while tokens and tokens[-1].startswith("R_"):
        tokens.pop()
    return tokens


def main(seq_len, all_parts, min_count):
    files = sorted(p for p in config.MIDI_DIR.rglob("*") if p.suffix.lower() in MIDI_EXT)
    if not files:
        raise SystemExit("No MIDI files found. Run: python -m src.collect_data --synthetic  "
                         "(or copy real MIDI files into data/midi/real/)")
    pieces = []
    for i, f in enumerate(files, 1):
        try:
            t = midi_to_tokens(f, all_parts)
        except Exception as e:
            print(f"skip {f.name}: {e}")
            continue
        if len(t) > seq_len + 1:
            pieces.append(t)
        if i % 50 == 0 or i == len(files):
            print(f"parsed {i}/{len(files)} files")

    counts = Counter(t for p in pieces for t in p)
    keep = {t for t, c in counts.items() if c >= min_count}
    pieces = [[t for t in p if t in keep] for p in pieces]      # drop very rare tokens
    vocab = sorted(keep)
    tok2id = {t: i for i, t in enumerate(vocab)}

    X, y, pid = [], [], []
    for k, p in enumerate(pieces):
        ids = [tok2id[t] for t in p]
        for s in range(len(ids) - seq_len):
            X.append(ids[s:s + seq_len])
            y.append(ids[s + seq_len])
            pid.append(k)
    X, y, pid = np.array(X, np.int32), np.array(y, np.int32), np.array(pid, np.int32)

    config.VOCAB_PATH.write_text(json.dumps({"tokens": vocab, "seq_len": seq_len}, indent=1))
    np.savez_compressed(config.SEQUENCES_PATH, X=X, y=y, pid=pid)
    print(f"pieces: {len(pieces)} | tokens: {sum(map(len, pieces))} | vocab: {len(vocab)} | "
          f"training windows: {len(X)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq-len", type=int, default=config.SEQ_LEN)
    ap.add_argument("--all-parts", action="store_true", help="merge all MIDI tracks instead of the busiest one")
    ap.add_argument("--min-count", type=int, default=2, help="drop tokens seen fewer times than this")
    a = ap.parse_args()
    main(a.seq_len, a.all_parts, a.min_count)
