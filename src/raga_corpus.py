"""
Rule-based raga corpus generator (bootstrap dataset).

WHY THIS EXISTS
---------------
Transcribed MIDI of Pakistani classical (Hindustani-tradition) music is very hard to
find online. So that the whole pipeline is runnable out-of-the-box, this module writes
MIDI files that follow the *grammar* of well-known ragas performed in Pakistan:
scale (swar), characteristic phrases (pakad), stepwise melodic movement (alap-style),
phrase endings that resolve to Sa, and long held notes at cadences.

These files are SYNTHETIC and simplified. They are NOT recordings or transcriptions
of real performances. For a serious model, add real MIDI to `data/midi/real/`
(see README) and re-run the pipeline; nothing else needs to change.

Scale degrees below are indexes into the raga's scale (0 = Sa). Negative indexes go
below Sa (lower octave), indexes >= len(scale) go into the upper octave.
"""
import argparse
import random
from pathlib import Path

from music21 import instrument, note, stream, tempo

from . import config

RAGAS = {
    # name: semitone offsets from Sa, and simplified characteristic phrases
    "yaman":    dict(scale=[0, 2, 4, 6, 7, 9, 11], vadi=2,
                     pakad=[[-1, 1, 2], [2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1, 0], [-1, 1, 2, 3, 4, 5, 6, 7]]),
    "bhairav":  dict(scale=[0, 1, 4, 5, 7, 8, 11], vadi=5,
                     pakad=[[0, 2, 3, 4], [2, 3, 5, 4], [5, 4, 2, 3, 1, 0], [4, 5, 4, 2, 3, 1, 0]]),
    "bhairavi": dict(scale=[0, 1, 3, 5, 7, 8, 10], vadi=5,
                     pakad=[[0, 1, 2, 3], [3, 2, 1, 0], [2, 3, 4, 5], [5, 4, 3, 2, 1, 0], [-1, 0, 1, 0]]),
    "bilawal":  dict(scale=[0, 2, 4, 5, 7, 9, 11], vadi=5,
                     pakad=[[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [2, 4, 5, 6, 7], [7, 6, 5, 4, 2, 0]]),
    "kafi":     dict(scale=[0, 2, 3, 5, 7, 9, 10], vadi=4,
                     pakad=[[0, 1, 2, 3], [3, 4, 5, 4, 3], [4, 3, 2, 1, 0], [-1, 0, 1, 2, 1, 0]]),
    "khamaj":   dict(scale=[0, 2, 4, 5, 7, 9, 10], vadi=2,
                     pakad=[[4, 5, 6, 7, 6, 5, 4], [2, 3, 4, 3, 2], [6, 5, 4, 3, 2, 1, 0], [-1, 0, 2, 3]]),
    "marwa":    dict(scale=[0, 1, 4, 6, 9, 11], vadi=1,
                     pakad=[[4, 1, 2, 3, 4], [5, 4, 3, 2, 1, 0], [-1, 1, 2, 3], [3, 2, 1, 0]]),
    "todi":     dict(scale=[0, 1, 3, 6, 7, 8, 11], vadi=5,
                     pakad=[[0, 1, 2], [2, 3, 4, 5], [5, 4, 2, 1, 0], [1, 2, 1, 0]]),
    "darbari":  dict(scale=[0, 2, 3, 5, 7, 8, 10], vadi=4,
                     pakad=[[2, 1, 0, -1, 0], [3, 4, 3, 2, 1, 0], [4, 5, 4, 3, 2], [0, 2, 3, 4]]),
    "malkauns": dict(scale=[0, 3, 5, 8, 10], vadi=2,
                     pakad=[[0, 1, 2], [2, 3, 4, 5], [4, 3, 2, 1, 0], [-1, 0, 1, 0]]),
}

DUR_CHOICES = [0.25, 0.5, 1.0, 1.5, 2.0]
DUR_WEIGHTS = [0.15, 0.35, 0.30, 0.10, 0.10]


def degree_to_midi(scale, tonic, idx):
    octave, d = divmod(idx, len(scale))
    return tonic + scale[d] + 12 * octave


def compose(raga, tonic, n_notes, rng):
    """Return a list of (midi_pitch | None, duration) events."""
    scale = raga["scale"]
    n = len(scale)
    lo, hi = -n // 2, n + n // 2          # allowed melodic range (approx. low N .. high P)
    idx, events, since_cadence = 0, [], 0

    def add(i, long_end=False):
        d = rng.choices(DUR_CHOICES, DUR_WEIGHTS)[0]
        if long_end:
            d = rng.choice([2.0, 3.0, 4.0])
        events.append((degree_to_midi(scale, tonic, i), d))

    while len(events) < n_notes:
        r = rng.random()
        if since_cadence >= rng.randint(14, 24):          # cadence: descend to Sa and rest on it
            while idx > 0:
                idx -= 1
                add(idx, long_end=(idx == 0))
            while idx < 0:
                idx += 1
                add(idx, long_end=(idx == 0))
            if not events or events[-1][0] != tonic:
                add(0, long_end=True)
            since_cadence = 0
        elif r < 0.35:                                    # characteristic phrase (pakad)
            phrase = rng.choice(raga["pakad"])
            for k, i in enumerate(phrase):
                add(i, long_end=(k == len(phrase) - 1 and rng.random() < 0.4))
            idx = phrase[-1]
            since_cadence += len(phrase)
        else:                                             # stepwise alap-style movement
            for _ in range(rng.randint(3, 6)):
                pull = 1 if idx < raga["vadi"] else -1    # gently attracted to the vadi note
                step = rng.choices([-2, -1, 0, 1, 2], [0.10, 0.35, 0.10, 0.35, 0.10])[0]
                if rng.random() < 0.15:
                    step = pull
                idx = max(lo, min(hi, idx + step))
                add(idx)
                since_cadence += 1
        if rng.random() < 0.04:
            events.append((None, 0.5))                    # small pause
    return events[:n_notes]


def write_midi(events, path, bpm=80):
    part = stream.Part()
    part.insert(0, instrument.Sitar())
    part.insert(0, tempo.MetronomeMark(number=bpm))
    for pitch, dur in events:
        part.append(note.Rest(quarterLength=dur) if pitch is None
                    else note.Note(pitch, quarterLength=dur))
    sc = stream.Score()
    sc.insert(0, part)
    sc.write("midi", fp=str(path))


def build_corpus(out_dir, pieces_per_raga=30, notes_per_piece=140, tonic=60, seed=42):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    count = 0
    for name, raga in RAGAS.items():
        for k in range(pieces_per_raga):
            events = compose(raga, tonic, notes_per_piece, rng)
            write_midi(events, out_dir / f"{name}_{k:03d}.mid", bpm=rng.choice([70, 80, 90]))
            count += 1
    print(f"Wrote {count} synthetic raga MIDI files ({len(RAGAS)} ragas) to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(config.MIDI_DIR / "synthetic"))
    ap.add_argument("--pieces-per-raga", type=int, default=30)
    ap.add_argument("--notes-per-piece", type=int, default=140)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    build_corpus(a.out, a.pieces_per_raga, a.notes_per_piece, seed=a.seed)
