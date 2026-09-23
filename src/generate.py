"""
Step 5 - GENERATE NEW MUSIC and CONVERT IT TO MIDI (and optionally WAV audio).

The trained LSTM predicts the next note token again and again. `--temperature` controls
how adventurous it is (low = safe/repetitive, high = wild) and `--top-k` restricts
sampling to the k most likely tokens. A tanpura-style Sa-Pa drone is added underneath.
"""
import argparse
import json
import time
from collections import Counter

import numpy as np
from music21 import instrument, note, stream, tempo

from . import config


def sample(probs, temperature, top_k, rng):
    logits = np.log(np.clip(probs, 1e-9, 1.0)) / max(temperature, 1e-3)
    if top_k and top_k < len(logits):
        keep = np.argpartition(logits, -top_k)[-top_k:]
        masked = np.full_like(logits, -np.inf)
        masked[keep] = logits[keep]
        logits = masked
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def generate_tokens(model, vocab, n_tokens, temperature, top_k, rng):
    tokens = vocab["tokens"]
    seq_len = vocab["seq_len"]
    windows = np.load(config.SEQUENCES_PATH)["X"]
    window = list(windows[rng.integers(len(windows))])       # random real seed from the data
    out = []
    for _ in range(n_tokens):
        x = np.array(window[-seq_len:], np.int32)[None]
        probs = model(x, training=False).numpy()[0]
        nxt = sample(probs, temperature, top_k, rng)
        out.append(tokens[nxt])
        window.append(nxt)
    return out


def split(token):
    kind, dur = token.split("_")
    return (None if kind == "R" else int(kind[1:])), float(dur)


def guess_tonic_pc(events):
    """Sa is usually where long phrases rest -> most common pitch class among long notes."""
    longs = [p % 12 for p, d in events if p is not None and d >= 1.5]
    pool = longs or [p % 12 for p, _ in events if p is not None]
    return Counter(pool).most_common(1)[0][0]


def tokens_to_midi(tokens, path, bpm=80, drone=True):
    events = [split(t) for t in tokens]
    melody = stream.Part()
    melody.insert(0, instrument.Sitar())
    melody.insert(0, tempo.MetronomeMark(number=bpm))
    total = 0.0
    for p, d in events:
        melody.append(note.Rest(quarterLength=d) if p is None else note.Note(p, quarterLength=d))
        total += d
    sc = stream.Score()
    sc.insert(0, melody)
    if drone:
        sa = 36 + guess_tonic_pc(events)            # low Sa (C2..B2); Pa = Sa + 7 semitones
        dr = stream.Part()
        dr.insert(0, instrument.Harp())
        t, k = 0.0, 0
        while t < total:
            dr.append(note.Note(sa + (7 if k % 2 else 0), quarterLength=4.0))
            t += 4.0
            k += 1
        sc.insert(0, dr)
    sc.write("midi", fp=str(path))


def main(a):
    from tensorflow import keras
    rng = np.random.default_rng(a.seed)
    vocab = json.loads(config.VOCAB_PATH.read_text())
    model = keras.models.load_model(config.MODEL_PATH)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for i in range(a.count):
        toks = generate_tokens(model, vocab, a.notes, a.temperature, a.top_k, rng)
        mid = config.OUTPUT_DIR / f"generated_{stamp}_{i + 1}.mid"
        tokens_to_midi(toks, mid, bpm=a.bpm, drone=not a.no_drone)
        print(f"saved {mid}")
        if a.wav:
            from .render_audio import render_midi_to_wav
            wav = mid.with_suffix(".wav")
            render_midi_to_wav(mid, wav)
            print(f"saved {wav}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notes", type=int, default=160, help="number of note tokens to generate")
    ap.add_argument("--count", type=int, default=1, help="how many pieces")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--bpm", type=int, default=80)
    ap.add_argument("--no-drone", action="store_true")
    ap.add_argument("--wav", action="store_true", help="also render a .wav (plucked-string synth)")
    ap.add_argument("--seed", type=int, default=None)
    main(ap.parse_args())
