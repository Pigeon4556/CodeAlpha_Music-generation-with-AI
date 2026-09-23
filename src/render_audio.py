"""
Render a MIDI file to a WAV so you can listen to it - no external synth or soundfont needed.

Melody notes use a Karplus-Strong plucked-string model (a rough sitar/sarod-like pluck);
the second track (drone) uses the same model with a very slow decay so it sustains.
For nicer instrument sounds you can instead use FluidSynth + a SoundFont:
    fluidsynth -ni soundfont.sf2 outputs/file.mid -F outputs/file.wav -r 44100
"""
import argparse
from pathlib import Path

import numpy as np
from music21 import converter, tempo
from scipy.io import wavfile

SR = 22050


def pluck(freq, dur, half_life, sr, rng):
    n = int(dur * sr)
    L = max(2, int(round(sr / freq)))
    decay = 0.5 ** (L / (sr * half_life))
    cur = rng.uniform(-1, 1, L)
    cur = 0.5 * (cur + np.roll(cur, 1))                      # soften the attack a little
    out = np.empty(n)
    pos = 0
    while pos < n:
        take = min(L, n - pos)
        out[pos:pos + take] = cur[:take]
        cur = decay * 0.5 * (cur + np.roll(cur, -1))
        pos += L
    fade = min(int(0.05 * sr), n)
    out[-fade:] *= np.linspace(1, 0, fade)
    return out


def render_midi_to_wav(midi_path, wav_path, sr=SR, seed=0):
    rng = np.random.default_rng(seed)
    score = converter.parse(str(midi_path))
    marks = list(score.flatten().getElementsByClass(tempo.MetronomeMark))
    bpm = marks[0].number if marks else 80
    spb = 60.0 / bpm
    parts = list(score.parts) or [score]

    events = []                                              # (start_s, dur_s, midi, is_drone)
    for pi, part in enumerate(parts):
        for el in part.flatten().notes:
            pitches = el.pitches if el.isChord else [el.pitch]
            for p in pitches:
                events.append((float(el.offset) * spb, float(el.quarterLength) * spb, p.midi, pi > 0))
    if not events:
        raise ValueError("no notes in MIDI file")

    total = max(s + d for s, d, _, _ in events) + 2.0
    audio = np.zeros(int(total * sr) + sr)
    for s, d, m, drone in events:
        f = 440.0 * 2 ** ((m - 69) / 12)
        tone = pluck(f, d + (1.2 if not drone else 0.2), 8.0 if drone else 1.0, sr, rng)
        gain = 0.35 if drone else 1.0
        i = int(s * sr)
        audio[i:i + len(tone)] += gain * tone[:len(audio) - i]
    audio /= max(1e-9, np.abs(audio).max()) / 0.9
    wavfile.write(str(wav_path), sr, (audio * 32767).astype(np.int16))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("midi")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = a.out or str(Path(a.midi).with_suffix(".wav"))
    render_midi_to_wav(a.midi, out)
    print(f"saved {out}")
