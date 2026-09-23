# Music Generation with AI — Pakistani Classical (Raga) Melodies

An end-to-end deep-learning pipeline that learns raga-style melodic patterns from MIDI and
generates new melodies with an **LSTM**, then exports them as **MIDI** and **WAV audio**.

| Task step | Where it is done |
|---|---|
| 1. Collect MIDI data | `src/collect_data.py`, `src/raga_corpus.py` |
| 2. Preprocess into note sequences (music21) | `src/preprocess.py` |
| 3. Build a deep learning model (LSTM) | `src/model.py` |
| 4. Train the model | `src/train.py` |
| 5. Generate sequences -> MIDI -> audio | `src/generate.py`, `src/render_audio.py` |

## Important note about the dataset (please read)

Transcribed MIDI of real Pakistani classical performances is very scarce and mostly
copyrighted, so this repo **does not ship or scrape any real recordings**. Instead it includes:

* a **synthetic raga corpus generator** (`src/raga_corpus.py`) that writes MIDI following the
  scale and characteristic phrases (pakad) of 10 ragas commonly performed in the Pakistani /
  Hindustani tradition: Yaman, Bhairav, Bhairavi, Bilawal, Kafi, Khamaj, Marwa, Todi,
  Darbari, Malkauns. The rules are simplified and are **not** real performances.
* a **drop-in path for real data**: copy any real MIDI files (your own transcriptions,
  public-domain or licensed material) into `data/midi/real/`, or list direct links in
  `data/sources.txt`. Re-run the pipeline and the model trains on them; no code changes.

To get MIDI from recordings you have the rights to, [Basic Pitch](https://github.com/spotify/basic-pitch)
can transcribe audio to MIDI (`pip install basic-pitch`, then `basic-pitch <out_dir> <audio_file>`).
Monophonic vocal/sitar/bansuri recordings work best.

The included model and samples were trained on the synthetic corpus only.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.collect_data --synthetic     # or put real .mid files in data/midi/real/
python -m src.preprocess
python -m src.train --epochs 40            # add --units 128 on a slow CPU
python -m src.generate --notes 160 --count 3 --wav
```

Or simply `./run_all.sh`. For a point-and-click interface use `streamlit run app.py`. Output goes to `outputs/` (`.mid` + `.wav`). Listen to the ready-made
samples in `outputs/`.

Useful generation options: `--temperature 0.6` (safer) ... `1.2` (more adventurous),
`--top-k 10`, `--bpm 70`, `--no-drone`.

## Web interface

```bash
streamlit run app.py
```

Opens in your browser (http://localhost:8501). Pick length, creativity (temperature), tempo and
drone on the left, press **Generate music**, then listen, view the piano roll and download the
MIDI/WAV. The *Model & training* tab shows the dataset size and training curves.

## How it works

1. **Tokens.** Each note becomes a token such as `P62_0.5` (MIDI pitch 62, half a beat);
   rests are `R_1.0`. Durations are snapped to a small set (0.25 ... 4 beats). Chords keep the
   top note; multi-track files use the busiest track as the melody.
2. **Windows.** A sliding window of 32 tokens predicts the 33rd (40k training windows here).
3. **Model.** `Embedding(128) -> LSTM -> Dropout -> LSTM -> Dense(ReLU) -> Dropout -> Softmax`.
   Adam, cross-entropy, early stopping, LR decay, and a per-piece validation split so the
   model is never validated on pieces it trained on.
4. **Generation.** Seed with a random real window, sample the next token with temperature and
   top-k, repeat. A Sa-Pa drone (tanpura-like) is added on a second track. The tonic is
   estimated from the notes the melody rests on.
5. **Audio.** `render_audio.py` synthesizes a plucked-string (Karplus-Strong) sound with numpy,
   so no SoundFont is needed. For better timbres use FluidSynth (command in the file header).

## Results (synthetic corpus, CPU-only, 25 epochs, LSTM 128 units)

* 300 pieces, ~50k note tokens, vocabulary 175, ~40k windows.
* Validation loss 4.40 -> 2.98, next-token accuracy ~16% (a token is pitch **and** duration and
  durations in the synthetic data are random, so accuracy has a low ceiling; pitch alone is far
  more predictable).
* Musical check: in the three sample pieces every generated note falls inside the note set of a
  single raga (Yaman, Malkauns, Bhairavi), i.e. the model learned raga scale discipline.
  See `models/training_curves.png`.

## Limitations and ideas

* Trained on rule-based data, so it reproduces those rules; it does not capture ornaments
  (meend, gamak, kan), microtones (shruti), or tala/rhythm cycles that MIDI note tokens cannot
  express well.
* Not conditioned on raga; it picks one from the seed. Next step: add a raga embedding.
* Improvements: real transcribed data, pitch-bend/ornament tokens, Transformer or GAN
  variant, human listening evaluation.

## Project layout

```
app.py          Streamlit web UI
src/            pipeline code (run with python -m src.<module>)
data/midi/      real/ and synthetic/ MIDI (processed arrays go to data/processed/)
models/         trained model, training history, curves
outputs/        generated MIDI/WAV samples
```
