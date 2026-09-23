"""Central configuration shared by all scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIDI_DIR = ROOT / "data" / "midi"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

# Length of the input window (in tokens) the LSTM sees before predicting the next one.
SEQ_LEN = 32

# Note lengths (in quarter notes) that every duration is snapped to.
DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

MODEL_PATH = MODELS_DIR / "raga_lstm.keras"
VOCAB_PATH = PROCESSED_DIR / "vocab.json"
SEQUENCES_PATH = PROCESSED_DIR / "sequences.npz"

for _d in (MIDI_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
