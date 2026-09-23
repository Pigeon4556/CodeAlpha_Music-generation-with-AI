"""
Web UI for the raga music generator.

Run with:   streamlit run app.py
"""
import io
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src import config
from src.generate import generate_tokens, split, tokens_to_midi, guess_tonic_pc
from src.render_audio import render_midi_to_wav

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

st.set_page_config(page_title="Raga Music AI", page_icon="🎵", layout="wide")


@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    from tensorflow import keras
    return keras.models.load_model(config.MODEL_PATH)


def piano_roll(events):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    t = 0.0
    for p, d in events:
        if p is not None:
            ax.barh(p, d, left=t, height=0.8, color="#0f766e")
        t += d
    ax.set_xlabel("time (beats)")
    ax.set_ylabel("MIDI pitch")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


st.title("🎵 Raga Music Generator")
st.caption("LSTM trained on raga-style MIDI (Yaman, Bhairav, Bhairavi, Bilawal, Kafi, Khamaj, "
           "Marwa, Todi, Darbari, Malkauns). Generates new melodies with a Sa-Pa drone.")

ready = config.MODEL_PATH.exists() and config.VOCAB_PATH.exists() and config.SEQUENCES_PATH.exists()
tab_gen, tab_info = st.tabs(["Generate", "Model & training"])

with tab_gen:
    if not ready:
        st.error("No trained model found. Run these commands first, then reload this page:\n\n"
                 "`python -m src.collect_data --synthetic`  \n`python -m src.preprocess`  \n"
                 "`python -m src.train --epochs 40`")
        st.stop()

    with st.sidebar:
        st.header("Settings")
        n_notes = st.slider("Length (notes)", 40, 400, 160, 10)
        temperature = st.slider("Creativity (temperature)", 0.3, 1.5, 0.8, 0.05,
                                help="Low = safe and repetitive, high = adventurous")
        top_k = st.slider("Top-k (most likely notes considered)", 2, 40, 10)
        bpm = st.slider("Tempo (BPM)", 50, 140, 80, 5)
        drone = st.checkbox("Add Sa-Pa drone (tanpura-like)", True)
        seed_txt = st.text_input("Random seed (optional)", "", help="Same seed = same music")
        go = st.button("Generate music", type="primary", use_container_width=True)

    if go:
        try:
            seed = int(seed_txt) if seed_txt.strip() else None
        except ValueError:
            st.warning("Seed must be a whole number; using a random one.")
            seed = None
        rng = np.random.default_rng(seed)
        vocab = json.loads(config.VOCAB_PATH.read_text())
        model = load_model()
        with st.spinner("Composing..."):
            tokens = generate_tokens(model, vocab, n_notes, temperature, top_k, rng)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            mid_path = config.OUTPUT_DIR / f"ui_{stamp}.mid"
            wav_path = mid_path.with_suffix(".wav")
            tokens_to_midi(tokens, mid_path, bpm=bpm, drone=drone)
            render_midi_to_wav(mid_path, wav_path)
        st.session_state["result"] = dict(
            tokens=tokens, mid=mid_path.read_bytes(), wav=wav_path.read_bytes(), stamp=stamp)

    res = st.session_state.get("result")
    if res:
        events = [split(t) for t in res["tokens"]]
        pcs = sorted({p % 12 for p, _ in events if p is not None})
        secs = sum(d for _, d in events) * 60 / bpm
        st.subheader("Result")
        st.audio(res["wav"], format="audio/wav")
        st.pyplot(piano_roll(events))
        c1, c2, c3 = st.columns(3)
        c1.metric("Notes", len(events))
        c2.metric("Approx. length", f"{secs:.0f} s")
        c3.metric("Estimated Sa", NOTE_NAMES[guess_tonic_pc(events)])
        st.write("Notes used: " + ", ".join(NOTE_NAMES[p] for p in pcs))
        d1, d2 = st.columns(2)
        d1.download_button("Download MIDI", res["mid"], f"raga_{res['stamp']}.mid", "audio/midi")
        d2.download_button("Download WAV", res["wav"], f"raga_{res['stamp']}.wav", "audio/wav")
    else:
        st.info("Choose settings on the left and press **Generate music**.")

with tab_info:
    if ready:
        vocab = json.loads(config.VOCAB_PATH.read_text())
        data = np.load(config.SEQUENCES_PATH)
        c1, c2, c3 = st.columns(3)
        c1.metric("Vocabulary (note tokens)", len(vocab["tokens"]))
        c2.metric("Training windows", len(data["X"]))
        c3.metric("Input window", f"{vocab['seq_len']} notes")
    curves = config.MODELS_DIR / "training_curves.png"
    if curves.exists():
        st.image(str(curves), caption="Training / validation loss and next-note accuracy")
    st.markdown(
        "**Architecture:** Embedding -> LSTM -> Dropout -> LSTM -> Dense -> Softmax  \n"
        "**Data:** synthetic raga corpus by default; add real MIDI to `data/midi/real/` and retrain.")
