"""
Step 4 - TRAIN THE MODEL.

Validation split is done per *piece* (not per window) so the model is never tested on
notes from a piece it trained on.
"""
import argparse
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import config
from .model import build_model


def main(epochs, batch, val_frac, seed, units):
    tf.keras.utils.set_random_seed(seed)
    vocab = json.loads(config.VOCAB_PATH.read_text())
    data = np.load(config.SEQUENCES_PATH)
    X, y, pid = data["X"], data["y"], data["pid"]

    rng = np.random.default_rng(seed)
    pieces = np.unique(pid)
    val_pieces = set(rng.choice(pieces, max(1, int(len(pieces) * val_frac)), replace=False))
    is_val = np.array([p in val_pieces for p in pid])
    tr = tf.data.Dataset.from_tensor_slices((X[~is_val], y[~is_val])).shuffle(20000).batch(batch).prefetch(2)
    va = tf.data.Dataset.from_tensor_slices((X[is_val], y[is_val])).batch(batch).prefetch(2)
    print(f"train windows: {(~is_val).sum()} | validation windows: {is_val.sum()}")

    model = build_model(len(vocab["tokens"]), vocab["seq_len"], units=units)
    model.summary()
    cbs = [
        keras.callbacks.ModelCheckpoint(str(config.MODEL_PATH), monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]
    hist = model.fit(tr, validation_data=va, epochs=epochs, callbacks=cbs, verbose=2)
    (config.MODELS_DIR / "history.json").write_text(json.dumps(hist.history, indent=1))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].plot(hist.history["loss"], label="train"); ax[0].plot(hist.history["val_loss"], label="val")
        ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
        ax[1].plot(hist.history["accuracy"], label="train"); ax[1].plot(hist.history["val_accuracy"], label="val")
        ax[1].set_title("Next-note accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
        fig.tight_layout(); fig.savefig(config.MODELS_DIR / "training_curves.png", dpi=120)
    except ImportError:
        pass
    print(f"Best model saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--units", type=int, default=256, help="LSTM size (use 128 on a slow CPU)")
    a = ap.parse_args()
    main(a.epochs, a.batch, a.val_frac, a.seed, a.units)
