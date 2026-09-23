"""Step 3 - BUILD THE DEEP LEARNING MODEL: a stacked LSTM next-token predictor."""
from tensorflow import keras
from tensorflow.keras import layers


def build_model(vocab_size, seq_len, embed_dim=128, units=256, dropout=0.3):
    inp = keras.Input(shape=(seq_len,), dtype="int32")
    x = layers.Embedding(vocab_size, embed_dim)(inp)
    x = layers.LSTM(units, return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(units)(x)
    x = layers.Dense(units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(vocab_size, activation="softmax")(x)
    model = keras.Model(inp, out, name="raga_lstm")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
