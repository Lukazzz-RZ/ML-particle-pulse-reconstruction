import numpy as np
import tensorflow as tf
from scipy.stats import uniform
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision as mp
from tensorflow.keras.optimizers import Adam
import os
import argparse

# ---------------- CONFIG ----------------
MAX_DELTAS    = 10
SIGNAL_LENGTH = 512
MUTADOR       = 0.05
A_MIN, A_MAX  = 0, 25
T_MIN, T_MAX  = 100, 300
A_B_MIN       = 0.9*3 * (1 - MUTADOR)
A_B_MAX       = 1.1*3 * (1 + MUTADOR)

# --------------- SEÑALES ----------------

def H_variable(t, a, b, C):
    t_scaled = np.clip(t, 0, None) / 20
    return C * np.exp(-a * t_scaled) * t_scaled ** b * np.sin(t_scaled)

def Generar_Senal(IDEAL, deltas, snr=3e-2, p=0.5, t=None):
    if t is None:
        t = np.linspace(0, SIGNAL_LENGTH - 1, SIGNAL_LENGTH)
    sig = sum(A * H_variable(t - t0, a, b, C) for t0, A, C, a, b in deltas)
    if IDEAL:
        return sig
    ruido = p * uniform(-.5, 1.0).rvs(len(t)) + (1 - p) * np.random.randn(len(t))
    return sig + ruido * snr * max(np.max(sig), 1e-6)

def Generar_Datos_Etiquetados(n, min_d=1, max_d=8, mut=MUTADOR):
    sigs, t0s, y, coef, a_tot = [], [], [], [], []
    for _ in range(n):
        nd = np.random.randint(min_d, max_d + 1)
        t0c = np.random.randint(T_MIN, T_MAX)
        t0s.append(t0c)
        idx, usados = [], set()
        while len(idx) < nd:
            t0 = t0c + np.random.randint(-nd // 2, nd // 2 + 1)
            if t0 in usados:
                continue
            usados.add(t0)
            A = np.random.uniform(A_MIN + 1, A_MAX)
            C = 1.0
            a = np.random.uniform(A_B_MIN, A_B_MAX)
            b = np.random.uniform(A_B_MIN, A_B_MAX)
            idx.append((t0, A, C, a, b))
        idx.sort(key=lambda x: x[0])
        sigs.append(Generar_Senal(True, idx))
        a_tot.append(sum(p[1] for p in idx))
        yy, cc = [], []
        for t0, A, C, a, b in idx:
            yy.extend([t0, A])
            cc.extend([C, a, b])
        yy += [-1, -1] * (max_d - len(idx))
        cc += [-1, -1, -1] * (max_d - len(idx))
        y.append(yy)
        coef.append(cc)
    return (
        np.stack(sigs),
        np.array(t0s),
        np.stack(y),
        np.stack(coef),
        np.array(a_tot),
    )

# ------- CUSTOM CLIP LAYER ---------

class Clip(tf.keras.layers.Layer):
    def __init__(self, vmin, vmax, **kw):
        super().__init__(**kw)
        self.vmin = vmin
        self.vmax = vmax

    def call(self, x):
        return tf.clip_by_value(x, self.vmin, self.vmax)

# ------------- MODEL --------------

def build_transformer_model(L, max_d):
    inp_senal = tf.keras.Input(shape=(L, 1), name="signal")
    inp_t0 = tf.keras.Input(shape=(1,), name="t0_central")
    inp_Atot = tf.keras.Input(shape=(1,), name="A_total")

    # Convolución inicial más potente
    x = layers.Conv1D(256, 5, padding="same", activation="relu")(inp_senal)
    x = layers.LayerNormalization()(x)

    # Conv1D dilatada para mayor campo receptivo
    x = layers.Conv1D(256, 3, padding="same", dilation_rate=2, activation="relu")(x)
    x = layers.LayerNormalization()(x)

    # Bloque de atención multi-cabeza
    attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)

    # Feed-forward residual
    ff = layers.Dense(256, activation="relu")(x)
    ff = layers.Dense(256)(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)

    # Pooling global para vectorizar
    x = layers.GlobalAveragePooling1D()(x)

    # Concatenar inputs auxiliares
    x = layers.Concatenate()([x, inp_t0, inp_Atot])

    # Capas densas con dropout
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Expandir para cada delta
    x_rep = layers.RepeatVector(max_d)(x)

    # LSTM bidireccional más profundo
    x_rep = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x_rep)
    x_rep = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x_rep)

    # Salidas

    # Tiempos t0
    t_raw = layers.Dense(1)(x_rep)
    t_out = Clip(T_MIN, T_MAX)(layers.Add()([t_raw, layers.RepeatVector(max_d)(inp_t0)]))

    # Amplitudes A
    A_logits = layers.Dense(1)(x_rep)
    A_weights = layers.Softmax(axis=1)(A_logits)
    A_out = layers.Multiply()([A_weights, layers.RepeatVector(max_d)(inp_Atot)])

    # Coeficiente C
    C_raw = layers.Dense(1)(x_rep)
    C_out = layers.Rescaling(scale=0.4, offset=0.8)(layers.Activation("sigmoid")(C_raw))

    # Coeficiente a
    a_raw = layers.Dense(1)(x_rep)
    a_out = layers.Rescaling(scale=A_B_MAX - A_B_MIN, offset=A_B_MIN)(layers.Activation("sigmoid")(a_raw))

    # Coeficiente b
    b_raw = layers.Dense(1)(x_rep)
    b_out = layers.Rescaling(scale=A_B_MAX - A_B_MIN, offset=A_B_MIN)(layers.Activation("sigmoid")(b_raw))

    out = layers.Concatenate(name="preds")([t_out, A_out, C_out, a_out, b_out])

    return tf.keras.Model([inp_senal, inp_t0, inp_Atot], out)

# ------------- MÉTRICAS Y PÉRDIDAS ------------

def error_medio_tiempos(y_true, y_pred):
    mask = tf.cast(y_true[..., 1:2] > 0, tf.float32)
    abs_error = tf.abs(y_true[..., 0:1] - y_pred[..., 0:1])
    return tf.reduce_sum(abs_error * mask) / (tf.reduce_sum(mask) + 1e-6)

def mse_sin_padding(y_true, y_pred):
    PAD_VAL = -1.0
    pad_mask = tf.cast(tf.not_equal(y_true[..., 1:2], PAD_VAL), tf.float32)
    mse_hits = tf.reduce_sum(tf.square(y_true - y_pred) * pad_mask) / (
        tf.reduce_sum(pad_mask) + 1e-6
    )
    A_pred = y_pred[..., 1:2]
    fp_mask = 1.0 - pad_mask
    mse_fp = tf.reduce_sum(tf.square(A_pred) * fp_mask) / (tf.reduce_sum(fp_mask) + 1e-6)
    alpha = 10.0
    return mse_hits + alpha * mse_fp

# ------------- TRAIN / RESUME --------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or resume Transformer signal model")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing checkpoint")
    args = parser.parse_args()

    mp.set_global_policy("float32")
    os.makedirs("models", exist_ok=True)

    # Generar datos
    signals, t0s, ylab, coefs, atot = Generar_Datos_Etiquetados(
        7500 * MAX_DELTAS, max_d=MAX_DELTAS
    )

    x1 = signals[..., None].astype(np.float32)
    x2 = t0s.astype(np.float32).reshape(-1, 1)
    x3 = atot.astype(np.float32).reshape(-1, 1)

    y = np.concatenate(
        [ylab.reshape(-1, MAX_DELTAS, 2), coefs.reshape(-1, MAX_DELTAS, 3)], axis=-1
    ).astype(np.float32)

    # Decide si crear modelo nuevo o cargar existente
    if args.resume and os.path.exists("models/best_model.keras"):
        print("[INFO] Cargando modelo existente...")
        model = tf.keras.models.load_model(
            "models/best_model.keras",
            custom_objects={
                "Clip": Clip,
                "mse_sin_padding": mse_sin_padding,
                "error_medio_tiempos": error_medio_tiempos,
            },
        )
    else:
        print("[INFO] Creando modelo nuevo...")
        model = build_transformer_model(SIGNAL_LENGTH, MAX_DELTAS)

    opt = Adam(1e-4, clipnorm=1.0)
    model.compile(optimizer=opt, loss=mse_sin_padding, metrics=[error_medio_tiempos])

    # Callbacks
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "models/best_model.keras", save_best_only=True, verbose=1
    )

    # Entrenamiento
    model.fit(
        [x1, x2, x3],
        y,
        epochs=100,
        batch_size=64,
        validation_split=0.1,
        callbacks=[reduce_lr, early_stop, checkpoint],
    )

    # Guardar modelo final
    model.save("models/transformer_10_25_5perc.keras")
