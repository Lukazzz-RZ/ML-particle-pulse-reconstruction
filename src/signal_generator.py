import numpy as np
import tensorflow as tf
from scipy.stats import uniform
from tensorflow.keras import layers, models
import pandas as pd
import matplotlib.pyplot as plt

# ==== CONFIGURACION ====
MAX_DELTAS = 5
SIGNAL_LENGTH = 512
MUTADOR = 0.05
A_MIN = 1
A_MAX = 25
T_MIN = 100
T_MAX = 300
A_B_MIN = 3 * (1 - MUTADOR)
A_B_MAX = 3 * (1 + MUTADOR)

##### FUNCION DE TRANSFERENCIA VARIABLE #####
def H_variable(t, a, b, C):
    t_scaled = np.clip(t, 0, None) / 20
    return C * np.exp(-a * t_scaled) * t_scaled**b * np.sin(t_scaled)

##### GENERACION DE SEÑALES #####
def Generar_Senal(IDEAL, deltas, snr=3e-2, p=0.5, t=None):
    if t is None:
        t = np.linspace(0, 511, 512)
    clean_signal = np.zeros_like(t)
    for t0, A, C, a, b in deltas:
        clean_signal += A * H_variable(t - t0, a, b, C)
    if IDEAL:
        return clean_signal
    ruido_exp = np.random.normal(0.0, 1.0, len(t))
    ruido_rand = uniform(-0.5, 1.0).rvs(len(t))
    ruido = p * ruido_rand + (1 - p) * ruido_exp
    beta = snr * max(np.max(clean_signal), 1e-6)
    return clean_signal + ruido * beta

def Generar_Datos_Etiquetados(n_samples, min_deltas=1, max_deltas=8, mutador=MUTADOR, ide=True, snr=3e-2, p=0.5):
    signals = []
    t0_centros = []
    etiquetas = []
    coeficientes = []

    for _ in range(n_samples):
        n_deltas = np.random.randint(min_deltas, max_deltas + 1)
        t0_centro = np.random.randint(T_MIN, T_MAX)
        t0_centros.append(t0_centro)
        usados = set()
        deltas = []
        intentos = 0

        while len(deltas) < n_deltas and intentos < 20 * n_deltas:
            t0 = t0_centro + np.random.randint(-n_deltas // 2, n_deltas // 2 + 1)
            if t0 not in usados:
                usados.add(t0)
                A = np.random.uniform(A_MIN, A_MAX)
                if mutador == 0:
                    C, a, b = 1.0, 3.0, 3.0
                else:
                    C = 1.0 # Fijado por el momento
                    a = np.random.uniform(A_B_MIN, A_B_MAX)
                    b = np.random.uniform(A_B_MIN, A_B_MAX)
                deltas.append((t0, A, C, a, b))
            intentos += 1

        deltas.sort(key=lambda x: x[0])
        t = np.linspace(0, 511, 512)
        signal = Generar_Senal(ide, deltas, snr=snr, p=p, t=t)
        signals.append(signal)

        etiqueta = []
        for t0, A, *_ in deltas:
            etiqueta.extend([t0, A])
        while len(etiqueta) < max_deltas * 2:
            etiqueta.extend([-1, -1])
        etiquetas.append(etiqueta)

        coef = []
        for _, _, C, a, b in deltas:
            coef.extend([C, a, b])
        while len(coef) < max_deltas * 3:
            coef.extend([-1, -1, -1])
        coeficientes.append(coef)

    return (np.stack(signals),
            np.array(t0_centros),
            np.stack(etiquetas),
            np.stack(coeficientes))

##### METRICA Y PÉRDIDAS #####
def error_medio_tiempos(y_true, y_pred):
    mask = tf.cast(y_true[..., 1:2] > 0, tf.float32)
    t_true = y_true[..., 0:1]
    t_pred = y_pred[..., 0:1]
    abs_error = tf.abs(t_true - t_pred)
    return tf.reduce_sum(abs_error * mask) / (tf.reduce_sum(mask) + 1e-6)

def mse_sin_padding(y_true, y_pred):
    mask = tf.cast(y_true > 0, tf.float32)
    error = tf.square(y_true - y_pred) * mask
    return tf.reduce_sum(error) / (tf.reduce_sum(mask) + 1e-6)

##### FUNCION DE LECTURA ####
def Leer_Datos(ruta_signals, ruta_etiquetas, ruta_coef):
    signals = pd.read_csv(ruta_signals, header=None).values
    etiquetas = pd.read_csv(ruta_etiquetas, header=None).values
    coeficientes = pd.read_csv(ruta_coef, header=None).values
    return signals, etiquetas, coeficientes
