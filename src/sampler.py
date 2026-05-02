import os
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

###### CONFIGURACION ######
MAX_DELTAS = 10
SIGNAL_LENGTH = 512
MUTADOR = 0.05
A_MIN, A_MAX = 0.0, 25.0
T_MIN, T_MAX = 100, 300
A_B_MIN = 0.9 * 3 * (1 - MUTADOR)
A_B_MAX = 1.1 * 3 * (1 + MUTADOR)

MODEL_PATH = "models/best_model.keras"
N_TEST = 10  # SEÑALES A PROCESAR
OUT_DIR = "results"

###### FUNCIONES DE SEÑAL ######

def H_variable(t: np.ndarray, a: float, b: float, C: float) -> np.ndarray:
    """Respuesta impulso parametrizada."""
    t_scaled = np.clip(t, 0.0, None) / 20.0
    return C * np.exp(-a * t_scaled) * t_scaled**b * np.sin(t_scaled)

# GENERADOR SEÑAL (ADAPTADO)
def Generar_Senal(
    ideal: bool,
    deltas: List[Tuple[float, float, float, float, float]],
    snr: float = 3e-2,
    p: float = 0.5,
    t: np.ndarray | None = None,
    ) -> np.ndarray:

    if t is None:
        t = np.linspace(0, SIGNAL_LENGTH - 1, SIGNAL_LENGTH)
    sig = sum(A * H_variable(t - t0, a, b, C) for t0, A, C, a, b in deltas)
    if ideal:
        return sig
    ruido = p * (np.random.rand(len(t)) - 0.5) + (1 - p) * np.random.randn(len(t))
    return sig + ruido * snr * max(np.max(sig), 1e-6)


def Generar_Datos_Etiquetados(n: int, min_d: int = 1, max_d: int = 8, mut: float = MUTADOR):
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
        np.stack(sigs).astype(np.float32),
        np.array(t0s, dtype=np.float32),
        np.stack(y).astype(np.float32),
        np.stack(coef).astype(np.float32),
        np.array(a_tot, dtype=np.float32),
    )

###### CAPAS Y PÉRDIDAS ######

class Clip(layers.Layer):
    def __init__(self, vmin, vmax, **kw):
        super().__init__(**kw)
        self.vmin = vmin
        self.vmax = vmax

    def call(self, x):
        return tf.clip_by_value(x, self.vmin, self.vmax)

def mse_sin_padding(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

###### FUNCIONES AUXILIARES ######

def funcion_objetivo(params: np.ndarray, t: np.ndarray, señal_obs: np.ndarray, max_d: int):
    señal_pred = np.zeros_like(señal_obs)
    for i in range(max_d):
        base = 5 * i
        t0, A, C, a, b = params[base : base + 5]
        señal_pred += A * H_variable(t - t0, a, b, C)
        if A > A_MAX:
            señal_pred += 1e3 * (A - A_MAX)
    return señal_pred - señal_obs


def optimizar_subconjunto(
    params: np.ndarray,
    idx_var: List[int],
    t: np.ndarray,
    señal_obs: np.ndarray,
    max_d: int,
    lb: np.ndarray,
    ub: np.ndarray,
):
    idx_var = np.asarray(idx_var, int)

    def fun(subset_vals):
        tmp = params.copy()
        tmp[idx_var] = subset_vals
        return funcion_objetivo(tmp, t, señal_obs, max_d)

    res = least_squares(
        fun,
        params[idx_var],
        method="trf",
        bounds=(lb, ub),
        ftol=1e-14,
        xtol=1e-20,
        gtol=1e-20,
        max_nfev=5000,
    )
    params[idx_var] = res.x
    return params

###### REFINADO CUASIESTÁTICO ######

def refinar_tiempos_y_amplitudes(
    params: np.ndarray,
    t: np.ndarray,
    señal_obs: np.ndarray,
    max_d: int,
    n_iter: int = 5,
    ) -> np.ndarray:

    """Ajusta A y t₀ (enteros) de cada pulso durante `n_iter` pasadas."""

    A_base = params[[5 * i + 1 for i in range(max_d)]].copy()
    for _ in range(n_iter):
        mejora_total = 0.0
        for i in range(max_d):
            base = 5 * i
            t0_i = int(round(params[base]))
            A_i = params[base + 1]
            C_i = params[base + 2]
            a_i = params[base + 3]
            b_i = params[base + 4]

            # Residual sin el pulso i
            resid = señal_obs.copy()
            for j in range(max_d):
                if j == i:
                    continue
                b2 = 5 * j
                t0_j, A_j, C_j, a_j, b_j = params[b2 : b2 + 5]
                resid -= A_j * H_variable(t - t0_j, a_j, b_j, C_j)

            # Barrido t₀ ±1 bin (enteros)
            best_t0, best_A, best_err = t0_i, A_i, np.inf
            for t0_int in range(max(T_MIN, t0_i - 1), min(T_MAX, t0_i + 1) + 1):
                phi = H_variable(t - t0_int, a_i, b_i, C_i)
                A_opt = np.dot(phi, resid) / (np.dot(phi, phi) + 1e-12)
                A_opt = np.clip(A_opt, 0.5 * A_base[i], 1.5 * A_base[i])
                err = np.linalg.norm(resid - A_opt * phi)
                if err < best_err:
                    best_err, best_t0, best_A = err, t0_int, A_opt
            mejora_total += abs(params[base] - best_t0) + abs(params[base + 1] - best_A)
            params[base] = best_t0
            params[base + 1] = best_A
        if mejora_total < 1e-3:
            break
    return params

def ajustar_prediccion_completa(
    t: np.ndarray,
    señal_obs: np.ndarray,
    params_iniciales: np.ndarray,
    max_d: int,
    ciclos: int = 5,
    tol: float = 1e-10,
    ) -> np.ndarray:
    
    """Repite el ajuste completo (a/b → A → t₀ → refino) hasta `ciclos` veces
    o hasta que la mejora de la norma L2 sea < `tol`."""

    params = params_iniciales.copy()
    prev_norm = np.linalg.norm(funcion_objetivo(params, t, señal_obs, max_d))

    for _ in range(ciclos):
        # --- Fase 1: ajustar a y b ---
        idx_ab = [5 * i + 3 for i in range(max_d)] + [5 * i + 4 for i in range(max_d)]
        params = optimizar_subconjunto(
            params,
            idx_ab,
            t,
            señal_obs,
            max_d,
            np.full(len(idx_ab), A_B_MIN),
            np.full(len(idx_ab), A_B_MAX),
        )

        # --- Fase 2: ajustar amplitudes A (±50 %) ---
        idx_A = [5 * i + 1 for i in range(max_d)]
        A0 = params[idx_A].copy()
        params = optimizar_subconjunto(
            params,
            idx_A,
            t,
            señal_obs,
            max_d,
            0.75 * A0,
            1.25 * A0,
        )

        # --- Fase 3: ajustar tiempos t₀ (±1 bin) ---
        idx_t0 = [5 * i for i in range(max_d)]
        t0_orig = params[idx_t0].copy()
        params = optimizar_subconjunto(
            params,
            idx_t0,
            t,
            señal_obs,
            max_d,
            np.maximum(T_MIN, t0_orig - 1),
            np.minimum(T_MAX, t0_orig + 1),
        )

        # --- Ejecución de refinado ---
        params = refinar_tiempos_y_amplitudes(params, t, señal_obs, max_d)
        curr_norm = np.linalg.norm(funcion_objetivo(params, t, señal_obs, max_d))
        if prev_norm - curr_norm < tol:
            break
        prev_norm = curr_norm

    return params

###### OTRAS FUNCIONES ######

def reconstruir_deltas(vec: np.ndarray):
    """Convierte vector [t0,A,t0,A,...] en lista de tuplas excluyendo padding."""
    return [(vec[i], vec[i + 1]) for i in range(0, len(vec), 2) if vec[i + 1] > 0]


def agrupar_pred(salida: np.ndarray, A_min: float = 1):
    """Agrupa predicciones que caen en el mismo bin entero y suma amplitudes."""
    d = defaultdict(list)
    for t, A, *_ in salida:
        if A < A_min:
            continue
        d[int(round(t))].append(A)
    out = [(t, sum(A_list)) for t, A_list in d.items()]
    return sorted(out, key=lambda x: x[0])

if __name__ == "__main__":
    Path(OUT_DIR).mkdir(exist_ok=True)

    print("Generando datos de prueba …")
    sigs, t0s, y_lab, coefs, atot = Generar_Datos_Etiquetados(N_TEST, max_d=MAX_DELTAS)

    print("Cargando modelo …")
    model = load_model(
        MODEL_PATH,
        custom_objects={"Clip": Clip, "mse_sin_padding": mse_sin_padding},
        compile=False,
    )

    tiempo_vector = np.linspace(0, SIGNAL_LENGTH - 1, SIGNAL_LENGTH)
    resumen = []

    for i in range(N_TEST):
        sig  = sigs[i].reshape(1, -1, 1)
        t0   = np.array(t0s[i]).reshape(1, 1)
        Atot = np.array(atot[i]).reshape(1, 1)

        pred_raw = model.predict([sig, t0, Atot], verbose=0)[0]

        ###### Ajuste completo iterado ######
        params_ajustados = ajustar_prediccion_completa(
            tiempo_vector, sigs[i], pred_raw.flatten(), MAX_DELTAS, ciclos=15
        )
        pred_ajustada = params_ajustados.reshape(MAX_DELTAS, 5)
        pred_proc_ajustada = agrupar_pred(pred_ajustada)

        ###### VISUALIZADO #####
        reales = reconstruir_deltas(y_lab[i])

        coefs_real = []
        coefs_pred = []
        for j in range(MAX_DELTAS):
            # Coeficientes reales
            c_r = coefs[i][3 * j + 0]
            a_r = coefs[i][3 * j + 1]
            b_r = coefs[i][3 * j + 2]
            coefs_real.append((c_r, a_r, b_r))

            # Coeficientes predichos (C fijo a 1)
            c_p = pred_ajustada[j, 2]
            a_p = pred_ajustada[j, 3]
            b_p = pred_ajustada[j, 4]
            coefs_pred.append((c_p, a_p, b_p))

        # Señal reconstruida y residual
        señal_pred = np.zeros_like(tiempo_vector)
        for t0_p, A_p, C_p, a_p, b_p in pred_ajustada:
            señal_pred += A_p * H_variable(tiempo_vector - t0_p, a_p, b_p, C_p)
        residual_norm = np.linalg.norm(sigs[i] - señal_pred)

        # Impresión en consola 
        print(f"\n[{i+1}/{N_TEST}] ----------------------------")
        print("Reales   t,A :", reales)
        print("Ajustados t,A:", pred_proc_ajustada if pred_proc_ajustada else "(vacío)")
        print("Reales   C,a,b:", [(round(c,2), round(a,2), round(b,2)) for c,a,b in coefs_real])
        print("Ajustados C,a,b:", [(round(c,2), round(a,2), round(b,2)) for c,a,b in coefs_pred])
        print(f"Norma residual: {residual_norm:.6f}")

        # Resumen para CSV 
        resumen.append({
            "idx": i,
            "t_real":       [t for t,_ in reales],
            "A_real":       [A for _,A in reales],
            "t_pred":       [t for t,_ in pred_proc_ajustada],
            "A_pred":       [A for _,A in pred_proc_ajustada],
            "C_real":       [round(c,4) for c,_,_ in coefs_real],
            "a_real":       [round(a,4) for _,a,_ in coefs_real],
            "b_real":       [round(b,4) for _,_,b in coefs_real],
            "C_pred":       [round(c,4) for c,_,_ in coefs_pred],
            "a_pred":       [round(a,4) for _,a,_ in coefs_pred],
            "b_pred":       [round(b,4) for _,_,b in coefs_pred],
            "residual_norm": residual_norm,
        })

    # PATH
    csv_path = Path(OUT_DIR) / f"comparacion_ajuste_iterado_{N_TEST}.csv"
    pd.DataFrame(resumen).to_csv(csv_path, index=False)
    print("\nCSV guardado en:", csv_path)


##### GRAFICADO SECUENCIAL #####

plt.rcParams["font.size"] = 14  
a_real, a_pred, b_real, b_pred = [], [], [], []

for r in resumen:
    for ar, ap, br, bp in zip(r["a_real"], r["a_pred"],
                              r["b_real"], r["b_pred"]):
        if ar > 0 and br > 0:      # descarta padding (-1)
            a_real.append(ar);  a_pred.append(ap)
            b_real.append(br);  b_pred.append(bp)

idx_a = np.arange(len(a_real))
idx_b = np.arange(len(b_real))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

# ---- coeficiente a
ax1.bar(idx_a, a_real, color="tab:blue", alpha=0.4, width=0.7, label="a real")

ax1.plot(idx_a, a_pred, "x", color="tab:red", markersize=7, label="a predicho")
ax1.set_title("Coeficiente a – real (barra) vs. predicho (x)")
ax1.set_xlabel("Índice de pulso válido")
ax1.set_ylabel("Valor de a")
ax1.legend(loc="lower left")

ax1.grid(True)

# Etiquetas en la base de las barras
for i, val in zip(idx_b, b_real):
    ax2.text(i - 0.3, 0.05, f"{val:.2f}", ha="left", va="bottom", fontsize=11, color="tab:green")

ax2.set_title("Coeficiente b – real (barra) vs. predicho (x)")
ax2.set_xlabel("Índice de pulso válido")
ax2.set_ylabel("Valor de b")
ax2.legend(loc="lower left")
ax2.grid(True)

# ---- coeficiente b
ax2.bar(idx_b, b_real, color="tab:green", alpha=0.4, width=0.7, label="b real")
ax2.plot(idx_b, b_pred, "x", color="tab:orange", markersize=7, label="b predicho")
ax2.set_title("Coeficiente b – real (barra) vs. predicho (x)")
ax2.set_xlabel("Índice de pulso válido")
ax2.set_ylabel("Valor de b")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------

a_real, a_pred, b_real, b_pred = [], [], [], []

plt.rcParams["font.size"] = 16
plt.rcParams["axes.grid"] = True

err_t, err_A = [], []
for r in resumen:
    # tiempos
    t_r, t_p = np.array(r["t_real"]), np.array(r["t_pred"])
    n = min(len(t_r), len(t_p))
    err_t.append(np.sum(np.abs(t_r[:n] - t_p[:n])))

    # amplitudes
    A_r, A_p = np.array(r["A_real"]), np.array(r["A_pred"])
    nA = min(len(A_r), len(A_p))
    err_A.append(np.sum(np.abs(A_r[:nA] - A_p[:nA])))

fig, ax = plt.subplots(1, 2, figsize=(13, 4), sharex=True)

ax[0].plot(err_t, marker="o", lw=1.5)
ax[0].set_title("Σ |Δt| por señal")
ax[0].set_xlabel("Índice de señal")
ax[0].set_ylabel("Error total en muestras")

ax[1].plot(err_A, marker="x", color="tab:orange", lw=1.5)
ax[1].set_title("Σ |ΔA| por señal")
ax[1].set_xlabel("Índice de señal")
ax[1].set_ylabel("Error total en amplitud")

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------