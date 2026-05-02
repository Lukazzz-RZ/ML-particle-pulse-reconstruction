import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import minimize
from scipy.sparse.linalg import lsqr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
# ------------------------------------------------------------
def H(t_shifted):
    t = np.clip(t_shifted, 0, None) / 20
    return np.exp(-3 * t) * t**3 * np.sin(t)


def taylor_H(t_shifted, a0=3.0, b0=3.0):

    t = np.clip(np.asarray(t_shifted), 0, None) / 20
    eps = 1e-12                       
    t_safe = np.where(t == 0, eps, t) 

    f0    = np.exp(-a0 * t) * t**b0            * np.sin(t)
    df_da = -t**(b0 + 1) * np.exp(-a0 * t)     * np.sin(t)
    df_db =  t**b0      * np.log(t_safe) * np.exp(-a0 * t) * np.sin(t)
    return f0, df_da, df_db


def R2(exp_signal, fit_signal):
    ss_res = np.sum((exp_signal - fit_signal) ** 2)
    ss_tot = np.sum((exp_signal - np.mean(exp_signal)) ** 2)
    return 1 - ss_res / ss_tot, ss_res


def refinar_componentes_taylor(signal, t0s, t, a0=3.0, b0=3.0, alpha=0.0001):
    bloques = []
    for t0 in t0s:
        f0, df_da, df_db = taylor_H(t - t0, a0, b0)
        bloques.extend([f0, df_da, df_db])

    M = np.column_stack(bloques)

    # Verificación de estabilidad
    cond = np.linalg.cond(M)
    print(f"[INFO] Condición de la matriz: cond = {cond:.2e}")

    # Normalización de columnas para estabilidad
    scaler = StandardScaler()
    M_scaled = scaler.fit_transform(M)

    # Ajuste con Ridge regression (regularización L2)
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(M_scaled, signal)
    coef_scaled = reg.coef_

    # Desescalar coeficientes a escala original
    coef = coef_scaled / scaler.scale_

    componentes = []
    for i, t0 in enumerate(t0s):
        A    = coef[3*i]
        daA  = coef[3*i + 1]
        dbA  = coef[3*i + 2]
        componentes.append((t0, A, daA, dbA))

    return componentes

# ----------------------------------------------
def Fit_transfer(N, signal, t=np.arange(512)):

    def H_model(t0, A, daA, dbA):
        f0, df_da, df_db = taylor_H(t - t0, 3, 3)
        return A * f0 + daA * df_da + dbA * df_db

    def loss(params):
        t0, A, daA, dbA = params
        model = H_model(t0, A, daA, dbA)
        _, ss_res = R2(signal, model)
        return ss_res

    def estimar_t0(sig, porcentaje=0.3):
        sig = np.asarray(sig)
        t_max = np.argmax(sig)
        umbral = porcentaje * sig[t_max]
        for idx in range(t_max, -1, -1):
            if sig[idx] <= umbral:
                return idx
        return 0

    # -------- Ajuste inicial global para centrar ventana --------
    x0 = [estimar_t0(signal, 0.15), np.max(signal), 3*np.max(signal), 3*np.max(signal)]
    bounds = [
        (0, 511),          # t0
        (1e-3, 1e2),       # A
        (-1, 1),       # daA
        (-1, 1)        # dbA
    ]

    res = minimize(
        loss,
        x0=x0,
        bounds=bounds,
        method='L-BFGS-B',
        options={'ftol': 1e-15, 'gtol': 1e-15, 'eps': 1e-9, 'maxiter': 1000}
    )

    tcentro_opt = int(round(res.x[0]))

    
    N_real = (N)
    offset = -(N_real // 2)
    t0s = np.arange(tcentro_opt + offset, tcentro_opt + offset + N_real)
    t0s = np.clip(t0s, 0, 511).astype(int)

    
    componentes = refinar_componentes_taylor(signal, t0s, t)


    fit_signal = np.sum([
        A * f0 + daA * df_da + dbA * df_db
        for (t0, A, daA, dbA) in componentes
        for f0, df_da, df_db in [taylor_H(t - t0, 3, 3)]
    ], axis=0)

    r2_final, _ = R2(signal, fit_signal)

    
    param_fit = np.empty(2 * len(componentes))    
    taylor_diffs = np.empty(2 * len(componentes))    

    for i, (t0, A, daA, dbA) in enumerate(componentes):
        param_fit[2*i:2*i+2] = [t0, A]
        if A != 0:
            taylor_diffs[2*i:2*i+2] = [daA / A, dbA / A]
        else:
            taylor_diffs[2*i:2*i+2] = [0.0, 0.0]

    #print(taylor_diffs)
    return param_fit, r2_final, taylor_diffs

# ------------------ ------------------
def procesar_fit(fit, taylor_diffs=None, umbral_amp = 0.05, tolerancia_t0=0.1):
    if len(fit) == 0:
        return (np.array([]), np.array([])) if taylor_diffs is not None else np.array([])

    Amps = fit[1::2]
    umbral_amplitud = umbral_amp*np.max(Amps)
    deltas = [(fit[i], fit[i + 1]) for i in range(0, len(fit), 2)
              if fit[i + 1] >= umbral_amplitud]

    if not deltas:
        return (np.array([]), np.array([])) if taylor_diffs is not None else np.array([])

    deltas.sort(key=lambda x: x[0])
    agrupadas = []
    indices_agrupados = []

    for idx, (t0, A) in enumerate(deltas):
        if not agrupadas:
            agrupadas.append((t0, A))
            indices_agrupados.append([idx])
        else:
            t0_prev, A_prev = agrupadas[-1]
            if abs(t0 - t0_prev) <= tolerancia_t0:
                t0_agg = (t0_prev * A_prev + t0 * A) / (A_prev + A)
                A_agg = A_prev + A
                agrupadas[-1] = (t0_agg, A_agg)
                indices_agrupados[-1].append(idx)
            else:
                agrupadas.append((t0, A))
                indices_agrupados.append([idx])

    resultado = []
    coef_result = []

    for group, idxs in zip(agrupadas, indices_agrupados):
        t0, A_total = group
        resultado.extend([t0, A_total])

        if taylor_diffs is not None:
            suma_ponderada = np.zeros(2)
            peso_total = 0.0
            for i in idxs:
                A_i = fit[2*i + 1]
                if A_i < 1e-6:
                    continue  # evita división por cero o inestabilidades
                rel = np.array(taylor_diffs[2*i:2*i + 2])
                suma_ponderada += A_i * rel
                peso_total += A_i
            if peso_total > 0:
                promedio_ponderado = suma_ponderada / peso_total
            else:
                promedio_ponderado = np.array([0.0, 0.0])
            coef_result.append(promedio_ponderado)

    if taylor_diffs is not None:
        return np.array(resultado), np.array(coef_result).reshape(-1, 2)
    else:
        return np.array(resultado)
