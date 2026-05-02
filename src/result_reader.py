import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from matplotlib.lines import Line2D
import ast

# ------------------ FUNCIONES AUXILIARES ------------------

def reparar_array_string(s):

    if isinstance(s, str) and '[' in s and ' ' in s and ',' not in s:
        return '[' + ','.join(s.strip('[]').split()) + ']'
    return s

def string_a_lista(txt, col_name, row_idx):

    if pd.isna(txt) or str(txt).strip() == "":
        return []
    try:
        val = ast.literal_eval(txt)
        if isinstance(val, list) and all(isinstance(x, tuple) for x in val):
            # [(t0, A), ...] -> [t0, A, t0, A, ...]
            return [x for pair in val for x in pair]
        elif isinstance(val, list):
            return val
        else:
            return []
    except Exception as e:
        print(f"[WARN] Fila {row_idx} col '{col_name}': {e}. Se devuelve [].")
        return []

def cargar_resultados(path_csv):
    df = pd.read_csv(path_csv)

    for col in ('fit_resultado', 'etiqueta'):
        if col in df.columns:
            df[col] = [
                string_a_lista(reparar_array_string(val), col, idx)
                for idx, val in enumerate(df[col].values)
            ]

    if 'signal' in df.columns:
        df['signal'] = [
            ast.literal_eval(reparar_array_string(val)) if isinstance(val, str) else val
            for val in df['signal']
        ]

    return df

def plot_errores(df):

    if 'norma_deltas' not in df.columns or 'norma_amplis' not in df.columns:
        print("No hay columnas de errores para plotear.")
        return

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(df['norma_deltas'], marker='o')
    axs[0].set_title("Error en tiempos (|Δt|)")
    axs[0].set_xlabel("Índice de señal")
    axs[0].set_ylabel("Suma de |Δt|")
    axs[0].grid(True)

    axs[1].plot(df['norma_amplis'], marker='o', color='orange')
    axs[1].set_title("Error en amplitudes (|ΔA|)")
    axs[1].set_xlabel("Índice de señal")
    axs[1].set_ylabel("Suma de |ΔA|")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show(block=False)

def visualizar_indice(df, idx):
    if idx < 0 or idx >= len(df):
        print("Índice fuera de rango.")
        return

    fila     = df.iloc[idx]
    signal   = np.array(fila['signal'])
    etiqueta = fila['etiqueta']
    fit      = fila['fit_resultado']

    print(f"\n>> Señal {idx}")
    print(f"Etiqueta  : {etiqueta}")
    print(f"Fit       : {fit}")
    print(f"|Δt| total: {fila['norma_deltas']:.2f}, |ΔA| total: {fila['norma_amplis']:.2f}")

    plt.figure(figsize=(10, 4))
    plt.plot(signal, color="black")                  

    for t0 in etiqueta[::2]:
        plt.axvline(t0, color='green', linestyle='-', alpha=0.7)

    for t0 in fit[::2]:
        plt.axvline(t0, color='red', linestyle='-', alpha=0.5)

    custom_lines = [
        Line2D([0], [0], color='black', linewidth=2),
        Line2D([0], [0], color='green', linestyle='-', linewidth=2),
        Line2D([0], [0], color='red',   linestyle='-',  linewidth=2)
    ]
    plt.legend(custom_lines, ['Señal', 'Deltas reales', 'Deltas ajustados'])
    plt.title(f"Señal {idx}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------ EJECUCIÓN ------------------

if __name__ == "__main__":

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title="Selecciona archivo de resultados (.csv)", filetypes=[("CSV files", "*.csv")])
    
    if not path:
        print("No se seleccionó ningún archivo.")
        exit()

    df = cargar_resultados(path)
    plot_errores(df)

    while True:
        try:
            idx = int(input("\nIntroduce un índice para visualizar (-1 para salir): "))
            if idx == -1:
                break
            visualizar_indice(df, idx)
        except ValueError:
            print("Entrada no válida.")
