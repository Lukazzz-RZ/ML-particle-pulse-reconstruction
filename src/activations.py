import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

##### GRAFICA PARA CADA FUNCIÓN DE ACTIVACIÓN #####

activations = {
    "ReLU": tf.keras.activations.relu,
    "Sigmoid": tf.keras.activations.sigmoid,
    "Tanh": tf.keras.activations.tanh,
    "Softmax": lambda x: tf.keras.activations.softmax(np.hstack([x, -x]), axis=1)[:, 0],
    "Clip (0, 1)": lambda x: tf.clip_by_value(x, 0.0, 1.0)
}

x = np.linspace(-6, 6, 500).reshape(-1, 1)
for name, func in activations.items():
    y = func(x).numpy() if "Softmax" not in name else func(x)
    
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, linewidth=2)
    plt.title(name, fontsize=18)
    plt.xlabel("input", fontsize=14)
    plt.ylabel("output", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
