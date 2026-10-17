"""Short integration check for the saved Micromegas reconstruction model.

Run from the repository root after installing TensorFlow and the numerical dependencies.
This checks model loading, a prediction, and one bounded refinement step on
synthetic data; it does not establish detector-level reconstruction accuracy.
"""
from pathlib import Path
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import tensorflow as tf

from src.sampler import (
    MAX_DELTAS,
    SIGNAL_LENGTH,
    Clip,
    Generar_Datos_Etiquetados,
    H_variable,
    mse_sin_padding,
    refinar_tiempos_y_amplitudes,
)


def main() -> None:
    np.random.seed(2026)
    tf.random.set_seed(2026)

    checkpoint = Path("models/best_model.keras")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing saved model: {checkpoint}")

    model = tf.keras.models.load_model(
        checkpoint,
        custom_objects={"Clip": Clip, "mse_sin_padding": mse_sin_padding},
        compile=False,
    )

    signals, times, _labels, _coefficients, total_amplitude = (
        Generar_Datos_Etiquetados(1, max_d=MAX_DELTAS)
    )
    prediction = np.asarray(
        model.predict(
            [
                signals[..., None],
                times.reshape(-1, 1),
                total_amplitude.reshape(-1, 1),
            ],
            verbose=0,
        )
    )
    expected_shape = (1, MAX_DELTAS, 5)
    if prediction.shape != expected_shape:
        raise AssertionError(
            f"Unexpected output shape: {prediction.shape}; expected {expected_shape}"
        )
    if not np.isfinite(prediction).all():
        raise AssertionError("Non-finite model outputs")

    t = np.arange(SIGNAL_LENGTH)
    refined = refinar_tiempos_y_amplitudes(
        prediction[0].reshape(-1).copy(), t, signals[0], MAX_DELTAS, n_iter=1
    ).reshape(MAX_DELTAS, 5)
    if not np.isfinite(refined).all():
        raise AssertionError("Non-finite refined parameters")

    reconstructed = np.zeros(SIGNAL_LENGTH, dtype=np.float64)
    for time, amplitude, scale, a, b in refined:
        reconstructed += amplitude * H_variable(t - time, a, b, scale)
    if not np.isfinite(reconstructed).all():
        raise AssertionError("Non-finite reconstructed waveform")

    residual = float(np.linalg.norm(signals[0] - reconstructed))
    print(f"Model loaded: {checkpoint}")
    print(f"Output shape: {prediction.shape}")
    print(f"Synthetic waveform reconstruction: OK, residual norm {residual:.5g}")
    print("This is an engineering smoke test, not a physics validation.")


if __name__ == "__main__":
    main()
