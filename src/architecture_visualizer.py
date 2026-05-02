# ─────────────────────────────────────────────────────────
#  0.  Imports y modelo base
# ─────────────────────────────────────────────────────────
import random, os, tensorflow as tf, visualkeras
from collections import defaultdict
from transformer_model import build_transformer_model, Clip   # tu capa custom
os.makedirs("vis_transformer", exist_ok=True)           # carpeta de salida

SIGNAL_LENGTH = 512
MAX_DELTAS    = 10
full_model = build_transformer_model(SIGNAL_LENGTH, MAX_DELTAS)

# ─────────────────────────────────────────────────────────
#  1.  Un color POR CLASE de capa
# ─────────────────────────────────────────────────────────
def pastel() -> tuple[int,int,int]:
    return tuple(random.randint(80,230) for _ in range(3))

# crea diccionario clase→color
layer_colors: dict[type, tuple[int,int,int]] = {}
for lyr in full_model.layers:
    layer_colors.setdefault(type(lyr), pastel())

DEFAULT = (200,200,200)

def build_color_map(model: tf.keras.Model):
    """Mapea cada tensor-salida a su color según la clase de la capa."""
    cmap = defaultdict(lambda: DEFAULT)
    for lyr in model.layers:
        cmap[lyr.output] = layer_colors.get(type(lyr), DEFAULT)
    return cmap

VK = dict(draw_volume=True, spacing=10,
          scale_xy=0.05, scale_z=0.18, max_xy=3, legend=False)

# ─────────────────────────────────────────────────────────
#  2.  ▓ BLOQUE COMÚN ▓
# ─────────────────────────────────────────────────────────
bloque_comun = tf.keras.Model(
    full_model.input,
    full_model.get_layer("global_average_pooling1d").output,
    name="bloque_comun"
)
visualkeras.layered_view(
    bloque_comun,
    to_file="vis_transformer/bloque_comun.png",
    color_map=build_color_map(bloque_comun),
    **VK
)

# ─────────────────────────────────────────────────────────
#  3.  ▓ RAMAS ESPECÍFICAS ▓
# ─────────────────────────────────────────────────────────
enc_lat   = tf.keras.Input((256,), name="enc_lat")
t0_inp    = tf.keras.Input((1,),   name="t0_central")
Atot_inp  = tf.keras.Input((1,),   name="A_total")
concat_258 = tf.keras.layers.Concatenate()([enc_lat, t0_inp, Atot_inp])

# ─ Rama t₀
x = full_model.get_layer("dense_2")(concat_258)
x = full_model.get_layer("dropout_1")(x)
x = full_model.get_layer("repeat_vector")(x)
x = full_model.get_layer("bidirectional")(x)
x = full_model.get_layer("bidirectional_1")(x)
out_t0 = full_model.get_layer("dense_4")(x)
rama_t0 = tf.keras.Model([enc_lat,t0_inp,Atot_inp], out_t0, name="rama_t0")
visualkeras.layered_view(
    rama_t0,
    to_file="vis_transformer/rama_t0.png",
    color_map=build_color_map(rama_t0),
    min_xy = 100,
    **VK
)

# ─ Rama A
x = full_model.get_layer("dense_2")(concat_258)
x = full_model.get_layer("dropout_2")(x)
x = full_model.get_layer("repeat_vector_1")(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)   # ajuste canales
out_A = full_model.get_layer("dense_5")(x)
rama_A = tf.keras.Model([enc_lat,t0_inp,Atot_inp], out_A, name="rama_A")
visualkeras.layered_view(
    rama_A,
    to_file="vis_transformer/rama_A.png",
    color_map=build_color_map(rama_A),
    min_xy = 100,
    **VK
)

# ─ Rama a,b
x = full_model.get_layer("dense")(enc_lat)
out_ab = full_model.get_layer("dense_1")(x)
rama_ab = tf.keras.Model(enc_lat, out_ab, name="rama_ab")
visualkeras.layered_view(
    rama_ab,
    to_file="vis_transformer/rama_ab.png",
    color_map=build_color_map(rama_ab),
    min_xy = 100,
    **VK
)

print("✅ Bloques individualizados guardados en 'vis_transformer/'")

# ─────────────────────────────────────────────────────────
#  4.  LEYENDA GLOBAL  (todas las clases de capa)
# ─────────────────────────────────────────────────────────
inp = tf.keras.Input((1,), name="dummy")
x   = inp
# Recorre COPIA de claves para no alterar el dict mientras iteras
for cls in list(layer_colors.keys()):
    try:
        if cls is tf.keras.layers.Conv1D:
            x = cls(1,1)(tf.keras.layers.Reshape((1,1))(x))
        elif cls is tf.keras.layers.Dense:
            x = cls(1)(x)
        elif cls is tf.keras.layers.MultiHeadAttention:
            tmp = tf.keras.layers.Reshape((1,1))(x)
            x = cls(1, key_dim=1)(tmp, tmp)
        elif cls is Clip:
            x = cls(0,1)(x)
        else:
            x = cls()(x)
    except Exception:
        # si no se puede instanciar rápido, quítala de la leyenda
        layer_colors.pop(cls, None)

legend_model = tf.keras.Model(inp, x)

VK = dict(
    draw_volume=True,
    spacing=10,
    scale_xy=0.1,   # ← sube el 0.05 anterior a 0.09
    scale_z =0.1,
    min_xy = 100,
    legend  =False
)

visualkeras.layered_view(
    bloque_comun,
    to_file="vis_transformer/bloque_comun.png",
    color_map=build_color_map(bloque_comun),
    **VK
)

print("✅ Leyenda global generada: vis_transformer/legend_layers.png")
