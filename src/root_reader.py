import uproot as urt
import numpy as np
import matplotlib.pyplot as plt
dN = 512

class DATA:
    def __init__(self, t, signal_ids, signal_values):
        self.t = t
        self.ids = signal_ids
        self.channel = {}

        for i, canal_id in enumerate(signal_ids):
            start = i * dN
            end = start + dN
            self.channel[canal_id] = np.array(signal_values[start:end])

file = urt.open("PyProc_IAXO_Fe55_1.4bar_133_Vcmbar_RawSig_Samp20ns_Sha51_Noise30_V2.4.3 (1).root")
arrays = file["events"].arrays(["timestamp", "signal_ids", "signal_values"], library="np")

datalist = []
for j in range(len(arrays["timestamp"])):
    t = arrays["timestamp"][j]
    ids = arrays["signal_ids"][j]
    vals = np.array(arrays["signal_values"][j], dtype=np.uint16).astype(np.int16).tolist()
    datalist.append(DATA(t, ids, vals))
    


evento1 = datalist[350]
plt.figure()
for canal in evento1.ids:
    print(len(evento1.channel[canal]))
    plt.plot(evento1.channel[canal], label=f"Canal {canal}")
    plt.legend()
    plt.title("Señal muestreada en un canal")
    plt.xlabel("Muestra")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.show()
plt.show()

print()