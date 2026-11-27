import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path
from fit_function import power_model

def trova_file(nome, radici):
    """Cerca `nome` in ciascuna radice (e nelle sue sottocartelle), nell'ordine dato."""
    if isinstance(radici, (str, Path)):
        radici = [radici]                      # accetto anche una radice singola

    for radice in map(Path, radici):
        if not radice.exists():
            print(f"Radice non presente su questo computer, la salto: {radice}")
            continue

        trovati = sorted(radice.rglob(nome))
        if trovati:
            if len(trovati) > 1:
                print(f"Attenzione: {len(trovati)} file chiamati {nome} sotto {radice}:")
                for p in trovati:
                    print("  ", p)
                print("Uso il primo.")
            return trovati[0]

    raise FileNotFoundError(f"{nome} non trovato in nessuna di queste radici: {[str(r) for r in radici]}")

radici = [
    "/heronas/simeone/COSI",
    "/home/lorysimeone/COSI/Cygnus X-1 soft/model",
]

dat_file = trova_file("cygX1_soft_spec.dat", radici)
print(f"Uso: {dat_file}")
out_file = "cygX1_soft_spectrum_model.png"

# Leggo le righe "DP energia(keV) dN/dE(ph cm^-2 s^-1 keV^-1)"
energy, dnde = [], []
with open(dat_file) as f:
    for line in f:
        p = line.split()
        if p and p[0] == "DP":
            energy.append(float(p[1]))
            dnde.append(float(p[2]))
energy = np.array(energy)
dnde = np.array(dnde)

# NUOVO: creo il modello spettrale e lo valuto su una griglia di energie in keV
spectrum = power_model()       # usa i valori di default: index=-2.0, K=8.6e-5 a 100 keV
e_grid = np.geomspace(80, 12500, 200)     # keV, stesso intervallo dell'asse x del grafico
model_dnde = spectrum(e_grid)             # dN/dE in ph cm^-2 s^-1 keV^-1

# Stile: assi grigio scuro, niente bordi in alto e a destra
col = "#3d3d3d"
plt.rcParams.update({
    "axes.edgecolor": col,
    "axes.labelcolor": col,
    "xtick.color": col,
    "ytick.color": col,
    "text.color": col,
    "axes.linewidth": 1.2,
})

fig, ax = plt.subplots()

# Lo spettro della tabella
ax.plot(energy, energy * energy * dnde, color="black", ls=":", lw=2.2, label="Injected")

# NUOVO: il modello della funzione (valori iniziali del fit)
model_label = (f"Initial model\nK = {spectrum.K.value:.2e} (at {spectrum.piv.value:g} keV)\n"
               f"\u0393 = {spectrum.index.value:.2f}")
ax.plot(e_grid, e_grid * e_grid * model_dnde, color="tab:red", lw=1.8, label=model_label)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Energy (keV)", fontsize=14)
ax.set_ylabel(r"$E^2 \frac{dN}{dE}$ (keV cm$^{-2}$ s$^{-1}$)", fontsize=14)

ax.set_xlim(80, 12500)
ax.set_ylim(0.01, 10)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", which="major", labelsize=12)

ax.legend(fontsize=8, loc="upper right", frameon=True, fancybox=True)

fig.savefig(out_file, dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)
print(f"Salvato: {out_file}")