import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

dat_file = "cygX1_soft_spec.dat"
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

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Energy (keV)", fontsize=14)
ax.set_ylabel(r"$E^2 \frac{dN}{dE}$ (keV cm$^{-2}$ s$^{-1}$)", fontsize=14)

# Stessi limiti x dell'immagine; i limiti y sono spostati rispetto a (0.1, 100)
# perché questa sorgente ha E^2 dN/dE tra circa 0.03 e 0.9 keV cm^-2 s^-1
ax.set_xlim(80, 12500)
ax.set_ylim(0.01, 10)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", which="major", labelsize=12)

ax.legend(fontsize=8, loc="upper right", frameon=True, fancybox=True)

fig.savefig(out_file, dpi=200, bbox_inches="tight")
print(f"Salvato: {out_file}")