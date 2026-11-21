import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("spectra_parameters.json") as f:
    d = json.load(f)

energy = np.array(d["energy_keV"])


def band(x, K, alpha, beta, xp, piv):
    """Funzione di Band, stessa parametrizzazione di astromodels."""
    x = np.asarray(x, dtype=float)
    xb = (alpha - beta) * xp / (2.0 + alpha)  # energia di break
    low = K * (x / piv) ** alpha * np.exp(-(2.0 + alpha) * x / xp)
    high = (K * np.exp(beta - alpha)
            * ((alpha - beta) * xp / (piv * (2.0 + alpha))) ** (alpha - beta)
            * (x / piv) ** beta)
    return np.where(x < xb, low, high)


# Spettro iniettato ricostruito dai parametri salvati
inj = {k: v["value"] for k, v in d["injected"].items()}
flux_inj = band(energy, inj["K"], inj["alpha"], inj["beta"], inj["xp"], inj["piv"])

# Bande del fit (già calcolate nello script di fit)
sb = d["fit_source_plus_background"]["band"]
so = d["fit_source_only"]["band"]
med_sb, lo_sb, hi_sb = (np.array(sb[k]) for k in ("median", "lo_68", "hi_68"))
med_so, lo_so, hi_so = (np.array(so[k]) for k in ("median", "lo_68", "hi_68"))

# ------------------------------------------------------------------
# Plot (identico all'originale)
# ------------------------------------------------------------------
C_SB = "#EE6A35"   # arancione
C_SO = "#1FAF7C"   # verde

fig, ax = plt.subplots(figsize=(6.5, 4.8))

e2 = energy ** 2
ax.plot(energy, e2 * flux_inj, color="black", ls="--", lw=1.5, label="Injected (.dat)")

ax.plot(energy, e2 * med_sb, color=C_SB, lw=.5,
        label="Source + background  (band: statistical 1σ)")
ax.fill_between(energy, e2 * lo_sb, e2 * hi_sb, color=C_SB, alpha=0.4, lw=0)

ax.plot(energy, e2 * med_so, color=C_SO, lw=.5,
        label="Source only  (band: statistical 1σ)")
ax.fill_between(energy, e2 * lo_so, e2 * hi_so, color=C_SO, alpha=0.4, lw=0)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(energy[0] * 0.9, energy[-1])
ax.set_ylim(.1, 100)

ax.set_xlabel("Energy [keV]", fontsize=13)
ax.set_ylabel(r"$E^2 \frac{dN}{dE}$ (keV cm$^{-2}$ s$^{-1}$)", fontsize=13)
ax.tick_params(axis="both", labelsize=12, colors="#444444")

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.xaxis.grid(True, which="major", color="#DDDDDD", lw=1.2)
ax.yaxis.grid(False)
ax.set_axisbelow(True)

ax.legend(frameon=False, loc="upper right", fontsize=10.5, handlelength=2.2)

fig.savefig("injected_vs_fit_from_json.png", dpi=200, bbox_inches="tight")