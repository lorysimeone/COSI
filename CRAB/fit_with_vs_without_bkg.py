import time

t_start = time.perf_counter()

import logging

logging.basicConfig(level=logging.INFO)  # serve solo per avere le informazioni durante i caricamenti

from cosipy import BinnedData
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.response.FullDetectorResponse import FullDetectorResponse
from cosipy.util import fetch_wasabi_file

from cosipy.statistics import PoissonLikelihood
from cosipy.background_estimation import FreeNormBinnedBackground
from cosipy.interfaces import ThreeMLPluginInterface
from cosipy.response import BinnedThreeMLModelFolding, BinnedInstrumentResponse, BinnedThreeMLPointSourceResponse
from cosipy.data_io import EmCDSBinnedData

import sys

import astropy.units as u

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from threeML import Band, PointSource, Model, JointLikelihood, DataList
from astromodels import Parameter, Powerlaw

from pathlib import Path

data_path = Path("/heronas/simeone/COSI/CRAB/data")

# Carico il file di orientazione
sc_orientation = SpacecraftHistory.open(data_path / "DC4_final_530km_3_month_with_slew_1sbins_GalacticEarth_SAA.fits")

'''# Scarico i dati del fondo
single_bkg_fit = True # tratto il fondo come un'unica componente
bkg_components = {"PrimaryProtons":{'filename':'PrimaryProtons_WithDetCstbinned_data_filtered_with_SAAcut.hdf5', 'checksum':'7597f04210e59340a0888c66fc5cbc63'},
                  "PrimaryAlphas": {'filename': 'PrimaryAlphas_WithDetCstbinned_data_filtered_with_SAAcut.hdf5', 'checksum': '76a68da730622851b8e1c749248c3b40'},
                  "AlbedoPhotons": {'filename': 'AlbedoPhotons_WithDetCstbinned_data_filtered_with_SAAcut.hdf5', 'checksum': '76c58361d2c9b43b66ef2e41c18939c4'},
                  "AlbedoNeutrons": {'filename': 'AlbedoNeutrons_WithDetCstbinned_data_filtered_with_SAAcut.hdf5', 'checksum': '8f3cb418c637b839665a4fcbd000d2eb'},
                  "CosmicPhotons": {'filename': 'CosmicPhotons_3months_binned_data_filtered_with_SAAcut.hdf5', 'checksum': '93c4619b383572d318328e6380e35a70'},
                  "CosmicDiffuse": {'filename': 'GalTotal_SA100_F98_3months_binned_data_filtered_with_SAAcut.hdf5', 'checksum': 'd0415d4d04b040af47f23f5d08cb7d64'},
                  "SecondaryPositrons": {'filename': 'SecondaryPositrons_3months_binned_data_filtered_with_SAAcut.hdf5', 'checksum': '5fec2212dcdbb4c43c3ac02f02524f68'},
                  "SecondaryProtons": {'filename': 'SecondaryProtons_WithDetCstbinned_data_filtered_with_SAAcut.fits.gz.hdf5', 'checksum': '78aefa46707c98563294a898a62845c1'},
                  "SAAprotons": {'filename': 'SAA_3months_unbinned_data_filtered_with_SAAcut_statreduced_akaHEPD01result.hdf5', 'checksum': 'fc69fbbfd94cd595f57a8b11fc721169'},
                  }

for bkg in bkg_components.values(): # carico e lavoro su ogni componente del fondo
    binned_data = BinnedData(data_path / "background.yaml")
    binned_data.load_binned_data_from_hdf5(binned_data=data_path/bkg['filename']) # carica il file di quella componente
    bkg['dist'] = binned_data.binned_data.project('Em', 'Phi', 'PsiChi') # proietta l'istogramma sugli assi Em, Phi, PsiChi

# Genero un unico fondo
total_bkg = None
for bkg in bkg_components.values():
    if total_bkg is None:
        total_bkg = bkg['dist']
    else:
        total_bkg = total_bkg + bkg['dist'] # Issues with in-place operations for sparse contents

# Creo l'albero decisionale sul se lavoro su uno o più fondi
if single_bkg_fit:
    bkg_dist = {"total_bkg":total_bkg}
else:
    bkg_dist = {l: b['dist'] for l, b in bkg_components.items()}

# Workaround to avoid inf values. Out bkg should be smooth, but currently it's not.
# Reproduces results before refactoring. It's not _exactly_ the same, since this fudge value was 1e-12, and
# it was added to the expectation, not the normalized bkg

# Evito che ci siano bin simulati con valore 0
for bckfile in bkg_dist.keys() :
    bkg_dist[bckfile] += sys.float_info.min'''

# Carico i dati binnati della Crab
crab_data_path = data_path / "crab_binned_data.hdf5"

crab = BinnedData(data_path / "crab.yaml")
crab.load_binned_data_from_hdf5(binned_data=crab_data_path)

# Carico i dati binnati del fondo
single_bkg_fit = True
bkg_data_path = data_path / "bkg_binned_data.hdf5"

bkg_binned = BinnedData(data_path / "crab.yaml")
bkg_binned.load_binned_data_from_hdf5(binned_data=bkg_data_path)

total_bkg = bkg_binned.binned_data.project('Em', 'Phi', 'PsiChi')
bkg_dist = {"total_bkg": total_bkg}

# Evito che ci siano bin simulati con valore 0
for bckfile in bkg_dist.keys():
    bkg_dist[bckfile] += sys.float_info.min

# Carico la risposta del rilevatore
dr_path = str(
    data_path / "ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284.filtered.nonsparse.binnedimaging.imagingresponse.h5")
dr = FullDetectorResponse.open(dr_path)

# ------------------------------------------------------------------
# Parametri della sorgente (fit)
# ------------------------------------------------------------------
l = 184.56
b = -5.78

alpha = -1.99
beta = -2.32
E0 = 531. * (alpha - beta) * u.keV
xp = E0 * (alpha + 2) / (alpha - beta)
piv = 500. * u.keV
K = 3.07e-5 / u.cm / u.cm / u.s / u.keV

# ------------------------------------------------------------------
# Spettro iniettato nella simulazione (solo per il plot)
# ------------------------------------------------------------------
alpha_inj = -1.99
beta_inj = -2.32
E0_inj = 531. * (alpha_inj - beta_inj) * u.keV
xp_inj = E0_inj * (alpha_inj + 2) / (alpha_inj - beta_inj)
piv_inj = 100. * u.keV
K_inj = 7.56e-4 / u.cm / u.cm / u.s / u.keV

spectrum_inj = Band()

spectrum_inj.alpha.min_value = -2.14
spectrum_inj.alpha.max_value = 3.0
spectrum_inj.beta.min_value = -5.0
spectrum_inj.beta.max_value = -2.15
spectrum_inj.xp.min_value = 1.0

spectrum_inj.alpha.value = alpha_inj
spectrum_inj.beta.value = beta_inj
spectrum_inj.xp.value = xp_inj.value
spectrum_inj.K.value = K_inj.value
spectrum_inj.piv.value = piv_inj.value

spectrum_inj.xp.unit = xp_inj.unit
spectrum_inj.K.unit = K_inj.unit
spectrum_inj.piv.unit = piv_inj.unit

from matplotlib.lines import Line2D

# ------------------------------------------------------------------
# Funzione che esegue l'intero fit per un dato istogramma di dati
# ------------------------------------------------------------------
def run_fit(data_hist, fit_bkg=True):
    data_ = EmCDSBinnedData(data_hist)

    bkg_ = FreeNormBinnedBackground(bkg_dist, sc_history=sc_orientation, copy=False)
    instr_resp = BinnedInstrumentResponse(dr, data_)
    psr_ = BinnedThreeMLPointSourceResponse(data=data_,
                                            instrument_response=instr_resp,
                                            sc_history=sc_orientation,
                                            energy_axis=dr.axes['Ei'],
                                            polarization_axis=dr.axes['Pol'] if 'Pol' in dr.axes.labels else None,
                                            nside=2 * data_.axes['PsiChi'].nside)
    response_ = BinnedThreeMLModelFolding(data=data_, point_source_response=psr_)
    like_fun_ = PoissonLikelihood(data_, response_, bkg_)
    cosi_ = ThreeMLPluginInterface('cosi', like_fun_, response_, bkg_)

    for bkg_label in bkg_dist.keys():
        p = Parameter(bkg_label, 1 if fit_bkg else 0,
                      min_value=0, max_value=100,
                      delta=0.05, unit=u.Hz)
        cosi_.bkg_parameter[bkg_label] = p
        if not fit_bkg:
            p.free = False          # fondo fissato a 0 -> "source only"

    spec = Band()
    spec.alpha.min_value = -2.14
    spec.alpha.max_value = 3.0
    spec.beta.min_value = -5.0
    spec.beta.max_value = -2.15
    spec.xp.min_value = 1.0
    spec.alpha.delta = 0.01
    spec.beta.delta = 0.01
    spec.alpha.value = alpha
    spec.beta.value = beta
    spec.xp.value = xp.value
    spec.K.value = K.value
    spec.piv.value = piv.value
    spec.xp.unit = xp.unit
    spec.K.unit = K.unit
    spec.piv.unit = piv.unit

    src = PointSource("source", l=l, b=b, spectral_shape=spec)
    model_ = Model(src)
    like_ = JointLikelihood(model_, DataList(cosi_), verbose=False)
    res_, like_df_ = like_.fit()
    return like_, like_.results, response_, bkg_, data_


crab_hist = crab.binned_data.project('Em', 'Phi', 'PsiChi')

# Fit 1: source + background
like, res_sb, response, bkg, data = run_fit(crab_hist + total_bkg, fit_bkg=True)
print(res_sb.display())

# Fit 2: source only
_, res_so, _, _, _ = run_fit(crab_hist, fit_bkg=False)
print(res_so.display())


# ------------------------------------------------------------------
# Banda di incertezza del fit
# ------------------------------------------------------------------
energy = np.geomspace(100 * u.keV, 10 * u.MeV).to_value(u.keV)

def spectrum_band(results):
    pars = {par.name: results.get_variates(par.path)
            for par in results.optimized_model["source"].parameters.values()
            if par.free}
    func = results.propagate(results.optimized_model["source"].spectrum.main.shape.evaluate_at, **pars)
    med, lo, hi = (np.zeros_like(energy) for _ in range(3))
    for i, e in enumerate(energy):
        f = func(e)
        med[i] = f.median
        lo[i], hi[i] = f.equal_tail_interval(cl=0.68)
    return med, lo, hi

med_sb, lo_sb, hi_sb = spectrum_band(res_sb)
med_so, lo_so, hi_so = spectrum_band(res_so)
flux_inj = np.array([spectrum_inj.evaluate_at(e) for e in energy])

# ------------------------------------------------------------------
# Plot
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

# Stile: solo spine sinistra e in basso, griglia verticale chiara sulle decadi
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.xaxis.grid(True, which="major", color="#DDDDDD", lw=1.2)
ax.yaxis.grid(False)
ax.set_axisbelow(True)

# Legenda senza cornice, in alto a destra
ax.legend(frameon=False, loc="upper right", fontsize=10.5, handlelength=2.2)

fig.savefig("injected_vs_fit.png", dpi=200, bbox_inches="tight")