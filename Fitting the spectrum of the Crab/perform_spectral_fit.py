import logging
logging.basicConfig(level=logging.INFO) # serve solo per avere le informazioni durante i caricamenti

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

data_path = Path("/heronas/simeone/COSI/crab_data")

# Carico il file di orientazione
sc_orientation = SpacecraftHistory.open(data_path / "20280301_3_month_with_orbital_info.fits")

# Scarico i dati del fondo
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


data_path = Path("/heronas/simeone/COSI/crab_data")
crab_data_path = data_path / "crab_standard_3months_binned_data_filtered_with_SAAcut.fits.gz.hdf5"

crab = BinnedData(data_path / "crab.yaml")
crab.load_binned_data_from_hdf5(binned_data=crab_data_path) # carica i dati della Crab

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
    bkg_dist[bckfile] += sys.float_info.min

# Carico la risposta del rilevatore
dr_path = str(data_path / "ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284.filtered.nonsparse.binnedimaging.imagingresponse.h5")
dr = FullDetectorResponse.open(dr_path)



# Incapsulo gli oggetti BinnedData nell'interfaccia appropriata
data = EmCDSBinnedData(crab.binned_data.project('Em', 'Phi', 'PsiChi') + total_bkg)

# Uso il modello del fondo per inizializzare un'interfaccia di fondo atteso
# Per questo fondo, sono liberi solo i parametri di normalizzazione
bkg = FreeNormBinnedBackground(bkg_dist,
                               sc_history=sc_orientation,
                               copy = False)

# Wrape the raw response with BinnedInstrumentResponseInterface implementation
instrument_response = BinnedInstrumentResponse(dr, data)

# Inizializzo la PSR di 3ML
psr = BinnedThreeMLPointSourceResponse(data = data,
                                       instrument_response = instrument_response,
                                       sc_history=sc_orientation,
                                       energy_axis = dr.axes['Ei'],
                                       polarization_axis = dr.axes['Pol'] if 'Pol' in dr.axes.labels else None,
                                       nside = 2*data.axes['PsiChi'].nside)

# Passo la PSR di 3ML alla funzione che esegue la convoluzione con lo spettro
response = BinnedThreeMLModelFolding(data = data, point_source_response = psr)

like_fun = PoissonLikelihood(data, response, bkg) # la likelihood da usare

# Creo il plugin di CPSI
cosi = ThreeMLPluginInterface('cosi',
                              like_fun,
                              response,
                              bkg)
# Creo i parametri del fondo, ovvero le ambpiezze per ogni bin
for bkg_label in bkg_dist.keys():
    cosi.bkg_parameter[bkg_label] = Parameter(bkg_label,  # parametro del fondo
                                      1,  # valore iniziale del parametro
                                      min_value=0,  # valore minimo del parametro
                                      max_value= 100 if single_bkg_fit else 20,  # valore massimo del parametro
                                      delta=0.05,  # passo iniziale usato dal minimizzatore
                                      unit = u.Hz # unità: conteggi al secondo
                                      )

# Definisco il modello di sorgente puntiforme
l = 184.56
b = -5.78

alpha = -1.99
beta = -2.32
E0 = 531. * (alpha - beta) * u.keV
xp = E0 * (alpha + 2) / (alpha - beta)
piv = 500. * u.keV
K = 3.07e-5 / u.cm / u.cm / u.s / u.keV

spectrum = Band()

spectrum.alpha.min_value = -2.14
spectrum.alpha.max_value = 3.0
spectrum.beta.min_value = -5.0
spectrum.beta.max_value = -2.15
spectrum.xp.min_value = 1.0
spectrum.alpha.delta = 0.01
spectrum.beta.delta = 0.01

spectrum.alpha.value = alpha
spectrum.beta.value = beta
spectrum.xp.value = xp.value
spectrum.K.value = K.value
spectrum.piv.value = piv.value

spectrum.xp.unit = xp.unit
spectrum.K.unit = K.unit
spectrum.piv.unit = piv.unit

source = PointSource("source",                     # nome della sorgente (arbitrario, ma deve essere univoco)
                     l = l,                        # Longitude (deg)
                     b = b,                        # Latitude (deg)
                     spectral_shape = spectrum)    # Spectral model

# Opzionale: rendo liberi i parametri della posizione
#source.position.l.free = True
#source.position.b.free = True

model = Model(source)  # Modello a unica sorgente (altrimenti è Model(source1, source2, ...))

plugins = DataList(cosi) # raccoglie il dataset da usare nel fit

like = JointLikelihood(model, plugins, verbose = False) # calcola la verosomiglianza

results, like_df = like.fit()
print(results)
print(like_df)

# Definisco la funzione di Band della simulazione per il plot
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

# Definisco i risultati per il confronto

results = like.results

print(results.display())

parameters = {par.name:results.get_variates(par.path)
              for par in results.optimized_model["source"].parameters.values()
              if par.free}

results_err = results.propagate(results.optimized_model["source"].spectrum.main.shape.evaluate_at, **parameters)

print(results.optimized_model["source"])

# Definisco il flusso e gli errori nei range utilizzati per il fit e nei dati immessi
energy = np.geomspace(100*u.keV,10*u.MeV).to_value(u.keV)

flux_lo = np.zeros_like(energy)
flux_median = np.zeros_like(energy)
flux_hi = np.zeros_like(energy)
flux_inj = np.zeros_like(energy)

for i, e in enumerate(energy):
    flux = results_err(e)
    flux_median[i] = flux.median
    flux_lo[i], flux_hi[i] = flux.equal_tail_interval(cl=0.68)
    flux_inj[i] = spectrum_inj.evaluate_at(e)

binned_energy_edges = crab.binned_data.axes['Em'].edges.value
binned_energy = np.array([])
bin_sizes = np.array([])

for i in range(len(binned_energy_edges)-1):
    binned_energy = np.append(binned_energy, (binned_energy_edges[i+1] + binned_energy_edges[i]) / 2)
    bin_sizes = np.append(bin_sizes, binned_energy_edges[i+1] - binned_energy_edges[i])

expectation = response.expectation()

# Plotto lo spettro immesso e lo spettro fittato
fig,ax = plt.subplots()

ax.plot(energy, energy*energy*flux_median, label = "Best fit")
ax.fill_between(energy, energy*energy*flux_lo, energy*energy*flux_hi, alpha = .5, label = "Best fit (errors)")
ax.plot(energy, energy*energy*flux_inj, color = 'black', ls = ":", label = "Injected")

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Energy (keV)")
ax.set_ylabel(r"$E^2 \frac{dN}{dE}$ (keV cm$^{-2}$ s$^{-1}$)")

ax.set_ylim(.1,100)

_ = ax.legend()

ax.figure.savefig("injected_vs_fit.png", dpi=200, bbox_inches="tight")

# Plotto lo spettro convoluto con la risposta e lo spettro fittato

fig,ax = plt.subplots()

ax.stairs(expectation.project('Em').to_dense(copy=False).contents, binned_energy_edges, color='purple', label = "Best fit convolved with response")
ax.errorbar(binned_energy, expectation.project('Em').to_dense(copy=False).contents, yerr=np.sqrt(expectation.project('Em').to_dense(copy=False).contents), color='purple', linewidth=0, elinewidth=1)
ax.stairs(crab.binned_data.project('Em').to_dense(copy=False).contents, binned_energy_edges, color = 'black', ls = ":", label = "Source counts")
ax.errorbar(binned_energy, crab.binned_data.project('Em').to_dense(copy=False).contents, yerr=np.sqrt(crab.binned_data.project('Em').to_dense(copy=False).contents), color='black', linewidth=0, elinewidth=1)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Energy (keV)")
ax.set_ylabel("Counts")

_ = ax.legend()

ax.figure.savefig("convolved_spectrum_vs_fit.png", dpi=200, bbox_inches="tight")

# Aggiungo il fondo
expectation_bkg = bkg.expectation(copy = True)

fig,ax = plt.subplots()

ax.stairs(expectation.project('Em').to_dense(copy=False).contents + expectation_bkg.project('Em').to_dense(copy=False).contents, binned_energy_edges, color='purple', label = "Best fit convolved with response plus background")
ax.errorbar(binned_energy, expectation.project('Em').to_dense(copy=False).contents+expectation_bkg.project('Em').to_dense(copy=False).contents, yerr=np.sqrt(expectation.project('Em').to_dense(copy=False).contents+expectation_bkg.project('Em').to_dense(copy=False).contents), color='purple', linewidth=0, elinewidth=1)
ax.stairs(data.data.project('Em').to_dense(copy=False).contents, binned_energy_edges, color = 'black', ls = ":", label = "Total counts")
ax.errorbar(binned_energy, data.data.project('Em').to_dense(copy=False).contents, yerr=np.sqrt(data.data.project('Em').to_dense(copy=False).contents), color='black', linewidth=0, elinewidth=1)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Energy (keV)")
ax.set_ylabel("Counts")

ax.legend()

plt.show()

ax.figure.savefig("convolved_spectrum_vs_fit_with_background.png", dpi=200, bbox_inches="tight")