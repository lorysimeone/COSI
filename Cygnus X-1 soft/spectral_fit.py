import time

t_start = time.perf_counter()

import sys
import pickle

import logging

logging.basicConfig(level=logging.INFO)  # attivo i log informativi durante i caricamenti

from cosipy import BinnedData
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.response.FullDetectorResponse import FullDetectorResponse
from cosipy.util import fetch_wasabi_file

from cosipy.statistics import PoissonLikelihood
from cosipy.background_estimation import FreeNormBinnedBackground
from cosipy.interfaces import ThreeMLPluginInterface
from cosipy.response import BinnedThreeMLModelFolding, BinnedInstrumentResponse, BinnedThreeMLPointSourceResponse
from cosipy.data_io import EmCDSBinnedData

import astropy.units as u

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from threeML import Band, PointSource, Model, JointLikelihood, DataList
from astromodels import Parameter, Powerlaw

from pathlib import Path


def prepare_cosi_plugin(
    sc_orientation_path: Path,
    yaml_path: Path,
    source_data_path: Path,
    bkg_data_path: Path,
    dr_path: Path,
):
    # Carico il file di orientazione
    sc_orientation = SpacecraftHistory.open(sc_orientation_path)

    # Carico i dati binnati
    source = BinnedData(yaml_path)
    source.load_binned_data_from_hdf5(binned_data=source_data_path)

    # Carico i dati binnati del fondo
    single_bkg_fit = True
    bkg_binned = BinnedData(yaml_path)
    bkg_binned.load_binned_data_from_hdf5(binned_data=bkg_data_path)

    total_bkg = bkg_binned.binned_data.project('Em', 'Phi', 'PsiChi')
    bkg_dist = {"total_bkg": total_bkg}

    # Evito che ci siano bin simulati con valore 0
    for bckfile in bkg_dist.keys():
        bkg_dist[bckfile] += sys.float_info.min

    # Carico la risposta del rilevatore
    dr = FullDetectorResponse.open(str(dr_path))

    # Incapsulo gli oggetti BinnedData nell'interfaccia appropriata
    data = EmCDSBinnedData(source.binned_data.project('Em', 'Phi', 'PsiChi') + total_bkg)

    # Uso il modello del fondo per inizializzare un'interfaccia di fondo atteso
    # Per questo fondo, sono liberi solo i parametri di normalizzazione
    bkg = FreeNormBinnedBackground(bkg_dist,
                                   sc_history=sc_orientation,
                                   copy=False)

    # Wrape the raw response with BinnedInstrumentResponseInterface implementation
    instrument_response = BinnedInstrumentResponse(dr, data)

    # Inizializzo la PSR di 3ML
    psr = BinnedThreeMLPointSourceResponse(data=data,
                                           instrument_response=instrument_response,
                                           sc_history=sc_orientation,
                                           energy_axis=dr.axes['Ei'],
                                           polarization_axis=dr.axes['Pol'] if 'Pol' in dr.axes.labels else None,
                                           nside=2 * data.axes['PsiChi'].nside)

    # Passo la PSR di 3ML alla funzione che esegue la convoluzione con lo spettro
    response = BinnedThreeMLModelFolding(data=data, point_source_response=psr)

    like_fun = PoissonLikelihood(data, response, bkg)  # la likelihood da usare

    # Creo il plugin di COSI
    cosi = ThreeMLPluginInterface('cosi',
                                  like_fun,
                                  response,
                                  bkg)
    # Creo i parametri del fondo, ovvero le ampiezze per ogni bin
    for bkg_label in bkg_dist.keys():
        cosi.bkg_parameter[bkg_label] = Parameter(bkg_label,  # parametro del fondo
                                                  1,  # valore iniziale del parametro
                                                  min_value=0,  # valore minimo del parametro
                                                  max_value=100 if single_bkg_fit else 20,  # valore massimo del parametro
                                                  delta=0.05,  # passo iniziale usato dal minimizzatore
                                                  unit=u.Hz  # unità: conteggi al secondo
                                                  )

    return cosi, response, bkg, source, data


data_path_ori_bkg = Path("/heronas/simeone/COSI/ori_bkg_yaml")
data_path_cygnus_soft = Path("/heronas/simeone/COSI/Cygnus X-1 soft")

sc_orientation_path = data_path_ori_bkg / "DC4_final_530km_3_month_with_slew_1sbins_GalacticEarth_SAA.fits"
yaml_path = data_path_ori_bkg / "DC4.yaml"
source_data_path = data_path_cygnus_soft / "cygnus_soft_binned_data.hdf5"
bkg_data_path = data_path_ori_bkg / "bkg_binned_data.hdf5"
dr_path = data_path_ori_bkg / "ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284.filtered.nonsparse.binnedimaging.imagingresponse.h5"

t0 = time.perf_counter()
cosi, response, bkg, source, data = prepare_cosi_plugin(
    sc_orientation_path= sc_orientation_path,
    yaml_path = yaml_path,
    source_data_path = source_data_path,
    bkg_data_path = bkg_data_path,
    dr_path = dr_path)
print(f"prepare_cosi_plugin: {time.perf_counter()-t0:.1f}s")

# Definisco il modello di sorgente puntiforme con legge di potenza singola
l = 184.56
b = -5.78

index = -2.0
piv = 100.0 * u.keV
K = 8.6e-5 / u.cm / u.cm / u.s / u.keV

spectrum = Powerlaw()  # creo il modello spettrale a legge di potenza singola

spectrum.index.min_value = -6.0  # imposto il limite inferiore per l'indice spettrale
spectrum.index.max_value = 2.0  # imposto il limite superiore per l'indice spettrale
spectrum.index.delta = 0.05  # imposto il passo iniziale per l'indice spettrale

spectrum.K.value = K.value  # imposto il valore iniziale della normalizzazione
spectrum.piv.value = piv.value  # imposto l'energia di pivot iniziale
spectrum.index.value = index  # imposto l'indice spettrale iniziale

spectrum.K.unit = K.unit  # assegno l'unità di misura alla normalizzazione
spectrum.piv.unit = piv.unit  # assegno l'unità di misura all'energia di pivot

spectrum.K.free = True  # lascio libera la normalizzazione durante il fit
spectrum.index.free = True  # lascio libero l'indice spettrale durante il fit
spectrum.piv.free = False  # tengo fissa l'energia di pivot

source = PointSource("source",  # nome della sorgente (arbitrario, ma deve essere univoco)
                     l=l,  # Longitude (deg)
                     b=b,  # Latitude (deg)
                     spectral_shape=spectrum)  # Spectral model

# Opzionale: rendo liberi i parametri della posizione
# source.position.l.free = True
# source.position.b.free = True

model = Model(source)  # Modello a unica sorgente (altrimenti è Model(source1, source2, ...))

plugins = DataList(cosi)  # raccoglie il dataset da usare nel fit

like = JointLikelihood(model, plugins, verbose=False)  # calcola la verosomiglianza
like = JointLikelihood(model, plugins, verbose=False)  # calcola la verosomiglianza

t0 = time.perf_counter()
results, like_df = like.fit()
print(f"fit: {time.perf_counter()-t0:.1f}s")
print(results)
print(like_df)

# Definisco i risultati per il confronto

results = like.results

print(results.display())

parameters = {par.name: results.get_variates(par.path)
              for par in results.optimized_model["source"].parameters.values()
              if par.free}

results_err = results.propagate(results.optimized_model["source"].spectrum.main.shape.evaluate_at, **parameters)

print(results.optimized_model["source"])

# Definisco il flusso e gli errori nel range utilizzato per il fit
energy = np.geomspace(100 * u.keV, 10 * u.MeV).to_value(u.keV)

flux_lo = np.zeros_like(energy)
flux_median = np.zeros_like(energy)
flux_hi = np.zeros_like(energy)

for i, e in enumerate(energy):
    flux = results_err(e)
    flux_median[i] = flux.median
    flux_lo[i], flux_hi[i] = flux.equal_tail_interval(cl=0.68)

# Plotto lo spettro fittato (con la sua banda d'errore al 68%) e i punti del file .dat
fig, ax = plt.subplots()  # creo una nuova figura per il plot dello spettro fittato

dat_file_path = data_path_cygnus_soft / "model/cygX1_soft_spec.dat"  # definisco il percorso del file .dat dello spettro
dat_energies = []  # preparo la lista per le energie lette dal file .dat
dat_fluxes = []  # preparo la lista per i flussi differenziali letti dal file .dat

with open(dat_file_path, "r") as f:  # apro il file .dat in lettura
    for line in f:  # leggo il file riga per riga
        parts = line.strip().split()  # rimuovo gli spazi superflui e divido la riga in campi
        if len(parts) == 3 and parts[0] == "DP":  # considero solo le righe dati formattate come DP <energia> <flusso>
            dat_energies.append(float(parts[1]))  # aggiungo l'energia in keV
            dat_fluxes.append(float(parts[2]))  # aggiungo il flusso differenziale in ph cm^-2 s^-1 keV^-1

dat_energies = np.array(dat_energies)  # converto le energie in un array NumPy
dat_fluxes = np.array(dat_fluxes)  # converto i flussi in un array NumPy

ax.scatter(dat_energies, dat_energies * dat_energies * dat_fluxes, color="black", s=6, zorder=4, label="cygX1_soft_spec.dat")  # aggiungo lo scatter plot dei punti del file .dat in scala E^2 dN/dE

spec_opt = results.optimized_model["source"].spectrum.main.shape  # recupero il modello spettrale ottimizzato
k_val = spec_opt.K.value  # estraggo il valore fittato di K
idx_val = spec_opt.index.value  # estraggo il valore fittato dell'indice spettrale
piv_val = spec_opt.piv.value  # estraggo il valore dell'energia di pivot

k_err = results.get_variates(spec_opt.K.path).std  # calcolo l'incertezza su K dai variates del fit
idx_err = results.get_variates(spec_opt.index.path).std  # calcolo l'incertezza sull'indice spettrale dai variates del fit

fit_label = (  # preparo l'etichetta della legenda con i parametri del fit
    f"Best fit\n"
    f"K = {k_val:.2e} ± {k_err:.2e} (at {piv_val:g} keV)\n"
    f"index = {idx_val:.2f} ± {idx_err:.2f}"
)

ax.plot(energy, energy * energy * flux_median, label=fit_label)  # traccio la mediana dello spettro fittato includendo i parametri in legenda
ax.fill_between(energy, energy * energy * flux_lo, energy * energy * flux_hi, alpha=.5, label="Best fit (errors)")  # ombreggio l'intervallo al 68%

ax.set_xscale("log")  # imposto la scala logaritmica sull'asse delle energie
ax.set_yscale("log")  # imposto la scala logaritmica sull'asse del flusso

ax.set_xlabel("Energy (keV)")  # etichetta dell'asse x
ax.set_ylabel(r"$E^2 \frac{dN}{dE}$ (keV cm$^{-2}$ s$^{-1}$)")  # etichetta dell'asse y

ax.set_ylim(.1, 100)  # imposto i limiti verticali del plot

_ = ax.legend()  # mostro la legenda aggiornata con i parametri e i punti del file .dat

output_dir = Path("spectra_figures")  # definisco il percorso della cartella di destinazione per le figure
output_dir.mkdir(parents=True, exist_ok=True)  # creo la cartella e le eventuali cartelle intermedie se non esistono già
ax.figure.savefig(output_dir / "fitted_spectrum_power_law.png", dpi=200, bbox_inches="tight")  # salvo il plot dello spettro fittato dentro la cartella
plt.close(fig)  # chiudo la figura per liberare memoria

t_end = time.perf_counter()
elapsed = t_end - t_start
minutes, seconds = divmod(elapsed, 60)
print(f"Tempo totale di esecuzione: {int(minutes)}m {seconds:.2f}s ({elapsed:.2f}s)")