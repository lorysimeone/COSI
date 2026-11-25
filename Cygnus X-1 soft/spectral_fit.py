import time

t_start = time.perf_counter()

import sys
import pickle
import hashlib
import logging
from pathlib import Path
import sys as _sys
from contextlib import contextmanager

_fit_function_dir = Path("/heronas/simeone/COSI/Cygnus X-1 soft/model")
if str(_fit_function_dir) not in _sys.path:
    _sys.path.insert(0, str(_fit_function_dir))

from fit_function import power_model

logging.basicConfig(level=logging.INFO)  # attivo i log informativi durante i caricamenti

from cosipy import BinnedData
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.response.FullDetectorResponse import FullDetectorResponse

from cosipy.statistics import PoissonLikelihood
from cosipy.background_estimation import FreeNormBinnedBackground
from cosipy.interfaces import ThreeMLPluginInterface
from cosipy.response import BinnedThreeMLModelFolding, BinnedInstrumentResponse, BinnedThreeMLPointSourceResponse
from cosipy.data_io import EmCDSBinnedData
from histpy import Histogram

import astropy.units as u
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from threeML import PointSource, Model, JointLikelihood, DataList
from astromodels import Parameter

CACHE_DIR = Path("cache_psr")
CACHE_BKG_DIR = Path("cache_bkg")
CACHE_SC_DIR = Path("cache_sc")


def _file_stamp(path: Path) -> str:
    # Recupero percorso assoluto e timestamp di modifica per validare la cache
    p = Path(path).resolve()
    return f"{p}:{p.stat().st_mtime_ns}"


def bkg_cache_file(bkg_data_path: Path, yaml_path: Path, axes=("Em", "Phi", "PsiChi")) -> Path:
    # Calcolo un hash univoco in base ai file sorgente, ai loro timestamp e agli assi proiettati
    h = hashlib.sha1()
    h.update(_file_stamp(bkg_data_path).encode())
    h.update(_file_stamp(yaml_path).encode())
    h.update(",".join(axes).encode())
    return CACHE_BKG_DIR / f"total_bkg_{h.hexdigest()[:10]}.h5"


def sc_cache_file(sc_orientation_path: Path) -> Path:
    # Calcolo un hash univoco in base al file FITS di orientazione e al suo timestamp
    h = hashlib.sha1()
    h.update(_file_stamp(sc_orientation_path).encode())
    return CACHE_SC_DIR / f"sc_history_{h.hexdigest()[:10]}.pkl"


def load_sc_history(sc_orientation_path: Path) -> SpacecraftHistory:
    # Carico la storia del satellite dalla cache se disponibile, altrimenti leggo il FITS
    cache_path = sc_cache_file(sc_orientation_path)
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                sc_orientation = pickle.load(f)
            print(f"SpacecraftHistory caricata da {cache_path}")
            return sc_orientation
        except Exception as e:
            print("Caricamento cache SpacecraftHistory fallito, rileggo FITS:", type(e).__name__, e)
    return SpacecraftHistory.open(sc_orientation_path)


def load_total_bkg(bkg_data_path: Path, yaml_path: Path, axes=("Em", "Phi", "PsiChi")):
    # Carico l'istogramma del fondo dalla cache se disponibile, altrimenti lo calcolo
    cache_path = bkg_cache_file(bkg_data_path, yaml_path, axes=axes)
    if cache_path.exists():
        try:
            total_bkg = Histogram.open(cache_path)
            print(f"total_bkg caricato da {cache_path}")
            return total_bkg
        except Exception as e:
            print("Caricamento cache total_bkg fallito, ricalcolo:", type(e).__name__, e)

    bkg_binned = BinnedData(yaml_path)
    bkg_binned.load_binned_data_from_hdf5(binned_data=bkg_data_path)
    return bkg_binned.binned_data.project(*axes)


def psr_cache_file(l, b, sc_orientation_path, dr_path, yaml_path, nside):
    # Calcolo il percorso della PSR in cache in base a coordinate, nside e file di input
    h = hashlib.sha1()
    for p in (sc_orientation_path, dr_path, yaml_path):
        p = Path(p)
        h.update(str(p.resolve()).encode())
        h.update(str(p.stat().st_mtime_ns).encode())
    h.update(f"{l:.6f}_{b:.6f}_{nside}".encode())
    return CACHE_DIR / f"psr_l{l:.4f}_b{b:.4f}_{h.hexdigest()[:10]}.h5"


def save_psr(response, path, name="source"):
    # Salvo la PSR calcolata dalla copia per-sorgente creata dal folding
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sr = getattr(response, "_source_responses", {}).get(name)
    if sr is None or sr._psr is None:
        print("PSR non ancora calcolata, niente da salvare")
        return False
    try:
        sr._psr.write(path, overwrite=True)
        print(f"PSR salvata in {path}")
        return True
    except Exception as e:
        print("Salvataggio PSR fallito:", type(e).__name__, e)
        return False


def load_psr(psr_template, path, point_source):
    # Inietto la PSR già calcolata nelle copie create da psr_template.copy()
    from cosipy.response import PointSourceResponse
    if not Path(path).exists():
        return False
    try:
        loaded_psr = PointSourceResponse.open(path)
        coord = point_source.position.sky_coord.copy()
        original_copy = psr_template.copy

        def cached_copy():
            new = original_copy()
            new._psr = loaded_psr
            new._last_convolved_source_skycoord = coord.copy()
            return new

        psr_template.copy = cached_copy
        print(f"PSR caricata da {path}")
        return True
    except Exception as e:
        print("Caricamento PSR fallito, ricalcolo:", type(e).__name__, e)
        return False


@contextmanager
def _step_timer(label: str):
    # Misuro il tempo trascorso per la singola fase di preparazione del plugin
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"  [prepare_cosi_plugin] {label}: {time.perf_counter() - t0:.2f}s")


def prepare_cosi_plugin(
    sc_orientation_path: Path,
    yaml_path: Path,
    source_data_path: Path,
    bkg_data_path: Path,
    dr_path: Path,
):
    with _step_timer("SpacecraftHistory load"):
        # Carico il file di orientazione
        sc_orientation = load_sc_history(sc_orientation_path)

    with _step_timer("load source BinnedData"):
        # Carico i dati binnati della sorgente
        source = BinnedData(yaml_path)
        source.load_binned_data_from_hdf5(binned_data=source_data_path)

    with _step_timer("load & project bkg BinnedData"):
        # Carico i dati binnati del fondo già proiettati
        single_bkg_fit = True
        total_bkg = load_total_bkg(bkg_data_path, yaml_path, axes=("Em", "Phi", "PsiChi"))
        bkg_dist = {"total_bkg": total_bkg}

        # Evito che ci siano bin simulati con valore 0
        for bckfile in bkg_dist.keys():
            bkg_dist[bckfile] += sys.float_info.min

    with _step_timer("project & sum source + bkg"):
        # Incapsulo i conteggi totali nell'interfaccia appropriata
        data = EmCDSBinnedData(source.binned_data.project("Em", "Phi", "PsiChi") + total_bkg)

    with _step_timer("FreeNormBinnedBackground"):
        # Inizializzo il fondo con normalizzazione libera
        bkg = FreeNormBinnedBackground(bkg_dist,
                                       sc_history=sc_orientation,
                                       copy=False)

    with _step_timer("FullDetectorResponse.open + BinnedInstrumentResponse"):
        # Carico la risposta del rilevatore e la associo ai dati
        dr = FullDetectorResponse.open(str(dr_path))
        instrument_response = BinnedInstrumentResponse(dr, data)

    with _step_timer("BinnedThreeMLPointSourceResponse"):
        # Inizializzo la PSR per 3ML
        psr = BinnedThreeMLPointSourceResponse(data=data,
                                               instrument_response=instrument_response,
                                               sc_history=sc_orientation,
                                               energy_axis=dr.axes["Ei"],
                                               polarization_axis=dr.axes["Pol"] if "Pol" in dr.axes.labels else None,
                                               nside=2 * data.axes["PsiChi"].nside)

    with _step_timer("BinnedThreeMLModelFolding + PoissonLikelihood + ThreeMLPluginInterface"):
        # Collego folding spettrale, likelihood di Poisson e plugin di 3ML
        response = BinnedThreeMLModelFolding(data=data, point_source_response=psr)
        like_fun = PoissonLikelihood(data, response, bkg)
        cosi = ThreeMLPluginInterface("cosi", like_fun, response, bkg)

        for bkg_label in bkg_dist.keys():
            cosi.bkg_parameter[bkg_label] = Parameter(bkg_label,
                                                      1,
                                                      min_value=0,
                                                      max_value=100 if single_bkg_fit else 20,
                                                      delta=0.05,
                                                      unit=u.Hz)

    return cosi, response, bkg, source, data, psr


data_path_ori_bkg = Path("/heronas/simeone/COSI/ori_bkg_yaml")
data_path_cygnus_soft = Path("/heronas/simeone/COSI/Cygnus X-1 soft")

sc_orientation_path = data_path_ori_bkg / "DC4_final_530km_3_month_with_slew_1sbins_GalacticEarth_SAA.fits"
yaml_path = data_path_ori_bkg / "DC4.yaml"
source_data_path = data_path_cygnus_soft / "cygnus_soft_binned_data.hdf5"
bkg_data_path = data_path_ori_bkg / "bkg_binned_data.hdf5"
dr_path = data_path_ori_bkg / "ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284.filtered.nonsparse.binnedimaging.imagingresponse.h5"

t0 = time.perf_counter()
cosi, response, bkg, source, data, psr = prepare_cosi_plugin(
    sc_orientation_path=sc_orientation_path,
    yaml_path=yaml_path,
    source_data_path=source_data_path,
    bkg_data_path=bkg_data_path,
    dr_path=dr_path)
print(f"prepare_cosi_plugin: {time.perf_counter()-t0:.1f}s")

# Definisco il modello di sorgente puntiforme
l = 71.33496
b = 3.066917

spectrum = power_model()

source = PointSource("source",
                     l=l,
                     b=b,
                     spectral_shape=spectrum)
model = Model(source)

cache_file = psr_cache_file(l, b, sc_orientation_path, dr_path, yaml_path,
                            nside=2 * data.axes["PsiChi"].nside)

response.set_model(model)
loaded = load_psr(psr, cache_file, source)

plugins = DataList(cosi)
like = JointLikelihood(model, plugins, verbose=False)

t0 = time.perf_counter()
results, like_df = like.fit()
print(f"fit: {time.perf_counter()-t0:.1f}s")

if not loaded:
    save_psr(response, cache_file)
print(results)
print(like_df)

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
fig, ax = plt.subplots()

dat_file_path = data_path_cygnus_soft / "model/cygX1_soft_spec.dat"
dat_energies = []
dat_fluxes = []

with open(dat_file_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == "DP":
            dat_energies.append(float(parts[1]))
            dat_fluxes.append(float(parts[2]))

dat_energies = np.array(dat_energies)
dat_fluxes = np.array(dat_fluxes)

ax.scatter(dat_energies, dat_energies * dat_energies * dat_fluxes, color="black", s=6, zorder=4, label="cygX1_soft_spec.dat")

spec_opt = results.optimized_model["source"].spectrum.main.shape
k_val = spec_opt.K.value
idx_val = spec_opt.index.value
piv_val = spec_opt.piv.value

k_err = results.get_variates(spec_opt.K.path).std
idx_err = results.get_variates(spec_opt.index.path).std

fit_label = (
    f"Best fit\n"
    f"K = {k_val:.2e} ± {k_err:.2e} (at {piv_val:g} keV)\n"
    f"index = {idx_val:.2f} ± {idx_err:.2f}"
)

ax.plot(energy, energy * energy * flux_median, label=fit_label)
ax.fill_between(energy, energy * energy * flux_lo, energy * energy * flux_hi, alpha=.5, label="Best fit (errors)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Energy (keV)")
ax.set_ylabel(r"$E^2 \frac{dN}{dE}$ (keV cm$^{-2}$ s$^{-1}$)")

dat_e2_flux = dat_energies**2 * dat_fluxes
fit_e2_median = energy**2 * flux_median
fit_e2_lo = energy**2 * flux_lo
fit_e2_hi = energy**2 * flux_hi

y_positive = np.concatenate([
    dat_e2_flux[dat_e2_flux > 0],
    fit_e2_median[fit_e2_median > 0],
    fit_e2_lo[fit_e2_lo > 0],
    fit_e2_hi[fit_e2_hi > 0],
])
if y_positive.size > 0:
    ax.set_ylim(y_positive.min() * 0.7, y_positive.max() * 1.4)

_ = ax.legend()

output_dir = Path("spectra_figures")
output_dir.mkdir(parents=True, exist_ok=True)
ax.figure.savefig(output_dir / "fitted_spectrum_power_law.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Plotto lo spettro convoluto con la risposta e lo spettro fittato
binned_data_source = BinnedData(yaml_path)
binned_data_source.load_binned_data_from_hdf5(binned_data=source_data_path)

binned_energy_edges = binned_data_source.binned_data.axes["Em"].edges.value
binned_energy = np.array([])
bin_sizes = np.array([])

for i in range(len(binned_energy_edges) - 1):
    binned_energy = np.append(binned_energy, (binned_energy_edges[i + 1] + binned_energy_edges[i]) / 2)
    bin_sizes = np.append(bin_sizes, binned_energy_edges[i + 1] - binned_energy_edges[i])

expectation = response.expectation()

fig, ax = plt.subplots()

ax.stairs(expectation.project("Em").to_dense(copy=False).contents, binned_energy_edges, color="purple",
          label="Best fit convolved with response")
ax.errorbar(binned_energy, expectation.project("Em").to_dense(copy=False).contents,
            yerr=np.sqrt(expectation.project("Em").to_dense(copy=False).contents), color="purple", linewidth=0,
            elinewidth=1)
ax.stairs(binned_data_source.binned_data.project("Em").to_dense(copy=False).contents, binned_energy_edges, color="black", ls=":",
          label="Source counts")
ax.errorbar(binned_energy, binned_data_source.binned_data.project("Em").to_dense(copy=False).contents,
            yerr=np.sqrt(binned_data_source.binned_data.project("Em").to_dense(copy=False).contents), color="black", linewidth=0,
            elinewidth=1)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Energy (keV)")
ax.set_ylabel("Counts")
_ = ax.legend()

ax.figure.savefig(output_dir / "convolved_spectrum_vs_fit.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Aggiungo il fondo
expectation_bkg = bkg.expectation(copy=True)

fig, ax = plt.subplots()

ax.stairs(expectation.project("Em").to_dense(copy=False).contents + expectation_bkg.project("Em").to_dense(
            copy=False).contents, binned_energy_edges, color="purple", label="Best fit convolved with response plus background")
ax.errorbar(binned_energy,
            expectation.project("Em").to_dense(copy=False).contents + expectation_bkg.project("Em").to_dense(
            copy=False).contents, yerr=np.sqrt(
            expectation.project("Em").to_dense(copy=False).contents + expectation_bkg.project("Em").to_dense(
            copy=False).contents), color="purple", linewidth=0, elinewidth=1)
ax.stairs(data.data.project("Em").to_dense(copy=False).contents, binned_energy_edges, color="black", ls=":",
            label="Total counts")
ax.errorbar(binned_energy, data.data.project("Em").to_dense(copy=False).contents,
            yerr=np.sqrt(data.data.project("Em").to_dense(copy=False).contents), color="black", linewidth=0,
            elinewidth=1)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Energy (keV)")
ax.set_ylabel("Counts")
ax.legend()

ax.figure.savefig(output_dir / "convolved_spectrum_vs_fit_with_background.png", dpi=200, bbox_inches="tight")
plt.close(fig)

t_end = time.perf_counter()
elapsed = t_end - t_start
minutes, seconds = divmod(elapsed, 60)
print(f"Tempo totale di esecuzione: {int(minutes)}m {seconds:.2f}s ({elapsed:.2f}s)")