import time

t_start = time.perf_counter()
_last_mark = t_start


def log_step(step_name: str) -> None:
    # Misuro il tempo trascorso dall'ultimo blocco completato e aggiorno il marcatore temporale
    global _last_mark
    now = time.perf_counter()
    delta = now - _last_mark
    total = now - t_start
    print(f"[TIMER] {step_name}: {delta:.2f}s (totale: {total:.2f}s)")
    _last_mark = now


import sys
import pickle
import hashlib
import inspect
import logging
from pathlib import Path
import sys as _sys

_fit_function_dir = Path("/heronas/simeone/COSI/Cygnus X-1 soft/model")
if str(_fit_function_dir) not in _sys.path:
    _sys.path.insert(0, str(_fit_function_dir))

from fit_function import cutoff_power_model

logging.basicConfig(level=logging.INFO)  # attivo i log informativi durante l'esecuzione

from cosipy import BinnedData
from cosipy.response.FullDetectorResponse import FullDetectorResponse
from cosipy.response import BinnedThreeMLModelFolding, BinnedInstrumentResponse, BinnedThreeMLPointSourceResponse, PointSourceResponse
from cosipy.statistics import PoissonLikelihood
from cosipy.background_estimation import FreeNormBinnedBackground
from cosipy.interfaces import ThreeMLPluginInterface
from cosipy.data_io import EmCDSBinnedData
from histpy import Histogram

import astropy.units as u
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from threeML import PointSource, Model, JointLikelihood, DataList
from astromodels import Parameter

CACHE_ROOT_DIR = Path("/heronas/simeone/COSI")
CACHE_PSR_DIR = CACHE_ROOT_DIR / "cache_psr"
CACHE_BKG_DIR = CACHE_ROOT_DIR / "cache_bkg"
CACHE_SC_DIR = CACHE_ROOT_DIR / "cache_sc"
CACHE_FIT_DIR = CACHE_ROOT_DIR / "cache_fit"


def _file_stamp(path: Path) -> str:
    # Recupero percorso assoluto e timestamp di modifica per individuare il file in cache
    p = Path(path).resolve()
    return f"{p}:{p.stat().st_mtime_ns}"


def _fit_model_fingerprint(fit_fn) -> str:
    # Estraggo il codice sorgente della funzione di fit e, se disponibile, il timestamp del suo file
    parts = [f"{getattr(fit_fn, '__module__', '')}.{getattr(fit_fn, '__qualname__', str(fit_fn))}"]
    try:
        parts.append(inspect.getsource(fit_fn))
    except Exception:
        pass
    try:
        src_file = inspect.getsourcefile(fit_fn)
        if src_file:
            parts.append(_file_stamp(Path(src_file)))
    except Exception:
        pass
    return "\n".join(parts)


def get_cached_sc_path(sc_orientation_path: Path) -> Path:
    # Costruisco il percorso atteso della SpacecraftHistory in cache
    h = hashlib.sha1(_file_stamp(sc_orientation_path).encode())
    return CACHE_SC_DIR / f"sc_history_{h.hexdigest()[:10]}.pkl"


def get_cached_bkg_path(bkg_data_path: Path, yaml_path: Path, axes=("Em", "Phi", "PsiChi")) -> Path:
    # Costruisco il percorso atteso di total_bkg in cache
    h = hashlib.sha1()
    h.update(_file_stamp(bkg_data_path).encode())
    h.update(_file_stamp(yaml_path).encode())
    h.update(",".join(axes).encode())
    return CACHE_BKG_DIR / f"total_bkg_{h.hexdigest()[:10]}.h5"


def get_cached_psr_path(l: float, b: float, sc_orientation_path: Path, dr_path: Path, yaml_path: Path, nside: int) -> Path:
    # Costruisco il percorso atteso della PSR in cache
    h = hashlib.sha1()
    for p in (sc_orientation_path, dr_path, yaml_path):
        p = Path(p)
        h.update(str(p.resolve()).encode())
        h.update(str(p.stat().st_mtime_ns).encode())
    h.update(f"{l:.6f}_{b:.6f}_{nside}".encode())
    return CACHE_PSR_DIR / f"psr_l{l:.4f}_b{b:.4f}_{h.hexdigest()[:10]}.h5"


def get_cached_fit_path(
    fit_fn,
    l: float,
    b: float,
    nside: int,
    sc_orientation_path: Path,
    yaml_path: Path,
    source_data_path: Path,
    bkg_data_path: Path,
    dr_path: Path,
) -> Path:
    # Calcolo un hash univoco includendo funzione di fit, coordinate, binning e timestamp dei file di input
    h = hashlib.sha1()
    h.update(_fit_model_fingerprint(fit_fn).encode())
    h.update(f"l={l:.8f};b={b:.8f};nside={nside}".encode())
    for p in (sc_orientation_path, yaml_path, source_data_path, bkg_data_path, dr_path):
        h.update(_file_stamp(p).encode())
    fn_name = getattr(fit_fn, "__name__", "fit_fn")
    return CACHE_FIT_DIR / f"fit_{fn_name}_{h.hexdigest()[:12]}.pkl"


def attach_cached_psr(psr_template, cache_path: Path, point_source: PointSource):
    # Collego la PSR pre-calcolata alle copie generate durante il folding
    if not cache_path.exists():
        raise FileNotFoundError(f"PSR in cache non trovata ({cache_path}). Esegui prima download_and_binning.py")

    loaded_psr = PointSourceResponse.open(cache_path)
    coord = point_source.position.sky_coord.copy()
    original_copy = psr_template.copy

    def cached_copy():
        new = original_copy()
        new._psr = loaded_psr
        new._last_convolved_source_skycoord = coord.copy()
        return new

    psr_template.copy = cached_copy
    print(f"PSR caricata da {cache_path}")


def prepare_cosi_plugin(
    sc_orientation_path: Path,
    yaml_path: Path,
    source_data_path: Path,
    bkg_data_path: Path,
    dr_path: Path,
):
    # Carico la storia del satellite già preparata in cache
    sc_cache = get_cached_sc_path(sc_orientation_path)
    if not sc_cache.exists():
        raise FileNotFoundError(f"SpacecraftHistory non trovata in cache ({sc_cache}). Esegui prima download_and_binning_data.")
    with open(sc_cache, "rb") as f:
        sc_orientation = pickle.load(f)
    print(f"SpacecraftHistory caricata da {sc_cache}")

    # Carico i dati binnati della sorgente già prodotti dal binning
    source_binned = BinnedData(yaml_path)
    source_binned.load_binned_data_from_hdf5(binned_data=source_data_path)

    # Carico total_bkg già proiettato e messo in cache
    bkg_cache = get_cached_bkg_path(bkg_data_path, yaml_path, axes=("Em", "Phi", "PsiChi"))
    if not bkg_cache.exists():
        raise FileNotFoundError(f"total_bkg non trovato in cache ({bkg_cache}). Esegui prima download_and_binning_data.")
    total_bkg = Histogram.open(bkg_cache)
    print(f"total_bkg caricato da {bkg_cache}")

    single_bkg_fit = True
    bkg_dist = {"total_bkg": total_bkg + sys.float_info.min}

    # Sommo conteggi sorgente e fondo per formare il dataset totale
    data = EmCDSBinnedData(source_binned.binned_data.project("Em", "Phi", "PsiChi") + total_bkg)

    # Inizializzo modello di fondo, risposta strumentale e plugin 3ML
    bkg = FreeNormBinnedBackground(bkg_dist, sc_history=sc_orientation, copy=False)
    dr = FullDetectorResponse.open(str(dr_path))
    instrument_response = BinnedInstrumentResponse(dr, data)
    psr = BinnedThreeMLPointSourceResponse(
        data=data,
        instrument_response=instrument_response,
        sc_history=sc_orientation,
        energy_axis=dr.axes["Ei"],
        polarization_axis=dr.axes["Pol"] if "Pol" in dr.axes.labels else None,
        nside=2 * data.axes["PsiChi"].nside,
    )
    response = BinnedThreeMLModelFolding(data=data, point_source_response=psr)
    like_fun = PoissonLikelihood(data, response, bkg)
    cosi = ThreeMLPluginInterface("cosi", like_fun, response, bkg)

    for bkg_label in bkg_dist.keys():
        cosi.bkg_parameter[bkg_label] = Parameter(
            bkg_label,
            1,
            min_value=0,
            max_value=100 if single_bkg_fit else 20,
            delta=0.05,
            unit=u.Hz,
        )

    return cosi, response, bkg, source_binned, data, psr


data_path_ori_bkg = Path("/heronas/simeone/COSI/ori_bkg_yaml")
data_path_cygnus_soft = Path("/heronas/simeone/COSI/Cygnus X-1 soft")

sc_orientation_path = data_path_ori_bkg / "DC4_final_530km_3_month_with_slew_1sbins_GalacticEarth_SAA.fits"
yaml_path = data_path_ori_bkg / "DC4.yaml"
source_data_path = data_path_cygnus_soft / "cygnus_soft_binned_data.hdf5"
bkg_data_path = data_path_ori_bkg / "bkg_binned_data.hdf5"
dr_path = data_path_ori_bkg / "ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284.filtered.nonsparse.binnedimaging.imagingresponse.h5"

cosi, response, bkg, source_binned, data, psr = prepare_cosi_plugin(
    sc_orientation_path=sc_orientation_path,
    yaml_path=yaml_path,
    source_data_path=source_data_path,
    bkg_data_path=bkg_data_path,
    dr_path=dr_path,
)
log_step("Preparazione plugin, dati e risposta strumentale")

# Definisco il modello di sorgente puntiforme con cutoff powerlaw + powerlaw
l = 71.33496
b = 3.066917
fit_fn = cutoff_power_model

point_source = PointSource("source", l=l, b=b, spectral_shape=fit_fn())
model = Model(point_source)

nside = 2 * data.axes["PsiChi"].nside
psr_cache = get_cached_psr_path(
    l=l,
    b=b,
    sc_orientation_path=sc_orientation_path,
    dr_path=dr_path,
    yaml_path=yaml_path,
    nside=nside,
)

response.set_model(model)
attach_cached_psr(psr, psr_cache, point_source)
log_step("Caricamento PSR da cache e collegamento modello")

fit_cache_path = get_cached_fit_path(
    fit_fn=fit_fn,
    l=l,
    b=b,
    nside=nside,
    sc_orientation_path=sc_orientation_path,
    yaml_path=yaml_path,
    source_data_path=source_data_path,
    bkg_data_path=bkg_data_path,
    dr_path=dr_path,
)

energy = np.geomspace(100 * u.keV, 10 * u.MeV).to_value(u.keV)

if fit_cache_path.exists():
    # Carico direttamente i risultati del fit e la banda d'errore salvati in cache
    with open(fit_cache_path, "rb") as f:
        fit_payload = pickle.load(f)
    print(f"Risultato del fit caricato da {fit_cache_path} (salto like.fit e propagazione errori)")

    energy = fit_payload["energy"]
    flux_median = fit_payload["flux_median"]
    flux_best = fit_payload.get("flux_best", flux_median)
    flux_cutoff = fit_payload.get("flux_cutoff", None)
    flux_tail = fit_payload.get("flux_tail", None)
    flux_lo = fit_payload["flux_lo"]
    flux_hi = fit_payload["flux_hi"]
    piv_val = fit_payload.get("piv_val", 100.0)
    k1_val = fit_payload["k1_val"]
    k1_err = fit_payload["k1_err"]
    idx1_val = fit_payload["idx1_val"]
    idx1_err = fit_payload["idx1_err"]
    xc_val = fit_payload["xc_val"]
    xc_err = fit_payload["xc_err"]
    k2_val = fit_payload["k2_val"]
    k2_err = fit_payload["k2_err"]
    idx2_val = fit_payload["idx2_val"]
    idx2_err = fit_payload["idx2_err"]
    flux_ref_val = fit_payload.get("flux_ref_val", float(np.interp(piv_val, energy, flux_best)))
    flux_ref_lo = fit_payload.get("flux_ref_lo", float(np.interp(piv_val, energy, flux_lo)))
    flux_ref_hi = fit_payload.get("flux_ref_hi", float(np.interp(piv_val, energy, flux_hi)))
    int_flux_val = fit_payload.get("int_flux_val", None)
    int_flux_lo = fit_payload.get("int_flux_lo", None)
    int_flux_hi = fit_payload.get("int_flux_hi", None)
    int_flux_unit = fit_payload.get("int_flux_unit", "erg / (s cm2)")
    exp_em = fit_payload["exp_em"]
    exp_bkg_em = fit_payload["exp_bkg_em"]
    log_step("Caricamento fit ed errori da cache")
else:
    # Eseguo il fit statistico e la propagazione dell'errore per questa configurazione
    like = JointLikelihood(model, DataList(cosi), verbose=False)
    results, like_df = like.fit()
    log_step("Fit statistico (like.fit)")
    print(results)
    print(like_df)

    results = like.results
    print(results.display())

    spec_opt = results.optimized_model["source"].spectrum.main.shape
    comp_cutoff, comp_tail = spec_opt.functions

    # Valuto in modo deterministico lo spettro totale ottimizzato e le due singole componenti
    flux_best = np.asarray(spec_opt(energy), dtype=float)
    flux_cutoff = np.asarray(comp_cutoff(energy), dtype=float)
    flux_tail = np.asarray(comp_tail(energy), dtype=float)

    # Filtro eventuali parametri bloccati sul bordo con varianza non fisica prima della propagazione
    parameters = {}
    for par in results.optimized_model["source"].parameters.values():
        if par.free:
            var = results.get_variates(par.path)
            at_lower = par.min_value is not None and np.isclose(par.value, par.min_value, rtol=1e-2, atol=1e-14)
            at_upper = par.max_value is not None and np.isclose(par.value, par.max_value, rtol=1e-2, atol=1e-14)
            if (at_lower or at_upper) and var.std > 10.0 * max(abs(par.value), 1e-12):
                continue
            parameters[par.name] = var

    results_err = results.propagate(spec_opt.evaluate_at, **parameters)
    print(results.optimized_model["source"])

    # Calcolo la banda d'errore al 68% attorno al best fit
    flux_lo = np.zeros_like(energy)
    flux_median = np.zeros_like(energy)
    flux_hi = np.zeros_like(energy)

    for i, e in enumerate(energy):
        flux = results_err(e)
        flux_median[i] = flux.median
        flux_lo[i], flux_hi[i] = flux.equal_tail_interval(cl=0.68)
    log_step("Propagazione errori e intervalli al 68%")

    xc_par = spec_opt.xc if hasattr(spec_opt, "xc") else spec_opt.xc_1
    piv_val = float(spec_opt.piv_1.value)
    k1_val = float(spec_opt.K_1.value)
    idx1_val = float(spec_opt.index_1.value)
    xc_val = float(xc_par.value)
    k2_val = float(spec_opt.K_2.value)
    idx2_val = float(spec_opt.index_2.value)

    k1_err = float(results.get_variates(spec_opt.K_1.path).std)
    idx1_err = float(results.get_variates(spec_opt.index_1.path).std)
    xc_err = float(results.get_variates(xc_par.path).std) if xc_par.free else 0.0
    k2_err = float(results.get_variates(spec_opt.K_2.path).std)
    idx2_err = float(results.get_variates(spec_opt.index_2.path).std)

    # Valuto il flusso differenziale totale esatto all'energia di pivot (100 keV)
    flux_ref_val = float(spec_opt(piv_val))
    flux_ref = results_err(piv_val)
    flux_ref_lo, flux_ref_hi = [float(x) for x in flux_ref.equal_tail_interval(cl=0.68)]

    # Calcolo il flusso integrato nella banda 100 keV - 10 MeV sul modello deterministico ottimizzato
    flux_unit = u.erg / (u.cm**2 * u.s)
    e_int_grid = np.geomspace(100.0, 10000.0, 2000)
    kev_to_erg = (1.0 * u.keV).to_value(u.erg)
    int_flux_val = float(np.trapezoid(e_int_grid * spec_opt(e_int_grid), e_int_grid) * kev_to_erg)
    int_flux_lo = float(np.trapezoid(e_int_grid * np.interp(e_int_grid, energy, flux_lo), e_int_grid) * kev_to_erg)
    int_flux_hi = float(np.trapezoid(e_int_grid * np.interp(e_int_grid, energy, flux_hi), e_int_grid) * kev_to_erg)
    int_flux_unit = str(flux_unit)

    expectation = response.expectation()
    expectation_bkg = bkg.expectation(copy=True)
    exp_em = np.asarray(expectation.project("Em").to_dense(copy=False).contents)
    exp_bkg_em = np.asarray(expectation_bkg.project("Em").to_dense(copy=False).contents)

    fit_payload = {
        "energy": energy,
        "flux_best": flux_best,
        "flux_cutoff": flux_cutoff,
        "flux_tail": flux_tail,
        "flux_median": flux_median,
        "flux_lo": flux_lo,
        "flux_hi": flux_hi,
        "piv_val": piv_val,
        "k1_val": k1_val,
        "k1_err": k1_err,
        "idx1_val": idx1_val,
        "idx1_err": idx1_err,
        "xc_val": xc_val,
        "xc_err": xc_err,
        "k2_val": k2_val,
        "k2_err": k2_err,
        "idx2_val": idx2_val,
        "idx2_err": idx2_err,
        "flux_ref_val": flux_ref_val,
        "flux_ref_lo": flux_ref_lo,
        "flux_ref_hi": flux_ref_hi,
        "int_flux_val": int_flux_val,
        "int_flux_lo": int_flux_lo,
        "int_flux_hi": int_flux_hi,
        "int_flux_unit": int_flux_unit,
        "exp_em": exp_em,
        "exp_bkg_em": exp_bkg_em,
    }
    try:
        CACHE_FIT_DIR.mkdir(parents=True, exist_ok=True)
        with open(fit_cache_path, "wb") as f:
            pickle.dump(fit_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Risultato del fit salvato in {fit_cache_path}")
    except Exception as e:
        print("Salvataggio cache fit fallito:", type(e).__name__, e)

# Preparo gli assi energetici binnati e i conteggi per i plot successivi
binned_energy_edges = source_binned.binned_data.axes["Em"].edges.value
binned_energy = 0.5 * (binned_energy_edges[:-1] + binned_energy_edges[1:])
bin_sizes = np.diff(binned_energy_edges)

src_em = source_binned.binned_data.project("Em").to_dense(copy=False).contents
tot_em = data.data.project("Em").to_dense(copy=False).contents
total_model_em = exp_em + exp_bkg_em

# Plotto lo spettro fittato con le due componenti separate
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

ax.scatter(dat_energies, dat_energies**2 * dat_fluxes, color="black", s=6, zorder=4, label="cygX1_soft_spec.dat")

flux_ref_sym_err = 0.5 * ((flux_ref_hi - flux_ref_val) + (flux_ref_val - flux_ref_lo))
int_flux_line = (
    f"F(100 keV–10 MeV) = {int_flux_val:.2e} ({int_flux_unit})\n"
    if int_flux_val is not None
    else ""
)
fit_label = (
    f"Best fit (Cutoff PL + PL)\n"
    f"dN/dE = {flux_ref_val:.2e} ± {flux_ref_sym_err:.2e} ph/(cm² s keV) (at {piv_val:g} keV)\n"
    f"{int_flux_line}"
    f"K_th = {k1_val:.2e} ± {k1_err:.2e}\n"
    f"\u0393_th = {idx1_val:.2f} ± {idx1_err:.2f}\n"
    f"E_c = {xc_val:.0f} ± {xc_err:.0f} keV\n"
    f"K_tail = {k2_val:.2e} ± {k2_err:.2e}\n"
    f"\u0393_tail = {idx2_val:.2f} ± {idx2_err:.2f}"
)

ax.plot(energy, energy**2 * flux_best, color="royalblue", lw=2, zorder=3, label=fit_label)
ax.fill_between(energy, energy**2 * flux_lo, energy**2 * flux_hi, alpha=0.25, color="royalblue", label="Best fit (68% errors)")

if flux_cutoff is not None:
    ax.plot(energy, energy**2 * flux_cutoff, color="crimson", ls="--", lw=1.5, label="Cutoff powerlaw (thermal)")
if flux_tail is not None:
    ax.plot(energy, energy**2 * flux_tail, color="darkorange", ls=":", lw=1.8, label="Powerlaw (tail)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Energy (keV)")
ax.set_ylabel(r"$E^2 \frac{dN}{dE}$ (keV cm$^{-2}$ s$^{-1}$)")

dat_e2_flux = dat_energies**2 * dat_fluxes
fit_e2_best = energy**2 * flux_best
fit_e2_lo = energy**2 * flux_lo
fit_e2_hi = energy**2 * flux_hi

y_ref = np.concatenate([
    dat_e2_flux[dat_e2_flux > 0],
    fit_e2_best[fit_e2_best > 0],
])
y_upper = np.concatenate([
    y_ref,
    fit_e2_hi[fit_e2_hi > 0],
])
if y_ref.size > 0:
    y_min = y_ref.min() * 0.25
    if y_upper.size > 0:
        ax.set_ylim(y_min, y_upper.max() * 1.5)

_ = ax.legend(fontsize=8)

output_dir = Path("spectra_figures")
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / "fitted_spectrum_power_law.png", dpi=200, bbox_inches="tight")
plt.close(fig)
log_step("Plot spettro fittato con componenti (fitted_spectrum_power_law.png)")

# Plotto lo spettro convoluto con la risposta e i conteggi della sorgente
fig, ax = plt.subplots()
ax.stairs(exp_em, binned_energy_edges, color="royalblue", label="Best fit convolved with response")
ax.errorbar(binned_energy, exp_em, yerr=np.sqrt(exp_em), color="royalblue", linewidth=0, elinewidth=1)
ax.stairs(src_em, binned_energy_edges, color="black", ls=":", label="Source counts")
ax.errorbar(binned_energy, src_em, yerr=np.sqrt(src_em), color="black", linewidth=0, elinewidth=1)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Energy (keV)")
ax.set_ylabel("Counts")
_ = ax.legend()
ax.figure.savefig(output_dir / "convolved_spectrum_vs_fit.png", dpi=200, bbox_inches="tight")
plt.close(fig)
log_step("Plot spettro convoluto vs sorgente (convolved_spectrum_vs_fit.png)")

# Plotto lo spettro convoluto più il fondo contro i conteggi totali
fig, ax = plt.subplots()
ax.stairs(total_model_em, binned_energy_edges, color="royalblue", label="Best fit convolved with response plus background")
ax.errorbar(binned_energy, total_model_em, yerr=np.sqrt(total_model_em), color="royalblue", linewidth=0, elinewidth=1)
ax.stairs(tot_em, binned_energy_edges, color="black", ls=":", label="Total counts")
ax.errorbar(binned_energy, tot_em, yerr=np.sqrt(tot_em), color="black", linewidth=0, elinewidth=1)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Energy (keV)")
ax.set_ylabel("Counts")
ax.legend()
ax.figure.savefig(output_dir / "convolved_spectrum_vs_fit_with_background.png", dpi=200, bbox_inches="tight")
plt.close(fig)
log_step("Plot spettro convoluto + fondo vs totali (convolved_spectrum_vs_fit_with_background.png)")

elapsed = time.perf_counter() - t_start
minutes, seconds = divmod(elapsed, 60)
print(f"Tempo totale di esecuzione: {int(minutes)}m {seconds:.2f}s ({elapsed:.2f}s)")