import time

t_start = time.perf_counter()

import sys
import pickle
import hashlib
import logging
from pathlib import Path

from astromodels import Powerlaw
from cosipy import BinnedData
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.response.FullDetectorResponse import FullDetectorResponse
from cosipy.response import BinnedInstrumentResponse, BinnedThreeMLPointSourceResponse, BinnedThreeMLModelFolding
from cosipy.data_io import EmCDSBinnedData
from cosipy.util import fetch_wasabi_file
from histpy import Histogram
from threeML import PointSource, Model

logging.basicConfig(level=logging.INFO)  # attivo i log informativi durante download, binning e preparazione

CACHE_ROOT_DIR = Path("/home/lorysimeone/COSI")
CACHE_PSR_DIR = CACHE_ROOT_DIR / "cache_psr"
CACHE_BKG_DIR = CACHE_ROOT_DIR / "cache_bkg"
CACHE_SC_DIR = CACHE_ROOT_DIR / "cache_sc"


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


def psr_cache_file(l, b, sc_orientation_path, dr_path, yaml_path, nside) -> Path:
    # Calcolo un hash univoco in base a coordinate, nside e timestamp dei file di input
    h = hashlib.sha1()
    for p in (sc_orientation_path, dr_path, yaml_path):
        p = Path(p)
        h.update(str(p.resolve()).encode())
        h.update(str(p.stat().st_mtime_ns).encode())
    h.update(f"{l:.6f}_{b:.6f}_{nside}".encode())
    return CACHE_PSR_DIR / f"psr_l{l:.4f}_b{b:.4f}_{h.hexdigest()[:10]}.h5"


def load_or_build_sc_history(sc_orientation_path: Path) -> SpacecraftHistory:
    # Verifico se esiste già una copia in cache valida per l'attuale timestamp del FITS
    cache_path = sc_cache_file(sc_orientation_path)
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                sc_orientation = pickle.load(f)
            print(f"SpacecraftHistory caricata da {cache_path}")
            return sc_orientation
        except Exception as e:
            # Se il caricamento fallisce, rileggo il file FITS originale
            print("Caricamento cache SpacecraftHistory fallito, rileggo FITS:", type(e).__name__, e)

    sc_orientation = SpacecraftHistory.open(sc_orientation_path)
    try:
        CACHE_SC_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(sc_orientation, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"SpacecraftHistory salvata in {cache_path}")
    except Exception as e:
        print("Salvataggio cache SpacecraftHistory fallito:", type(e).__name__, e)
    return sc_orientation


def load_or_build_total_bkg(bkg_data_path: Path, yaml_path: Path, axes=("Em", "Phi", "PsiChi")):
    # Verifico se esiste già l'istogramma proiettato in cache per i file correnti
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
    total_bkg = bkg_binned.binned_data.project(*axes)
    try:
        CACHE_BKG_DIR.mkdir(parents=True, exist_ok=True)
        total_bkg.write(cache_path, overwrite=True)
        print(f"total_bkg salvato in {cache_path}")
    except Exception as e:
        print("Salvataggio cache total_bkg fallito:", type(e).__name__, e)
    return total_bkg


def build_and_cache_psr(
    l: float,
    b: float,
    sc_orientation: SpacecraftHistory,
    source_binned: BinnedData,
    total_bkg,
    sc_orientation_path: Path,
    dr_path: Path,
    yaml_path: Path,
):
    # Preparo l'interfaccia dati e la risposta strumentale per calcolare e mettere in cache la PSR
    bkg_with_eps = total_bkg + sys.float_info.min
    data = EmCDSBinnedData(source_binned.binned_data.project("Em", "Phi", "PsiChi") + bkg_with_eps)
    nside = 2 * data.axes["PsiChi"].nside
    cache_path = psr_cache_file(l, b, sc_orientation_path, dr_path, yaml_path, nside=nside)

    if cache_path.exists():
        print(f"PSR già presente in cache: {cache_path}")
        return cache_path

    dr = FullDetectorResponse.open(str(dr_path))
    instrument_response = BinnedInstrumentResponse(dr, data)
    psr_builder = BinnedThreeMLPointSourceResponse(
        data=data,
        instrument_response=instrument_response,
        sc_history=sc_orientation,
        energy_axis=dr.axes["Ei"],
        polarization_axis=dr.axes["Pol"] if "Pol" in dr.axes.labels else None,
        nside=nside,
    )
    response = BinnedThreeMLModelFolding(data=data, point_source_response=psr_builder)
    temp_source = PointSource("source", l=l, b=b, spectral_shape=Powerlaw())
    response.set_model(Model(temp_source))
    _ = response.expectation()

    sr = getattr(response, "_source_responses", {}).get("source")
    if sr is None or sr._psr is None:
        print("PSR non calcolata durante la pre-elaborazione, salto il salvataggio")
        return cache_path

    try:
        CACHE_PSR_DIR.mkdir(parents=True, exist_ok=True)
        sr._psr.write(cache_path, overwrite=True)
        print(f"PSR salvata in {cache_path}")
    except Exception as e:
        print("Salvataggio cache PSR fallito:", type(e).__name__, e)
    return cache_path


data_path_ori_bkg = Path("/heronas/simeone/COSI/ori_bkg_yaml")
data_path_cygnus_soft = Path("/heronas/simeone/COSI/Cygnus X-1 soft")

sc_orientation_path = data_path_ori_bkg / "DC4_final_530km_3_month_with_slew_1sbins_GalacticEarth_SAA.fits"
yaml_path = data_path_ori_bkg / "DC4.yaml"
source_data_path = data_path_cygnus_soft / "cygnus_soft_binned_data.hdf5"
bkg_data_path = data_path_ori_bkg / "bkg_binned_data.hdf5"
dr_path = data_path_ori_bkg / "ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284.filtered.nonsparse.binnedimaging.imagingresponse.h5"

# Scarico i dati non binnati della sorgente se non già presenti localmente
unbinned_name = "cygX1_soft_3months_unbinned_data_filtered_with_SAAcut.fits.gz"
unbinned_path = data_path_cygnus_soft / unbinned_name
fetch_wasabi_file(f"COSI-SMEX/DC3/Data/Sources/{unbinned_name}", unbinned_path)

# Eseguo il binning dei dati della sorgente
binned_output_prefix = data_path_cygnus_soft / "cygnus_soft_binned_data"
binned_output_hdf5 = Path(f"{binned_output_prefix}.hdf5")
if not binned_output_hdf5.exists():
    analysis = BinnedData(yaml_path)
    analysis.get_binned_data(
        unbinned_data=unbinned_path,
        output_name=binned_output_prefix,
        make_binning_plots=False,
        show_plots=False,
    )
else:
    print(f"Dati binnati già presenti in {binned_output_hdf5}, salto il binning")

# Preparo e metto in cache la storia del satellite e l'istogramma proiettato del fondo
sc_orientation = load_or_build_sc_history(sc_orientation_path)
total_bkg = load_or_build_total_bkg(bkg_data_path, yaml_path, axes=("Em", "Phi", "PsiChi"))

# Carico i dati binnati della sorgente e pre-calcolo la PSR per le coordinate della sorgente
source_binned = BinnedData(yaml_path)
source_binned.load_binned_data_from_hdf5(binned_data=source_data_path)

l = 71.33496
b = 3.066917
_ = build_and_cache_psr(
    l=l,
    b=b,
    sc_orientation=sc_orientation,
    source_binned=source_binned,
    total_bkg=total_bkg,
    sc_orientation_path=sc_orientation_path,
    dr_path=dr_path,
    yaml_path=yaml_path,
)

elapsed = time.perf_counter() - t_start
minutes, seconds = divmod(elapsed, 60)
print(f"Preparazione completata in: {int(minutes)}m {seconds:.2f}s ({elapsed:.2f}s)")