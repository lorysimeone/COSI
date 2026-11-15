import os
from pathlib import Path
import h5py
from cosipy.response import FullDetectorResponse

# 1. Definisco i percorsi
data_dir = Path("./")
ori_path = data_dir / "20280301_3_month_with_orbital_info.fits"
rsp_filename = "ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284.filtered.nonsparse.binnedimaging.imagingresponse.h5"
rsp_path = data_dir / rsp_filename

# 2. Controllo e patch di sicurezza per l'attributo mancante 'TYPE' negli assi HDF5
print("Verifica e patch della struttura HDF5...")
with h5py.File(rsp_path, "a") as f:
    if "axes" in f:
        for axis_name in f["axes"]:
            axis = f["axes"][axis_name]
            if "TYPE" not in axis.attrs:
                # Imposto un tipo di default se assente per soddisfare cosipy
                axis.attrs["TYPE"] = "generic"

# 3. Apro e verifico la risposta
print("Apro e verifico la risposta...")
with FullDetectorResponse.open(rsp_path) as response:
    print("\nRisposta del rivelatore caricata con successo!")
    print(repr(response))