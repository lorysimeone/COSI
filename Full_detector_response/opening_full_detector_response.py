import os
from pathlib import Path
import h5py
from cosipy.response import FullDetectorResponse

# Definisco i percorsi di lavoro
data_dir = Path("./")
ori_path = data_dir / "20280301_3_month_with_orbital_info.fits"
rsp_filename = "ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284.filtered.nonsparse.binnedimaging.imagingresponse.h5"
rsp_path = data_dir / rsp_filename

# Definisco la funzione ricorsiva per applicare la patch a ogni elemento HDF5
print("Verifica e patch ricorsiva della struttura HDF5...")
def patch_node(name, obj):
    if hasattr(obj, 'attrs') and 'TYPE' not in obj.attrs:
        obj.attrs['TYPE'] = 'generic'

with h5py.File(rsp_path, "a") as f:
    # Applico la patch ricorsivamente su tutto il file e sulla radice
    f.visititems(patch_node)
    if 'TYPE' not in f.attrs:
        f.attrs['TYPE'] = 'generic'

# Apro e verifico la risposta del rivelatore
print("Apro e verifico la risposta...")
with FullDetectorResponse.open(rsp_path) as response:
    print("\nRisposta del rivelatore caricata con successo!")
    print(repr(response))