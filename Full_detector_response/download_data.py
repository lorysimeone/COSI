import os
from pathlib import Path
from cosipy.response import FullDetectorResponse
from cosipy.util import fetch_wasabi_file

# 1. Definisco i percorsi
data_dir = Path("./")
ori_path = data_dir / "20280301_3_month_with_orbital_info.fits"
rsp_filename = "ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284.filtered.nonsparse.binnedimaging.imagingresponse.h5"
rsp_path = data_dir / rsp_filename

# 2. Scarico i file da Wasabi
print("Scaricamento file di orientamento...")
fetch_wasabi_file(
    "COSI-SMEX/DC3/Data/Orientation/20280301_3_month_with_orbital_info.fits",
    ori_path,
)

print("Scaricamento nuova risposta del rivelatore (DC4)...")
fetch_wasabi_file(f"COSI-SMEX/DC4/Data/Responses/{rsp_filename}", rsp_path)

# 3. Apro e verifico la risposta
with FullDetectorResponse.open(rsp_path) as response:
    print("\nRisposta del rivelatore caricata con successo!")
    print(repr(response))