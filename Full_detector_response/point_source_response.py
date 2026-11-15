import matplotlib
matplotlib.use("Agg")            # backend senza interfaccia grafica: va PRIMA di importare pyplot
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord
from scoords import SpacecraftFrame
from cosipy.response import FullDetectorResponse

response_path = '/heronas/simeone/COSI/Full_detector_response/dc4_data/ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284.filtered.nonsparse.binnedimaging.imagingresponse.h5'
ori_path = '/heronas/simeone/COSI/Full_detector_response/dc4_data/dc4_data/DC4_final_530km_3_month_with_slew_1sbins_GalacticEarth_SAA.fits'

# Leggo l'orientazione
ori = SpacecraftHistory.open(ori_path)

# Definisco le coordinate del target (Crab)
target_coord = SkyCoord(184.5551, -05.7877, unit = "deg", frame = "galactic")

# Ottengo la mappa dei tempi di permanenza
with FullDetectorResponse.open(response_path) as response:
    dwell_time_map = ori.get_dwell_map(target_coord = target_coord, base = response) # Calcola per quanti secondi la sorgente resta posizionata in ciascuna direzione del campo di vista del rivelatore durante l'intervallo temporale dell'osservazione

    # Ottengo la point spread response (PSR) convolvendo la mappa di rilevazione con la risposta del rilevatore
    psr = response.get_point_source_response(exposure_map = dwell_time_map, coord = target_coord)
    print(f'Unit = {psr.unit}')
