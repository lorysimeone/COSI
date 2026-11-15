import matplotlib
matplotlib.use("Agg")            # backend senza interfaccia grafica: va PRIMA di importare pyplot
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord
from scoords import SpacecraftFrame
from cosipy.response import FullDetectorResponse

response_path = '/heronas/simeone/COSI/Full_detector_response/dc4_data/ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284.filtered.nonsparse.binnedimaging.imagingresponse.h5'

with FullDetectorResponse.open(response_path) as response: # apre il file, legge l'header e chiude il file

    # Matrice di risposta interpolata per la direzione on-axis (lat = 90°)
    drm = response.get_interp_response(
        SkyCoord(lon=0 * u.deg, lat=90 * u.deg, frame=SpacecraftFrame()))

    # Proiezione sugli assi Ei ed Em e disegno
    ax, plot = drm.get_spectral_response().plot()
    ax.figure.savefig("spectral_response.png", dpi=200, bbox_inches="tight")
    plt.close(ax.figure)

    # Proiezione su Ei dell'area efficace
    ax, plot = drm.get_effective_area().plot()
    ax.set_ylabel(f'Aeff [{drm.unit}]')
    ax.figure.savefig("effective_area.png", dpi=200, bbox_inches="tight")
    plt.close(ax.figure)

