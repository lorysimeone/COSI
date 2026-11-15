import numpy as np
import astropy.units as u
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time

import matplotlib.pyplot as plt
import matplotlib.pyplot as ply

from mhealpy import HealpixMap, HealpixBase
import pandas as pd
from pathlib import Path

from scoords import Attitude, SpacecraftFrame
from cosipy.response import FullDetectorResponse
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy import test_data
from cosipy.util import fetch_wasabi_file
from histpy import Histogram
import gc

from threeML import Model, Powerlaw

from cosipy.response import FullDetectorResponse

import shutil
import os

response_path = '/heronas/simeone/COSI/Full_detector_response/dc4_data/ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284.filtered.nonsparse.binnedimaging.imagingresponse.h5'
with FullDetectorResponse.open(response_path) as response: # apre il file, legge l'header e chiude il file

    print(f"NSIDE = {response.nside}")
    print(f"SCHEME = {response.scheme}")
    print(f"NPIX = {response.npix}")
    print(f"Pixel size = {np.sqrt(response.pixarea()).to(u.deg):.2f}")

    print(f"Pixel 35 centered at {response.pix2skycoord(0)}")
    drm = response[35] # ottengo la risposta di uno specifico pixel
    print(drm)

    print("Matrix for a specific direction: ")
    drm = response.get_interp_response(SkyCoord(lon=58 * u.deg, lat=65 * u.deg, frame=SpacecraftFrame())) # ottengo la matrice in una specifica direzione
    print(drm)

