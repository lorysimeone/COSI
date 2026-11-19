import matplotlib
matplotlib.use("Agg")            # backend senza interfaccia grafica: va PRIMA di importare pyplot
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord
from scoords import SpacecraftFrame
from cosipy.response import FullDetectorResponse

import numpy as np
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time

import matplotlib.pyplot as ply

from mhealpy import HealpixMap, HealpixBase
import pandas as pd
from pathlib import Path

from scoords import Attitude, SpacecraftFrame
from cosipy.response import FullDetectorResponse
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy import test_data
from cosipy.util import fetch_wasabi_file
from cosipy.response import PointSourceResponse
from histpy import Histogram
import gc

from threeML import Model, Powerlaw

from cosipy.response import FullDetectorResponse

import shutil
import os

response_path = '/heronas/simeone/COSI/Full_detector_response/dc4_data/ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284.filtered.nonsparse.binnedimaging.imagingresponse.h5'
ori_path = '/heronas/simeone/COSI/Full_detector_response/dc4_data/DC4_final_530km_3_month_with_slew_1sbins_GalacticEarth_SAA.fits'

# Leggo l'orientazione
ori = SpacecraftHistory.open(ori_path)

scatt_map = ori.get_scatt_map(nside = 16, earth_occ = False) # traccia il tempo speso in una data orientazione

coord = SkyCoord.from_name('Crab').galactic

# Ottengo la psr convolvendo la scatt map con la risposta del rilevatore
with FullDetectorResponse.open(response_path) as response:
    psr = response.get_point_source_response(coord = coord, scatt_map = scatt_map)

    psichi_slice = psr.slice[{'Ei': 4, 'Phi': 4}].project('PsiChi')

    ax, plot = psichi_slice.plot(ax_kw={'coord': 'G'})
    ax.scatter([coord.l.deg], [coord.b.deg], transform=ax.get_transform('world'), marker='x', color='red')
    ax.figure.savefig("scatt_map_galactic.png", dpi=200, bbox_inches="tight")
    plt.close(ax.figure)

    ax, plot = psichi_slice.plot(ax_kw={'coord': 'C'})
    ax.scatter([coord.icrs.ra.deg], [coord.icrs.dec.deg], transform=ax.get_transform('world'), marker='x', color='red')
    ax.figure.savefig("scatt_map_RA/DEC.png", dpi=200, bbox_inches="tight")
    plt.close(ax.figure)
