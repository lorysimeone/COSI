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

psr = PointSourceResponse.open("psr_crab_DC4.h5")

print(f'Unit = {psr.unit}')

# Convolvo il modello spettrale teorico con la PSR

index = -2.2
K = 10**-3 / u.cm / u.cm / u.s / u.keV
piv = 100 * u.keV

spectrum = Powerlaw()
spectrum.index.value = index
spectrum.K.value = K.value
spectrum.piv.value = piv.value
spectrum.K.unit = K.unit
spectrum.piv.unit = piv.unit

expectation = psr.get_expectation(spectrum)
ax, plot = expectation.project('Em').plot()

ax.set_ylabel('Expected counts')
ax.figure.savefig("conteggi_attesi_indice_2.2.png", dpi=200, bbox_inches="tight")
plt.close(ax.figure)

spectrum.index.value = -1

expectation = psr.get_expectation(spectrum)

ax, plot = expectation.project('Em').plot()

ax.set_ylabel('Expected counts')
ax.figure.savefig("conteggi_attesi_indice_-1.png", dpi=200, bbox_inches="tight")
plt.close(ax.figure)