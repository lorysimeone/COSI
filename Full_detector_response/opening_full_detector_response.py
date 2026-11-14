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
from cosipy.spacecraftfile import SpacecraftFile
from cosipy import test_data
from cosipy.util import fetch_wasabi_file
from histpy import Histogram
import gc

from threeML import Model, Powerlaw

from cosipy.response import FullDetectorResponse

import shutil
import os

with FullDetectorResponse.open("/heronas/simeone/COSI/Full_detector_response/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.h5") as response:

    print(repr(response))