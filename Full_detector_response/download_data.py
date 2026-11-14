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

data_dir = Path("") # Current directory by default. Modify if you can want a different path

ori_path = data_dir/"20280301_3_month_with_orbital_info.fits"
response_path = data_dir/"SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.h5"

# download orientation file ~684.38 MB
fetch_wasabi_file("COSI-SMEX/DC3/Data/Orientation/20280301_3_month_with_orbital_info.fits", ori_path, checksum = '5e69bc1d55fab9390f90635690f62896')

# download response file ~596.06 MB
fetch_wasabi_file("COSI-SMEX/cosipy_tutorials/Data/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.h5", checksum = 'eb72400a1279325e9404110f909c7785')

