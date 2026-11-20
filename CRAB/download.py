import logging
logging.basicConfig(level=logging.INFO) # serve solo per avere le informazioni durante i caricamenti

from cosipy import BinnedData
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.response.FullDetectorResponse import FullDetectorResponse
from cosipy.util import fetch_wasabi_file

from cosipy.statistics import PoissonLikelihood
from cosipy.background_estimation import FreeNormBinnedBackground
from cosipy.interfaces import ThreeMLPluginInterface
from cosipy.response import BinnedThreeMLModelFolding, BinnedInstrumentResponse, BinnedThreeMLPointSourceResponse
from cosipy.data_io import EmCDSBinnedData

import sys

import astropy.units as u

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from threeML import Band, PointSource, Model, JointLikelihood, DataList
from astromodels import Parameter, Powerlaw

from pathlib import Path


data_path = Path("/heronas/simeone/COSI/CRAB/data")

# Download orientazione
ori_name = "DC4_final_530km_3_month_with_slew_1sbins_GalacticEarth_SAA.fits"
ori_path = data_path / ori_name
fetch_wasabi_file(f"COSI-SMEX/DC4/Data/Orientation/{ori_name}", ori_path)

# Download risposta
dr_name = ("ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284."
           "filtered.nonsparse.binnedimaging.imagingresponse.h5")
dr_path = data_path / dr_name
fetch_wasabi_file(f"COSI-SMEX/DC4/Data/Responses/{dr_name}",
                  output=str(dr_path),
                  checksum="7121f094be50e7bfe9b31e53015b0e85")

# Download fondo non binnato
# bkg_name = "Total_DC4_BG_3months_unbinned_data_filtered_with_SAAcut_withSAAbck.fits.gz"
bkg_name = "Total_DC4_BG_3months_binned_data_filtered_with_SAAcut_withSAAbck.hdf5"
bkg_path = data_path / bkg_name
fetch_wasabi_file(f"COSI-SMEX/DC4/Data/Backgrounds/{bkg_name}", bkg_path)

# Download Crab non binnata
crab_name = "Crab_DC4_3months_unbinned_data_filtered_with_SAAcut.fits.gz"
crab_path = data_path / crab_name
fetch_wasabi_file(f"COSI-SMEX/DC4/Data/Sources/{crab_name}", crab_path)