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


data_path = Path("/heronas/simeone/COSI/Cygnus X-1 soft")

# Download
name = "cygX1_soft_3months_unbinned_data_filtered_with_SAAcut.fits.gz"
path = data_path / name
fetch_wasabi_file(f"COSI-SMEX/DC4/Data/Sources/{name}", path)

# Binning
analysis = BinnedData("/heronas/simeone/COSI/ori_bkg_yaml/crab.yaml")
analysis.get_binned_data(unbinned_data=path, output_name=path / "cygnus_soft_binned_data", make_binning_plots=False, show_plots=False)
