import matplotlib
import matplotlib.pyplot as plt
from cosipy import BinnedData # Importo la classe BinnedData

import logging

from CRAB.download import bkg_name

logging.basicConfig(level=logging.INFO) # serve solo per avere le informazioni durante i caricamenti

from pathlib import Path

data_path = Path("/heronas/simeone/COSI/CRAB/data")

crab_name = "Crab_DC4_3months_unbinned_data_filtered_with_SAAcut.fits.gz"
crab_path = data_path / crab_name

bkg_name_ = "Total_DC4_BG_3months_unbinned_data_filtered_with_SAAcut_withSAAbck.fits.gz"
bkg_path = data_path / bkg_name_

analysis = BinnedData(data_path / "crab.yaml") # indica a cosipy di caricare la lista di fotoni direttamente dal file HDF5
# Suddivido i bin definiti in crab.yaml in 4 dimensioni.
analysis.get_binned_data(unbinned_data=crab_path, output_name="crab_binned_data", make_binning_plots=True, show_plots=True)
analysis.get_binned_data(unbinned_data=bkg_path , output_name="bkg_binned_data" , make_binning_plots=True, show_plots=True)