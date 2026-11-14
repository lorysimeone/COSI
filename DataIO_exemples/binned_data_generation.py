import matplotlib
import matplotlib.pyplot as plt
from cosipy import BinnedData # Importo la classe BinnedData

import logging
logging.basicConfig(level=logging.INFO) # serve solo per avere le informazioni durante i caricamenti

analysis = BinnedData("inputs.yaml") # indica a cosipy di caricare la lista di fotoni direttamente dal file HDF5
# Suddivido i bin definiti in inputs.yaml in 4 dimensioni.
analysis.get_binned_data(unbinned_data="unbinned_data.hdf5", output_name="binned_data", make_binning_plots=True, show_plots=True)
# unbinned_data="unbinned_data.hdf5" indica di caricare direttamente la lista unbinned_data.hdf5
# output_name="binned_data" genera un istogramma a 4 dimensioni chiamato binned_data.hdf5
# make_binning_plots=True e show_plots=True generano e fanno visualizzare a schermo la proiezione bidimensionale dei conteggi sulla mappa HEALPix della dimensione spaziale $\psi\chi$ in coordinate galattiche