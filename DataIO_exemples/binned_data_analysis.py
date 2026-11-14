import matplotlib
import matplotlib.pyplot as plt
from cosipy import BinnedData # Importo la classe BinnedData

import logging
logging.basicConfig(level=logging.INFO) # serve solo per avere le informazioni durante i caricamenti

analysis = BinnedData("inputs.yaml") # indica a cosipy di caricare la lista di fotoni direttamente dal file HDF5
analysis.load_binned_data_from_hdf5("binned_data.hdf5")

# Ad esempio, posso proiettare nell'asse del tempo
analysis.binned_data.axes["Time"].centers

analysis.get_raw_spectrum(binned_data="binned_data.hdf5", output_name="crab_spec", show_plots=True) # ottengo lo spettro
analysis.get_raw_lightcurve(binned_data="binned_data.hdf5", output_name="crab_lc", show_plots=True) # ottengo la curva di luce