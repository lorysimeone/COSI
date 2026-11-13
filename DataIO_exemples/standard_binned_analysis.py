# Importo la classe BinnedData

from cosipy import BinnedData

import logging
logging.basicConfig(level=logging.INFO) # serve solo per avere le informazioni durante i caricamenti

analysis = BinnedData("inputs.yaml") # inputs.yaml definisce i parametri di input
analysis.read_tra(output_name="unbinned_data") # legge il file grezzo in formato .tra e lo converte in un dataset non binnato
analysis.get_binned_data() # i dati vengono binnati su 4 assi
analysis.get_raw_spectrum(show_plots=True)
# LC in plot below is normalized to initial time.
analysis.get_raw_lightcurve(show_plots=True)