import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cosipy import BinnedData # Importo la classe BinnedData

# Disattivo la funzione show() per impedire che la figura venga svuotata prima del salvataggio
plt.show = lambda *args, **kwargs: None

import logging
logging.basicConfig(level=logging.INFO) # serve solo per avere le informazioni durante i caricamenti

analysis = BinnedData("inputs.yaml") # inputs.yaml definisce i parametri di input
analysis.read_tra(output_name="unbinned_data") # legge il file grezzo in formato .tra e lo converte in un dataset non binnato

analysis.get_binned_data() # i dati vengono binnati su 4 assi
analysis.get_raw_spectrum(show_plots=True)
plt.savefig("spettri.png", dpi=300, bbox_inches='tight')
plt.clf()
# LC in plot below is normalized to initial time.
analysis.get_raw_lightcurve(show_plots=True)
plt.savefig("curve_luce.png", dpi=300, bbox_inches='tight')
plt.clf()