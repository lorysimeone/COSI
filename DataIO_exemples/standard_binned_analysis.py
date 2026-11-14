import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cosipy import BinnedData # Importo la classe BinnedData

import logging
logging.basicConfig(level=logging.INFO) # serve solo per avere le informazioni durante i caricamenti

analysis = BinnedData("inputs.yaml") # inputs.yaml definisce i parametri di input
analysis.read_tra(output_name="unbinned_data") # legge il file grezzo in formato .tra e lo converte in un dataset non binnato

analysis.get_binned_data() # i dati vengono binnati su 4 assi

import numpy as np
dense = analysis.binned_data.to_dense(copy=False)
print("Total counts in binned data:", np.sum(dense.contents))
print("Non-zero bins:", np.count_nonzero(dense.contents))

fig_spec = analysis.get_raw_spectrum(show_plots=True)
plt.savefig("spettro.png", dpi=300, bbox_inches='tight')
plt.clf()
# LC in plot below is normalized to initial time.
fig_lc = analysis.get_raw_lightcurve(show_plots=True)
plt.savefig("curva_luce.png", dpi=300, bbox_inches='tight')
plt.clf()