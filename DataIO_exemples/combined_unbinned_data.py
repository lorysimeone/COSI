import matplotlib
import matplotlib.pyplot as plt
from cosipy import BinnedData # Importo la classe BinnedData
import pandas as pd
import numpy as np

import logging
logging.basicConfig(level=logging.INFO) # serve solo per avere le informazioni durante i caricamenti

analysis = BinnedData("inputs.yaml")
# Combino i dati
analysis.combine_unbinned_data(["unbinned_data.hdf5","unbinned_data.hdf5","unbinned_data.hdf5"], output_name="combined_unbinned_data")
# Binno i dati combinati
analysis.get_binned_data(unbinned_data="combined_unbinned_data.hdf5", output_name="combined_binned_data")
analysis.get_raw_spectrum(binned_data="combined_binned_data.hdf5", output_name="crab_spec_3x", show_plots=True)
analysis.get_raw_lightcurve(binned_data="combined_binned_data.hdf5", output_name="crab_lc_3x", show_plots=True)

# Comparo i risultati con i dati originali moltiplicati per 3

# Curva di luce:
df = pd.read_csv("crab_lc.dat", delimiter='\t')
plt.semilogy(df["Time[UTC]"] - df["Time[UTC]"][0], df["Rate[ct/s]"],label="Crab")
plt.semilogy(df["Time[UTC]"] - df["Time[UTC]"][0], 3*df["Rate[ct/s]"],label="3xCrab")

df = pd.read_csv("crab_lc_3x.dat", delimiter='\t')
plt.semilogy(df["Time[UTC]"] - df["Time[UTC]"][0], df["Rate[ct/s]"],ls="--",label="Combined")

plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("ct/s")
plt.savefig("combined_lc_comparison.pdf")

# Spettro
df = pd.read_csv("crab_spec.dat", delimiter='\t')
plt.loglog(df["Energy[keV]"],df["Rate[ct/keV]"],label="Crab")
plt.loglog(df["Energy[keV]"],3*df["Rate[ct/keV]"],label="3xCrab")

df = pd.read_csv("crab_spec_3x.dat", delimiter='\t')
plt.loglog(df["Energy[keV]"],df["Rate[ct/keV]"],ls="--",label="Combined")

plt.legend()
plt.xlabel("Energy [keV]")
plt.ylabel("ct/keV")
plt.savefig("combined_spectrum_comparison.pdf")

# In alternativa, sommo gli istogrammi
analysis = BinnedData("inputs.yaml")
analysis.get_binned_data(unbinned_data="unbinned_data.hdf5", output_name="binned_data")
combined_hist = analysis.binned_data + analysis.binned_data + analysis.binned_data

print()
print("The total number of photons has increased by a factor of 3, as expected:")
print("single histogram: " + str(np.sum(analysis.binned_data.to_dense().contents)))
print("combined histogram: " + str(np.sum(combined_hist.to_dense().contents)))