import matplotlib
import matplotlib.pyplot as plt
from cosipy import BinnedData # Importo la classe BinnedData
import pandas as pd
import numpy as np

import logging
logging.basicConfig(level=logging.INFO) # Lo uso solo per avere le informazioni durante i caricamenti

analysis = BinnedData("inputs_half_time.yaml")

# Salto la selezione temporale e il binning poiché i file .hdf5 sono già presenti
analysis.get_raw_spectrum(binned_data="selected_combined_binned_data.hdf5", output_name="selected_crab_spec_3x", show_plots=True)
plt.figure()
analysis.get_raw_lightcurve(binned_data="selected_combined_binned_data.hdf5", output_name="selected_crab_lc_3x", show_plots=True)
plt.figure()

# Comparo il dataset
df = pd.read_csv("crab_lc_3x.dat", delimiter='\t')
plt.semilogy(df["Time[UTC]"] - df["Time[UTC]"][0], df["Rate[ct/s]"],ls="-",label="Combined")

# Curva di luce:
df = pd.read_csv("selected_crab_lc_3x.dat", delimiter='\t')
plt.semilogy(df["Time[UTC]"] - df["Time[UTC]"][0], df["Rate[ct/s]"],ls="--",label="Selected Combined")

plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("ct/s")
plt.savefig("combined_lc_comparison.pdf")
plt.figure()

# Spettro
df = pd.read_csv("crab_spec_3x.dat", delimiter='\t')
plt.loglog(df["Energy[keV]"],df["Rate[ct/keV]"],ls="-",label="Combined")

df = pd.read_csv("selected_crab_spec_3x.dat", delimiter='\t')
plt.loglog(df["Energy[keV]"],df["Rate[ct/keV]"],ls="--",label="Selected Combined")

plt.legend()
plt.xlabel("Energy [keV]")
plt.ylabel("ct/keV")
plt.savefig("combined_spectrum_comparison.pdf")
plt.figure()