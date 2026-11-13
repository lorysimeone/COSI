# Importo la classe BinnedData

from cosipy import BinnedData

import logging
logging.basicConfig(level=logging.INFO) # serve solo per avere le informazioni durante i caricamenti

analysis = BinnedData("inputs.yaml") # inputs.yaml definisce i parametri di input
