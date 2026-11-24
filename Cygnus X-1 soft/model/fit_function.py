import astropy.units as u
from astromodels import Powerlaw


def crea_modello_spettrale(index=-2.0, piv=100.0 * u.keV, K=8.6e-5 / u.cm / u.cm / u.s / u.keV):
    spectrum = Powerlaw()  # creo il modello spettrale a legge di potenza singola

    spectrum.index.min_value = -6.0  # imposto il limite inferiore per l'indice spettrale
    spectrum.index.max_value = 2.0  # imposto il limite superiore per l'indice spettrale
    spectrum.index.delta = 0.05  # imposto il passo iniziale per l'indice spettrale

    spectrum.K.value = K.value  # imposto il valore iniziale della normalizzazione
    spectrum.piv.value = piv.value  # imposto l'energia di pivot iniziale
    spectrum.index.value = index  # imposto l'indice spettrale iniziale

    spectrum.K.unit = K.unit  # assegno l'unità di misura alla normalizzazione
    spectrum.piv.unit = piv.unit  # assegno l'unità di misura all'energia di pivot

    spectrum.K.free = True  # lascio libera la normalizzazione durante il fit
    spectrum.index.free = True  # lascio libero l'indice spettrale durante il fit
    spectrum.piv.free = False  # tengo fissa l'energia di pivot

    return spectrum  # restituisco il modello spettrale configurato