import astropy.units as u
from astromodels import Powerlaw
from astromodels import Broken_powerlaw


def power_model(index=-2.7, piv=100.0 * u.keV, K=8.6e-5 / u.cm / u.cm / u.s / u.keV):
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


def broken_power_model(
    index_1=-2.65,
    index_2=-2.75,
    break_energy=100.0 * u.keV,
    K=8.6e-5 / u.cm / u.cm / u.s / u.keV,
):
    spectrum = Broken_powerlaw()  # creo il modello spettrale a legge di potenza spezzata non smooth

    spectrum.alpha.min_value = -6.0  # imposto il limite inferiore per il primo indice spettrale
    spectrum.alpha.max_value = 2.0  # imposto il limite superiore per il primo indice spettrale
    spectrum.alpha.delta = 0.05  # imposto il passo iniziale per il primo indice spettrale

    spectrum.beta.min_value = -6.0  # imposto il limite inferiore per il secondo indice spettrale
    spectrum.beta.max_value = 2.0  # imposto il limite superiore per il secondo indice spettrale
    spectrum.beta.delta = 0.05  # imposto il passo iniziale per il secondo indice spettrale

    spectrum.K.value = K.value  # imposto il valore iniziale della normalizzazione
    spectrum.xb.value = break_energy.value  # imposto l'energia di break iniziale
    spectrum.alpha.value = index_1  # imposto il primo indice spettrale iniziale
    spectrum.beta.value = index_2  # imposto il secondo indice spettrale iniziale

    spectrum.K.unit = K.unit  # assegno l'unità di misura alla normalizzazione
    spectrum.xb.unit = break_energy.unit  # assegno l'unità di misura all'energia di break

    spectrum.K.free = True  # lascio libera la normalizzazione durante il fit
    spectrum.alpha.free = True  # lascio libero il primo indice spettrale durante il fit
    spectrum.beta.free = True  # lascio libero il secondo indice spettrale durante il fit
    spectrum.xb.free = False  # tengo fissa l'energia di break

    return spectrum  # restituisco il modello spettrale configurato