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
    break_energy=350.0 * u.keV,
    K=1.7e-6 / u.cm / u.cm / u.s / u.keV,
):
    spectrum = BrokenPowerlaw()  # creo il modello spettrale a legge di potenza spezzata non smooth

    spectrum.index_1.min_value = -6.0  # imposto il limite inferiore per il primo indice spettrale
    spectrum.index_1.max_value = 2.0  # imposto il limite superiore per il primo indice spettrale
    spectrum.index_1.delta = 0.05  # imposto il passo iniziale per il primo indice spettrale

    spectrum.index_2.min_value = -6.0  # imposto il limite inferiore per il secondo indice spettrale
    spectrum.index_2.max_value = 2.0  # imposto il limite superiore per il secondo indice spettrale
    spectrum.index_2.delta = 0.05  # imposto il passo iniziale per il secondo indice spettrale

    spectrum.K.value = K.value  # imposto il valore iniziale della normalizzazione
    spectrum.break_energy.value = break_energy.value  # imposto l'energia di break iniziale
    spectrum.index_1.value = index_1  # imposto il primo indice spettrale iniziale
    spectrum.index_2.value = index_2  # imposto il secondo indice spettrale iniziale

    spectrum.K.unit = K.unit  # assegno l'unità di misura alla normalizzazione
    spectrum.break_energy.unit = break_energy.unit  # assegno l'unità di misura all'energia di break

    spectrum.K.free = True  # lascio libera la normalizzazione durante il fit
    spectrum.index_1.free = True  # lascio libero il primo indice spettrale durante il fit
    spectrum.index_2.free = True  # lascio libero il secondo indice spettrale durante il fit
    spectrum.break_energy.free = False  # tengo fissa l'energia di break

    return spectrum  # restituisco il modello spettrale configurato