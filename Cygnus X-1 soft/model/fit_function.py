import astropy.units as u
from astromodels import Powerlaw
from astromodels import Broken_powerlaw
from astromodels import SmoothlyBrokenPowerLaw


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
    break_energy=1000.0 * u.keV,
    K=17.0 / u.cm / u.cm / u.s / u.keV,  # K e' la normalizzazione a piv = 1 keV: K ~ 8.5e-5 * 100**(-index_1)
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

    spectrum.xb.min_value = 100.0  # imposto il limite inferiore per l'energia di break (keV, inizio del range dei dati)
    spectrum.xb.max_value = 10000.0  # imposto il limite superiore per l'energia di break (keV, fine del range dei dati)
    spectrum.xb.delta = 50.0  # imposto il passo iniziale per l'energia di break

    spectrum.K.unit = K.unit  # assegno l'unità di misura alla normalizzazione
    spectrum.xb.unit = break_energy.unit  # assegno l'unità di misura all'energia di break

    spectrum.K.free = True  # lascio libera la normalizzazione durante il fit
    spectrum.alpha.free = True  # lascio libero il primo indice spettrale durante il fit
    spectrum.beta.free = True  # lascio libero il secondo indice spettrale durante il fit
    spectrum.xb.free = True  # lascio libera l'energia di break durante il fit

    return spectrum  # restituisco il modello spettrale configurato


def smooth_broken_power_model(
    index_1=-2.65,
    index_2=-3.5,
    break_energy=1000.0 * u.keV,
    break_scale=0.3,
    piv=100.0 * u.keV,
    K=8.6e-5 / u.cm / u.cm / u.s / u.keV,
):
    spectrum = SmoothlyBrokenPowerLaw()  # creo il modello spettrale a legge di potenza spezzata smooth

    spectrum.alpha.min_value = -6.0  # imposto il limite inferiore per il primo indice spettrale
    spectrum.alpha.max_value = 2.0  # imposto il limite superiore per il primo indice spettrale
    spectrum.alpha.delta = 0.05  # imposto il passo iniziale per il primo indice spettrale

    spectrum.beta.min_value = -6.0  # imposto il limite inferiore per il secondo indice spettrale
    spectrum.beta.max_value = 2.0  # imposto il limite superiore per il secondo indice spettrale
    spectrum.beta.delta = 0.05  # imposto il passo iniziale per il secondo indice spettrale

    spectrum.break_energy.min_value = 200.0  # imposto il limite inferiore per l'energia di break (keV)
    spectrum.break_energy.max_value = 5000.0  # imposto il limite superiore per l'energia di break (keV)
    spectrum.break_energy.delta = 50.0  # imposto il passo iniziale per l'energia di break

    spectrum.break_scale.min_value = 0.01  # imposto il limite inferiore per il parametro di smussamento
    spectrum.break_scale.max_value = 2.0  # imposto il limite superiore per il parametro di smussamento
    spectrum.break_scale.delta = 0.05  # imposto il passo iniziale per il parametro di smussamento

    spectrum.K.value = K.value  # imposto il valore iniziale della normalizzazione a piv
    spectrum.pivot.value = piv.value  # imposto l'energia di pivot iniziale
    spectrum.alpha.value = index_1  # imposto il primo indice spettrale iniziale
    spectrum.beta.value = index_2  # imposto il secondo indice spettrale iniziale
    spectrum.break_energy.value = break_energy.value  # imposto l'energia di break iniziale
    spectrum.break_scale.value = break_scale  # imposto il parametro di smussamento iniziale

    spectrum.K.unit = K.unit  # assegno l'unità di misura alla normalizzazione
    spectrum.pivot.unit = piv.unit  # assegno l'unità di misura all'energia di pivot
    spectrum.break_energy.unit = break_energy.unit  # assegno l'unità di misura all'energia di break

    spectrum.K.free = True  # lascio libera la normalizzazione durante il fit
    spectrum.alpha.free = True  # lascio libero il primo indice spettrale durante il fit
    spectrum.beta.free = True  # lascio libero il secondo indice spettrale durante il fit
    spectrum.break_energy.free = True  # lascio libera l'energia di break durante il fit
    spectrum.break_scale.free = True  # lascio libero il parametro di smussamento durante il fit
    spectrum.pivot.free = False  # tengo fissa l'energia di pivot

    return spectrum  # restituisco il modello spettrale configurato