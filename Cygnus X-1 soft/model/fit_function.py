import astropy.units as u
from astromodels import Powerlaw
from astromodels import Broken_powerlaw
from astromodels import SmoothlyBrokenPowerLaw
from astromodels import Cutoff_powerlaw


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

def cutoff_power_model(
    index_1=-2.65,
    cutoff_energy=200.0 * u.keV,
    K_1=1.5e-4 / u.cm / u.cm / u.s / u.keV,
    index_2=-2.0,
    K_2=1.0e-5 / u.cm / u.cm / u.s / u.keV,
    piv=100.0 * u.keV,
):
    cutoff = Cutoff_powerlaw()  # creo la componente di Comptonizzazione termica con taglio esponenziale
    tail = Powerlaw()  # creo la componente a legge di potenza per la coda ad alta energia

    cutoff.index.min_value = -6.0  # imposto il limite inferiore per l'indice della cutoff powerlaw
    cutoff.index.max_value = 2.0  # imposto il limite superiore per l'indice della cutoff powerlaw
    cutoff.index.delta = 0.05  # imposto il passo iniziale per l'indice della cutoff powerlaw

    cutoff.xc.min_value = 50.0  # imposto il limite inferiore per l'energia di cutoff (keV)
    cutoff.xc.max_value = 10000.0  # imposto il limite superiore per l'energia di cutoff (keV)
    cutoff.xc.delta = 20.0  # imposto il passo iniziale per l'energia di cutoff

    cutoff.K.min_value = 0.0  # imposto il limite inferiore per la normalizzazione della cutoff powerlaw
    cutoff.K.value = K_1.value  # imposto il valore iniziale della normalizzazione della cutoff powerlaw
    cutoff.piv.value = piv.value  # imposto l'energia di pivot a 100 keV per la cutoff powerlaw
    cutoff.index.value = index_1  # imposto l'indice iniziale della cutoff powerlaw
    cutoff.xc.value = cutoff_energy.value  # imposto l'energia di cutoff iniziale

    cutoff.K.unit = K_1.unit  # assegno l'unità di misura alla normalizzazione della cutoff powerlaw
    cutoff.piv.unit = piv.unit  # assegno l'unità di misura all'energia di pivot della cutoff powerlaw
    cutoff.xc.unit = cutoff_energy.unit  # assegno l'unità di misura all'energia di cutoff

    cutoff.K.free = True  # lascio libera la normalizzazione della cutoff powerlaw durante il fit
    cutoff.index.free = True  # lascio libero l'indice della cutoff powerlaw durante il fit
    cutoff.xc.free = True  # lascio libera l'energia di cutoff durante il fit
    cutoff.piv.free = False  # tengo fissa l'energia di pivot della cutoff powerlaw a 100 keV

    tail.index.min_value = -6.0  # imposto il limite inferiore per l'indice della powerlaw
    tail.index.max_value = 2.0  # imposto il limite superiore per l'indice della powerlaw
    tail.index.delta = 0.05  # imposto il passo iniziale per l'indice della powerlaw

    tail.K.min_value = 0.0  # imposto il limite inferiore per la normalizzazione della powerlaw
    tail.K.value = K_2.value  # imposto il valore iniziale della normalizzazione della powerlaw
    tail.piv.value = piv.value  # imposto l'energia di pivot a 100 keV per la powerlaw
    tail.index.value = index_2  # imposto l'indice iniziale della powerlaw

    tail.K.unit = K_2.unit  # assegno l'unità di misura alla normalizzazione della powerlaw
    tail.piv.unit = piv.unit  # assegno l'unità di misura all'energia di pivot della powerlaw

    tail.K.free = True  # lascio libera la normalizzazione della powerlaw durante il fit
    tail.index.free = True  # lascio libero l'indice della powerlaw durante il fit
    tail.piv.free = False  # tengo fissa l'energia di pivot della powerlaw a 100 keV

    return cutoff + tail  # restituisco il modello spettrale composto configurato