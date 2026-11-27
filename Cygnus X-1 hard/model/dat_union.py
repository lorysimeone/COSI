from pathlib import Path
import urllib.request
import numpy as np

# definisco i percorsi locali e gli URL GitHub dei due file .dat
base_dir = Path(".")
file_low = base_dir / "cygX1_hard_0.1-0.4_spec.dat"
file_high = base_dir / "cygX1_hard_0.4-10_spec.dat"
output_file = base_dir / "cygX1_hard_spec.dat"

base_url = (
    "https://raw.githubusercontent.com/cositools/cosi-sim/main/"
    "cosi_sim/Source_Library/DC3/sources/Galactic/cygX1_hard"
)
url_low = f"{base_url}/cygX1_hard_0.1-0.4_spec.dat"
url_high = f"{base_url}/cygX1_hard_0.4-10_spec.dat"


def ensure_file(local_path: Path, url: str) -> None:
    # scarico il file da GitHub se non lo trovo nella cartella locale
    if not local_path.exists():
        print(f"Scarico {local_path.name} da GitHub...")
        urllib.request.urlretrieve(url, local_path)


def parse_dat_file(path: Path):
    # leggo il flusso integrato dall'intestazione e i punti DP dal file .dat
    energies = []
    fluxes = []
    integrated_flux = None

    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#flux"):
                # estraggo il valore numerico del flusso integrato dalla riga di commento
                after_colon = stripped.split(":", 1)[1].strip()
                integrated_flux = float(after_colon.split()[0])
            elif stripped.startswith("DP"):
                # estraggo energia e flusso differenziale da ciascuna riga DP
                parts = stripped.split()
                energies.append(float(parts[1]))
                fluxes.append(float(parts[2]))

    return np.array(energies, dtype=float), np.array(fluxes, dtype=float), integrated_flux


# carico i due file spettrali delle fasce 0.1-0.4 MeV e 0.4-10 MeV
ensure_file(file_low, url_low)
ensure_file(file_high, url_high)

e_low, f_low, int_flux_low = parse_dat_file(file_low)
e_high, f_high, int_flux_high = parse_dat_file(file_high)

# verifico che le due griglie energetiche abbiano lo stesso numero di punti e siano allineate
if e_low.shape != e_high.shape or not np.allclose(e_low, e_high, rtol=1e-4, atol=0.0):
    raise ValueError("Le griglie energetiche dei due file .dat non coincidono.")

# mantengo i valori di energia a precisione piena e sommo i flussi differenziali
energies_combined = np.where(f_low == 0.0, e_low, e_high)
fluxes_combined = f_low + f_high

# calcolo il flusso integrato totale sommando i flussi dei due file
if int_flux_low is not None and int_flux_high is not None:
    total_integrated_flux = int_flux_low + int_flux_high
else:
    total_integrated_flux = float(np.trapezoid(fluxes_combined, energies_combined))

# scrivo il file .dat unificato mantenendo il formato originale
with open(output_file, "w") as out:
    out.write("#Format: <DP> <Energy in keV> <Spectrum in ph cm^-2 s^-1 keV^-1>\n")
    out.write(f"#flux (100.0 - 10000.0 keV): {total_integrated_flux:.17g} ph/cm^2/s\n")
    out.write("\n\nIP LOGLOG\n\n")
    for e_val, f_val in zip(energies_combined, fluxes_combined):
        out.write(f"DP  {e_val}  {f_val}\n")
    out.write("EN\n")

print(f"File salvato in: {output_file.resolve()}")
print(f"Flusso integrato totale (100 - 10000 keV): {total_integrated_flux:.6e} ph/cm^2/s")