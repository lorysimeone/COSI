import h5py
import numpy as np

# Apro il file in modalità lettura e scrittura ('r+')
file_path = "/heronas/simeone/COSI/Full_detector_response/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.h5"

with h5py.File(file_path, "r+") as f:
    # Cerco il gruppo relativo agli assi esplorando la radice e il gruppo DRM
    axes_group = None
    if "AXES" in f:
        axes_group = f["AXES"]
    elif "DRM" in f and "AXES" in f["DRM"]:
        axes_group = f["DRM"]["AXES"]

    if axes_group is not None:
        # Scorro ogni asse presente all'interno del gruppo
        for axis_name in axes_group.keys():
            axis = axes_group[axis_name]

            # Controllo e aggiungo l'attributo TYPE se noto che manca
            if "TYPE" not in axis.attrs:
                class_attr = axis.attrs.get("__class__", ["", ""])
                class_name = class_attr[1] if len(class_attr) > 1 else class_attr[0]

                if isinstance(class_name, bytes):
                    class_name = class_name.decode('utf-8')

                class_name = class_name.lower()

                if "healpix" in class_name:
                    axis_type = "healpix"
                elif "continuous" in class_name:
                    axis_type = "continuous"
                else:
                    axis_type = "str"

                axis.attrs["TYPE"] = axis_type
                print(f"Ho aggiunto l'attributo TYPE='{axis_type}' all'asse {axis_name}.")

            # Verifico il tipo attuale dell'asse per controllare NSIDE e SCHEME
            current_type = axis.attrs.get("TYPE", "")
            if isinstance(current_type, bytes):
                current_type = current_type.decode('utf-8')

            # Applico la modifica per NSIDE e SCHEME solo se l'asse è healpix
            if current_type == "healpix":
                if "NSIDE" not in axis.attrs:
                    # Deduco i pixel spaziali totali dalla lunghezza del dataset escludendo il bin di overflow (1)
                    n_pixels = axis.shape[0] - 1

                    # Calcolo NSIDE ricavandolo matematicamente dalla formula N_pixels = 12 * NSIDE^2
                    nside = int(np.sqrt(n_pixels / 12))

                    # Assegno fisicamente l'attributo nel file
                    axis.attrs["NSIDE"] = nside
                    print(f"Ho aggiunto l'attributo NSIDE={nside} all'asse {axis_name}.")

                if "SCHEME" not in axis.attrs:
                    # Assegno lo schema standard 'ring'
                    axis.attrs["SCHEME"] = "ring"
                    print(f"Ho aggiunto l'attributo SCHEME='ring' all'asse {axis_name}.")

        print("Aggiornamento del file HDF5 completato con successo.")
    else:
        print("Non trovo il gruppo AXES nel file.")