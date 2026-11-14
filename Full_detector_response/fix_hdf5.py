import h5py

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
        # Scorro tutti gli assi presenti nel gruppo
        for axis_name in axes_group.keys():
            axis = axes_group[axis_name]
            
            # Verifico se manca l'attributo TYPE per evitare di sovrascriverlo se già presente
            if "TYPE" not in axis.attrs:
                # Recupero l'attributo __class__ per dedurre il tipo corretto
                class_attr = axis.attrs.get("__class__", ["", ""])
                class_name = class_attr[1] if len(class_attr) > 1 else class_attr[0]
                
                # Lo converto in stringa qualora fosse codificato in formato byte
                if isinstance(class_name, bytes):
                    class_name = class_name.decode('utf-8')
                    
                class_name = class_name.lower()
                
                # Assegno il valore TYPE corretto in base alla classe dell'asse
                if "healpix" in class_name:
                    axis_type = "healpix"
                elif "continuous" in class_name:
                    axis_type = "continuous"
                else:
                    axis_type = "str"
                    
                # Scrivo fisicamente il nuovo attributo nel file HDF5
                axis.attrs["TYPE"] = axis_type
                print(f"Ho aggiunto l'attributo TYPE='{axis_type}' all'asse {axis_name}.")
            else:
                print(f"L'asse {axis_name} ha già l'attributo TYPE.")
        print("Aggiornamento del file HDF5 completato con successo.")
    else:
        print("Non sono riuscito a trovare il gruppo AXES nel file.")
