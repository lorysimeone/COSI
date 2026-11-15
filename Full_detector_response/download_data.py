from pathlib import Path
from cosipy.util import fetch_wasabi_file

data_dir = Path("dc4_data")
data_dir.mkdir(exist_ok=True)

ori_name = "DC4_final_530km_3_month_with_slew_1sbins_GalacticEarth_SAA.fits"
resp_name = ("ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284."
             "filtered.nonsparse.binnedimaging.imagingresponse.h5")

ori_path = data_dir / ori_name
response_path = data_dir / resp_name

fetch_wasabi_file(f"COSI-SMEX/DC4/Data/Orientation/{ori_name}", ori_path)
fetch_wasabi_file(f"COSI-SMEX/DC4/Data/Responses/{resp_name}", response_path)