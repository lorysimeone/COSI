from cosipy.response import FullDetectorResponse

import shutil
import os

response_path = '/heronas/simeone/COSI/Full_detector_response/dc4_data/ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284.filtered.nonsparse.binnedimaging.imagingresponse.h5'
with FullDetectorResponse.open(response_path) as response:

    print(f"NSIDE = {response.nside}")
    print(f"SCHEME = {response.scheme}")
    print(f"NPIX = {response.npix}")
    print(f"Pixel size = {np.sqrt(response.pixarea()).to(u.deg):.2f}")

