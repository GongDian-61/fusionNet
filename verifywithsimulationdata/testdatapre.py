import os
import numpy as np
from tqdm import tqdm

def test_preprocess(TRAD_files, OPT_files, GT_files):

    trad_offset = 15

    I_TRAD = TRAD_files.astype(np.float32).transpose(2, 0, 1)
    I_OPT = OPT_files.astype(np.float32).transpose(0, 1, 2)
    I_GT = GT_files.astype(np.float32).transpose(2, 0, 1)

    I_TRAD = I_TRAD[:, 15:951, 15:171]
    I_OPT = I_OPT[:, 15:951, 15:171]
    I_GT = I_GT[:, 15:951, 15:171]

    I_TRAD[I_TRAD > trad_offset] = trad_offset
    I_TRAD[I_TRAD < -trad_offset] = -trad_offset
    I_TRAD = (I_TRAD + trad_offset) / (2 * trad_offset)
    I_OPT = (I_OPT + trad_offset) / (2 * trad_offset)
    I_GT = (I_GT + trad_offset) / (2 * trad_offset)

    return I_TRAD, I_OPT, I_GT