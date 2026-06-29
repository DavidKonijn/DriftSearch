import os
import sys

import numpy as np
import matplotlib

from DriftSearch import DriftSearch, DriftSearchConfig
from DriftSearch.plotting import configure_matplotlib

"""
Search a target dynamic spectrum using a given variance spectrum
"""

VARIANCE_PATH = "var_spectrum.npy"
TARGET_FITS   = "LTest_00_00.fits"
OUTPUT_DIR    = " "

SEARCH_SNR    = 9.0
SEARCH_STOKES = "I"


def load_variance(path):
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as d:
            return d["var_spectrum"]
    return np.load(path)


def main(variance_path, target_fits, output_dir):
    cfg = DriftSearchConfig(input_dir=os.path.dirname(target_fits) or ".",output_dir=output_dir,search_snr=SEARCH_SNR,search_stokes=SEARCH_STOKES,use_stix_fonts=True)

    search = DriftSearch(cfg)
    search.var_spectrum = load_variance(variance_path)

    configure_matplotlib(cfg)
    os.makedirs(output_dir, exist_ok=True)

    candidates = search.search_file(target_fits)

    print(f"Found {len(candidates)} candidate(s) in {os.path.basename(target_fits)}:")
    for c in candidates:
        print(f"  S/N={c.snr:5.1f}  drift={c.drift:5.0f}  loc={c.location}  "
              f"width={c.width}  ->  {output_dir}/")
    return candidates


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 3:
        main(args[0], args[1], args[2])
    else:
        main(VARIANCE_PATH, TARGET_FITS, OUTPUT_DIR)
