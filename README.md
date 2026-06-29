# DriftSearch
Python pipeline to search radio dynamic spectra for broadband drifting radio bursts

Modular search for drifting broadband radio bursts using inverse-variance weighting and matched filtering across a range of dispersion "drifts" (from no drift to strongly drifting).

## Layout

```
DriftSearch/                 # git repo root (what users clone)
├── pyproject.toml           # build/install config (enables `pip install -e .`)
├── README.md
├── DriftSearch/             # the importable package
│   ├── __init__.py          # public API
│   ├── config.py            # DriftSearchConfig dataclass (all the knobs)
│   ├── search.py            # DriftSearch class + Candidate
│   ├── plotting.py          # diagnostic candidate figure
│   ├── burst_funcs.py       # low-level helpers
│   ├── cli.py               # argparse CLI
│   └── __main__.py          # enables `python -m DriftSearch`
└── example/                # sibling of the package, NOT inside it
    ├── single_target.py     # search one target given a variance spectrum
    ├── LTest_00_00.fits     # demo (but real) target dynamic spectrum
    └── var_spectrum.npy     # precomputed real variance spectrum
```

## Install

```
git clone https://github.com/DavidKonijn/DriftSearch.git
cd DriftSearch
python3 -m pip install -e .
```
(it takes no longer than a minute to build) 

To run the demo example: 

```
cd example && python3 single_target.py
```

The pipeline will output two candidate bursts. The first clearly corresponds to the Type II radio burst reported by Callingham et al. (2025). Because this burst is exceptionally bright, a second candidate is also detected due to a noise realization that exceeds the SNR threshold. The example completes in under a minute on a typical system.

## System Requirements
### Hardware requirements
`DriftSearch` requires only a standard computer with enough RAM to support the in-memory operations. Building the variance spectrum loads the off-target dynamic spectra into memory, so RAM should comfortably exceed the combined size of those files (or use a precomputed variance spectrum to avoid loading them).

### Software requirements
#### OS Requirements
This package has no OS-specific code and should run on any *Linux*, *macOS*, or *Windows* machine that can install the Python dependencies below. Note that `numba` installs from prebuilt wheels, which exist for common OS / CPU / Python-version combinations; on unusual setups it may need to build from source. It has been tested on: + Linux: Debian 12 (Bookworm) and + MacOS: Sonoma 14.7.1

#### Python Dependencies
`DriftSearch` mainly depends on the Python scientific stack.

```
numpy
scipy
scikit-image
numba
astropy
matplotlib
tqdm
```

## Quick start with your own spectra

```python
from DriftSearch import DriftSearch, DriftSearchConfig

cfg = DriftSearchConfig(
    input_dir="/path/to/DynSpec",
    output_dir="/path/to/candidates",
    search_snr=9.0,
    search_stokes="I",
)
candidates = DriftSearch(cfg).run()   # builds variance spectrum, searches, plots PDFs
```
or

```
python -m DriftSearch /path/to/DynSpec /path/to/candidates \
    --search-snr 7 --search-stokes V --drift-max 200 --drift-steps 201 --no-plots
```

## Example Burst Candidate:
![Burst Candidate](Dynspec_Source_Drift_8_width_1_downsampled_1_peak-index_522-1.png)

## Configurable parameters (`DriftSearchConfig`)

| Field | Default | Meaning |
|---|---|---|
| `input_dir` | (required) | Where dynspecs live |
| `output_dir` | (required) | Where PDF cands go |
| `off_dir` | `input_dir` | Off-target spectra for the variance spectrum |
| `file_glob` | `"*.fits"` | Which files to load |
| `search_snr` | `9.0` | Detection threshold (watershed peak threshold) |
| `search_stokes` | `"I"` | Stokes parameter to search |
| `weight_stokes` | `"V"` | Stokes used to flag the worst spectra |
| `stokes_labels` | `("I","Q","U","V")` | Order of the Stokes axis in the FITS |
| `drift_min` | `0` | Drift sweep grid |
| `drift_max` | `450` | Drift sweep grid |
| `drift_steps` | `451` | Drift sweep grid |
| `bad_spectra_frac` | `0.10` | Fraction of worst spectra down-weighted |
| `flag_low_percentile` | `5` | Low end tail used to score "bad" spectra |
| `flag_high_percentile` | `95` | High end tail used to score "bad" spectra |
| `watershed_min_distance` | `5` | `peak_local_max` separation |
| `merge_drop` | `2.5` | Saddle-merge depth |
| `Tl_factor` | `1.5` | Lower threshold = `search_snr / Tl_factor` |
| `width_base` | `1/0.95` | Boxcar width ladder |
| `width_power` | `4` | Boxcar width ladder |
| `width_count` | `25` | Boxcar width ladder |
| `reject_top_k_frac` | `0.01` | Fraction of channels dropped in rejection test |
| `freq_min_key` | `FRQ-MIN` | FITS header → MHz |
| `freq_max_key` | `FRQ-MAX` | FITS header → MHz |
| `freq_scale` | `1e6` | FITS header → MHz |
| `make_plots` | `True` | Write diagnostic PDFs |
| `plot_window`| `200` | Plot window half-width / DPI |
| `plot_dpi` | `128` | Plot window half-width / DPI |
| `use_stix_fonts` | `True` | Apply STIX math fonts |

## Citing
`DriftSearch` is free to use and modify. If you use it in academic work, I'd appreciate a citation:

> Konijn, David C., et al. "Occurrence rate of stellar Type II radio bursts from a 100 star-year search for coronal mass ejections." Astronomy & Astrophysics 703 (2025): A198.

## License

MIT

Copyright (c) 2026 David Konijn

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
