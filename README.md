# DriftSearch
Python pipeline to search radio dynamic spectra for broadband drifting radio bursts

Modular search for drifting broadband radio bursts using inverse-variance
weighting and matched filtering across a range of dispersion "drifts" (from no
drift to strongly drifting).

## Layout

```
├── __init__.py      # public API
├── config.py        # DriftSearchConfig dataclass (all the knobs)
├── search.py        # DriftSearch class + Candidate
├── plotting.py      # diagnostic candidate figure
├── burst_funcs.py   # low-level helpers
├── cli.py           # argparse CLI
└── __main__.py      # enables `python -m DriftSearch
```

## Install

```
cd /path/containing/DriftSearch
pip install numpy scipy scikit-image scikit-learn numba astropy matplotlib tqdm
# then just import it, or add the parent dir to PYTHONPATH
```

## Quick start

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

## Configurable parameters (`DriftSearchConfig`)

| Field | Default | Meaning |
|---|---|---|
| `input_dir` / `output_dir` | — (required) | where dynspecs live / where PDFs go |
| `off_dir` | `input_dir` | off-target spectra for the variance spectrum |
| `file_glob` | `"*.fits"` | which files to load |
| `search_snr` | `9.0` | detection threshold (watershed peak threshold) |
| `search_stokes` | `"I"` | Stokes parameter to search |
| `weight_stokes` | `"V"` | Stokes used to flag the worst spectra |
| `stokes_labels` | `("I","Q","U","V")` | order of the Stokes axis in the FITS |
| `drift_min/max/steps` | `0 / 450 / 451` | drift sweep grid |
| `bad_spectra_frac` | `0.10` | fraction of worst spectra down-weighted |
| `flag_low/high_percentile` | `5 / 95` | tails used to score "bad" spectra |
| `watershed_min_distance` | `5` | `peak_local_max` separation |
| `merge_drop` | `2.5` | saddle-merge depth |
| `Tl_factor` | `1.5` | lower threshold = `search_snr / Tl_factor` |
| `width_base/power/count` | `1/0.95 / 4 / 25` | boxcar width ladder |
| `reject_top_k_frac` | `0.01` | fraction of channels dropped in rejection test |
| `freq_min/max_key`, `freq_scale` | `FRQ-MIN`, `FRQ-MAX`, `1e6` | FITS header → MHz |
| `make_plots` | `True` | write diagnostic PDFs |
| `plot_window` / `plot_dpi` | `200` / `128` | plot window half-width / DPI |
| `use_stix_fonts` | `True` | apply STIX math fonts |
