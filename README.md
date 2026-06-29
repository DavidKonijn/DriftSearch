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
git clone https://github.com/DavidKonijn/DriftSearch.git
cd DriftSearch
pip install -e .
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
