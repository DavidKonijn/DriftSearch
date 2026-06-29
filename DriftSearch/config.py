import numpy as np

from dataclasses import dataclass
from typing import Optional, Sequence

"""
Config file for the Drifting Burst Search

Every tunable parameter of the search that lives in the DriftSearchConfig class
"""

@dataclass
class DriftSearchConfig:
    # I/O    
    input_dir: str
    output_dir: str
    off_dir: Optional[str] = None
    file_glob: str = "*.fits"
    filename_prefix: str = "Dynspec"

    # detection params
    search_snr: float = 9.0
    search_stokes: str = "I"
    weight_stokes: str = "V"
    stokes_labels: Sequence[str] = ("I", "Q", "U", "V")

    # drift sweep params
    drift_min: float = 0.0
    drift_max: float = 450.0
    drift_steps: int = 451

    # variance / flagging
    bad_spectra_frac: float = 0.10 # percentage of offbeams with most points below flag_low and higher than flag_high that will not be used to create the variance spec
    flag_low_percentile: float = 5.0
    flag_high_percentile: float = 95.0

    # watershed seg. params (trial and error to set these)
    watershed_min_distance: int = 5
    merge_drop: float = 2.5
    Tl_factor: float = 1.5

    # boxcar widths
    width_base: float = 1.0 / 0.95
    width_power: int = 4
    width_count: int = 25

    # channel rejection
    reject_top_k_frac: float = 0.01 #if we drop these channels, we check if SNR>SNR_threshold

    # FITS headers
    freq_min_key: str = "FRQ-MIN"
    freq_max_key: str = "FRQ-MAX"
    freq_scale: float = 1e6

    # plotting
    make_plots: bool = True
    plot_window: int = 200
    plot_dpi: int = 128
    plot_figsize: Sequence[float] = (30, 20)
    use_stix_fonts: bool = True

    # derived helpers
    def stokes_index(self, name: str) -> int:
        return list(self.stokes_labels).index(name)
    @property
    def search_index(self) -> int:
        return self.stokes_index(self.search_stokes)
    @property
    def weight_index(self) -> int:
        return self.stokes_index(self.weight_stokes)
    @property
    def Th(self) -> float:
        return self.search_snr
    @property
    def Tl(self) -> float:
        return self.search_snr / self.Tl_factor
    def sweep_list(self) -> np.ndarray:
        return np.linspace(self.drift_min, self.drift_max, self.drift_steps)
    def width_list(self) -> np.ndarray:
        powers = np.arange(1, self.width_count + 1)
        return np.unique(np.int64((self.width_base ** self.width_power) ** powers))
