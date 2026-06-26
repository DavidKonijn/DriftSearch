import gc
import glob
import os
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits
from scipy.signal import convolve
from tqdm import tqdm

from .burst_funcs import dedup_smallest_width_wins_by_label_overlap,matched_filter_snr,remove_brightest_channels_reject,robust_zscore_1d,watershed_merge_labels
from .config import DriftSearchConfig

"""
Core of the Drifting Burst Search

The DriftSearch class can be called for three main features: 
    - Building inverse-variance-weighted spectrum
    - Finding actual burst candidates
    - Plotting the candidates as PDF (optional)

"""

@dataclass
class Candidate:
    """
    A single drifting-burst candidate
    """

    location: int                   # time bin of the burst peak
    width: int                      # boxcar width (time bins)
    drift_index: int                # index into the sweep list
    drift: float                    # sweep value (the physical drift in number of timesteps)
    snr: float                      # peak matched-filter S/N
    snr_dropped: float = np.nan     # S/N after dropping the brightest channels
    f_min: float = 0.0              # lowest freq
    f_max: float = 0.0              # highest freq
    chans: int = 0                  # number of chans
    gaia_id: str = ""               # targeted star gaia id
    tc_number: str = ""             # observation number
    source_file: str = ""           # source file name 


class DriftSearch:
    def __init__(self, config: DriftSearchConfig):
        self.cfg = config
        self.var_spectrum = None  # set by build_variance_spectrum()

    def parse_metadata(self, path):
        """
        NOTE: override if your files are named differently!!
        
        Return ``(target_coords, tc_number)`` parsed from the file name
        """
        base = os.path.basename(path)
        target_coords = (''.join(base.split("_")[-2:])).split(".fits")[0]
        tc_number = base.split("_")[-3]
        return target_coords, tc_number

    def build_variance_spectrum(self, off_files=None):
        """Calculate the weighted variance spectrum from off-target data

        The worst spectra (those most often in the tails of ultra-bright-pixels) are given zero weight
        the variance is then the inverse-variance-weighted sample variance over files
        """
        cfg = self.cfg
        if off_files is None:
            off_dir = cfg.off_dir or cfg.input_dir
            off_files = sorted(glob.glob(os.path.join(off_dir, cfg.file_glob)))
        if not off_files:
            raise FileNotFoundError(
                f"No off-target files matching {cfg.file_glob!r} in "
                f"{cfg.off_dir or cfg.input_dir!r}")

        with fits.open(off_files[0], memmap=True) as hdul:
            shape = hdul[0].data.shape
            dtype = hdul[0].data.dtype

        n = len(off_files)
        off_spectra = np.empty((n, *shape), dtype=dtype)
        for i, path in enumerate(tqdm(off_files, desc="loading off-target")):
            with fits.open(path, memmap=True) as hdul:
                off_spectra[i] = hdul[0].data

        # flag the spectra that most often fall in the distribution tails of the
        # chosen weighting Stokes parameter.
        wi = cfg.weight_index
        lo = np.percentile(off_spectra[:, wi], cfg.flag_low_percentile, axis=0)
        hi = np.percentile(off_spectra[:, wi], cfg.flag_high_percentile, axis=0)
        tail_count = (
            np.sum(off_spectra[:, wi] < lo, axis=(1, 2))
            + np.sum(off_spectra[:, wi] > hi, axis=(1, 2)))
        
        n_drop = int(n * cfg.bad_spectra_frac)
        worst = np.argsort(tail_count)[-n_drop:] if n_drop > 0 else np.empty(0, int)

        w = np.ones(n, dtype=np.float32)
        w[worst] = 0.0

        valid = np.isfinite(off_spectra)
        x0 = np.where(valid, off_spectra, 0.0)

        #einsum is faster than normal sum like this
        weighted_sum = np.einsum("i...,i->...", x0, w, optimize=True)
        sum_w = np.einsum("i...,i->...", valid.astype(np.float32), w, optimize=True)
        sum_w[sum_w == 0] = np.nan
        weighted_mean = weighted_sum / sum_w

        diff0 = np.where(valid, off_spectra - weighted_mean, 0.0)
        weighted_sq_diff = np.einsum("i...,i->...", diff0 * diff0, w, optimize=True)

        var_spectrum = weighted_sq_diff / sum_w
        var_spectrum[var_spectrum == 0] = np.nan

        self.var_spectrum = var_spectrum
        del off_spectra, valid, x0, diff0
        gc.collect()
        return var_spectrum


    def find_candidates(self, ds, f_min, f_max, chans):
        """
        Find burst cands in single dynspecs.
        """

        cfg = self.cfg
        if self.var_spectrum is None:
            raise RuntimeError("call build_variance_spectrum() before searching")

        s = cfg.search_index
        var = self.var_spectrum
        ntime = ds[s].shape[1]
        sweep_list = cfg.sweep_list()

        #create the drift swept spectra by looping over the sweeplist
        swept = np.zeros((len(sweep_list), ntime), dtype=np.float32)
        dynspec = ds[s].astype(np.float64)
        for ii, sweep in enumerate(sweep_list):
            ts = matched_filter_snr(dynspec, var[s], sweep, f_min, f_max, chans)
            swept[ii, :] = robust_zscore_1d(ts)

        # search the swept spectrum, using boxcar match to downsample, and watershed segmentation to differentiate different bursts
        per_width = []
        for width in tqdm(cfg.width_list(), desc="boxcar widths", leave=False):
            kernel = np.ones(int(width), dtype=np.float32)
            norm = float(width) ** 0.5
            down = np.zeros_like(swept)
            for ii in range(len(sweep_list)):
                down[ii] = convolve(swept[ii], kernel, mode="same") / norm

            labels, _ = watershed_merge_labels(
                down, Th=cfg.Th, Tl=cfg.Tl,
                min_distance=cfg.watershed_min_distance, merge_drop=cfg.merge_drop)
            
            if labels.max() == 0:
                continue

            per_width.append({"width": int(width), "labels_merged": labels, "S_s": down})

        if not per_width:
            return []

        # deduplicate across widths (smallest width wins on 2D overlap)
        unique_indices, _ = dedup_smallest_width_wins_by_label_overlap(per_width)
        top_k = np.int32(np.round(chans * cfg.reject_top_k_frac))

        candidates = []
        for loc, width, drift_idx, snr in unique_indices:
            drift = float(sweep_list[drift_idx])
            snr_dropped, reject = remove_brightest_channels_reject(
                dynspec, var[s], drift, loc, width,
                f_min, f_max, chans, cfg.search_snr, top_k=top_k)
            
            if reject:
                continue

            candidates.append(Candidate(location=int(loc), width=int(width), drift_index=int(drift_idx),
                                        drift=drift, snr=float(snr), snr_dropped=float(snr_dropped),
                                        f_min=f_min, f_max=f_max, chans=int(chans)))
        return candidates

    def search_file(self, fits_path):
        cfg = self.cfg
        gaia_id, tc_number = self.parse_metadata(fits_path)

        with fits.open(fits_path) as hdul:
            ds = hdul[0].data
            f_min = hdul[0].header[cfg.freq_min_key] / cfg.freq_scale
            f_max = hdul[0].header[cfg.freq_max_key] / cfg.freq_scale
            chans = ds.shape[1]

            candidates = self.find_candidates(ds, f_min, f_max, chans)
            for c in candidates:
                c.gaia_id, c.tc_number, c.source_file = gaia_id, tc_number, fits_path

            if cfg.make_plots and candidates:
                from .plotting import plot_candidate  # local import: matplotlib is heavy
                for c in candidates:
                    plot_candidate(ds, self.var_spectrum, c, cfg, self.plot_path(c))

        return candidates

    def plot_path(self, c):
        cfg = self.cfg
        tdown = max(1, 2 ** (int(np.log10(c.width) / np.log10(2)) - 2))
        fname = (
            f"{cfg.filename_prefix}_Source_{c.gaia_id}"
            f"_SNR_{np.round(abs(c.snr), 2)}"
            f"_Drift_{c.drift_index}"
            f"_width_{c.width}"
            f"_downsampled_{tdown}"
            f"_obs_{c.tc_number}"
            f"_peak-index_{c.location}.pdf"
        )
        return os.path.join(cfg.output_dir, fname)


    def run(self, input_dir=None):
        """Main function; builds the variance spectrum (if toggeled) and searches every target file

        returns: a flat list of all 'Candidate' objects found
        """
        cfg = self.cfg
        if cfg.make_plots:
            from .plotting import configure_matplotlib
            configure_matplotlib(cfg)
            os.makedirs(cfg.output_dir, exist_ok=True)

        if self.var_spectrum is None:
            self.build_variance_spectrum()

        in_dir = input_dir or cfg.input_dir
        targets = sorted(glob.glob(os.path.join(in_dir, cfg.file_glob)))

        all_candidates = []
        for path in tqdm(targets, desc="searching targets"):
            all_candidates.extend(self.search_file(path))

        gc.collect()
        return all_candidates

def run_drift_search(config: DriftSearchConfig):
    return DriftSearch(config).run()
