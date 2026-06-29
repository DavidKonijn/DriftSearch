import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import convolve
from .burst_funcs import downsamp, robust_zscore_1d, unsweep_ds

"""
Plots the identified candidate in a diagnostic four-column figure. 

The figure shows column one: Drift corrected Stokes I, Q, U, and V
                 column two: Swept spectrum Stokes I, Downsampled Stokes I, Downsampled Stokes V, and Stokes V
                 column three: Raw Stokes I, Q, U, and V
"""

def configure_matplotlib(cfg=None):
    """
    Makes figures pretty by using a close-to-LaTeX font
    """
    if cfg is not None and not getattr(cfg, "use_stix_fonts", True):
        return
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.family"] = "STIXGeneral"


def _trim_window(lo, hi, tdown):
    """
    Trim a (lo, hi) window so its length is a multiple of 'tdown'
    """
    wlen = hi - lo
    wlen_trim = (wlen // tdown) * tdown
    hi_trim = lo + wlen_trim
    if hi_trim <= lo:
        hi_trim = min(hi, lo + tdown)
    return hi_trim


def plot_candidate(ds, var_spectrum, c, cfg, save_path):
    """
    Render and save the diagnostic PDF for a single candidate 
    """
    stokesparam = list(cfg.stokes_labels)
    search_snr = cfg.search_snr
    win = cfg.plot_window

    burst_location = c.location
    burst_width = c.width
    burst_drift = c.drift
    burst_snr = c.snr
    dropped_chans_snr = c.snr_dropped
    f_min, f_max, chans = c.f_min, c.f_max, c.chans

    tdown = max(1, 2 ** (int(np.log10(burst_width) / np.log10(2)) - 2))

    fig = plt.figure(figsize=tuple(cfg.plot_figsize))
    fig.suptitle(
        "S/N: " + str(np.round(burst_snr, 2))
        + "; S/N_r: " + str(np.round(dropped_chans_snr, 2))
        + "; Drift: " + str(burst_drift)
        + "; Width: " + str(burst_width) + " timesteps"
        + "; Image downsampled to: " + str(tdown),
        y=0.91, fontsize=25,
    )

    rows, cols = 11, 7
    widths = [1, 0.01, 0.3, 1, 0.3, 1, 0.01]
    heights = [1, 4, 0.75, 1, 4, 0.75, 1, 4, 0.75, 1, 4]
    gs = gridspec.GridSpec(ncols=cols, nrows=rows, width_ratios=widths,
                           height_ratios=heights, wspace=0.0, hspace=0)

    axlist_timeseries = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[3, 0]),
                         fig.add_subplot(gs[6, 0]), fig.add_subplot(gs[9, 0])]
    axlist_dynspec = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[4, 0]),
                      fig.add_subplot(gs[7, 0]), fig.add_subplot(gs[10, 0])]
    axlist_swepttimeseries = [fig.add_subplot(gs[0, 5]), fig.add_subplot(gs[3, 5]),
                              fig.add_subplot(gs[6, 5]), fig.add_subplot(gs[9, 5])]
    axlist_sweptdynspec = [fig.add_subplot(gs[1, 5]), fig.add_subplot(gs[4, 5]),
                           fig.add_subplot(gs[7, 5]), fig.add_subplot(gs[10, 5])]

    for pol in range(len(stokesparam)):
        # ---- column 1: de-swept ("sweep straight") ----
        unswept_ds = unsweep_ds(ds[pol].astype(np.float64), burst_drift, f_min, f_max, chans)
        unswept_var_spectrum = unsweep_ds(var_spectrum[pol], burst_drift, f_min, f_max, chans)

        sum_1_over_variance = np.nansum(1 / unswept_var_spectrum, axis=0)
        sum_1_over_variance[sum_1_over_variance == 0] = np.nan
        ts = np.nansum(unswept_ds / unswept_var_spectrum, axis=0) / sum_1_over_variance
        ts_std = np.sqrt(1 / sum_1_over_variance)
        ts_snr = robust_zscore_1d(ts / ts_std)

        ts_search = convolve(ts_snr, np.ones(burst_width), mode="same") / (burst_width ** 0.5)
        lim_left, lim_right = np.clip([burst_location - win, burst_location + win],
                                      0, unswept_ds.shape[1])
        unswept_flip = np.flip(unswept_ds, axis=0)
        lim_right_trim = _trim_window(lim_left, lim_right, tdown)

        spec_win = unswept_flip[:, lim_left:lim_right_trim]
        dynspec_down = downsamp(np.nan_to_num(spec_win.astype(np.float32)), tdown=tdown, fdown=1)

        im = axlist_dynspec[pol].imshow(
            np.nan_to_num(dynspec_down), aspect="auto", cmap="viridis",
            vmin=np.nanpercentile(dynspec_down, 5), vmax=np.nanpercentile(dynspec_down, 99),
            interpolation="nearest", extent=[lim_left, lim_right_trim, 0, chans])
        axlist_dynspec[pol].tick_params(axis="both", which="major", labelsize=20)
        axlist_dynspec[pol].set_yticks(np.linspace(0, chans, 5))
        axlist_dynspec[pol].set_yticklabels(np.array(np.linspace(f_min, f_max, 5), dtype="int"))
        xt = np.linspace(lim_left, lim_right_trim, 6)
        axlist_dynspec[pol].set_xticks(xt)
        axlist_dynspec[pol].set_xticklabels(np.round((xt - lim_left) * 4 / 60, 2))
        axlist_dynspec[pol].set_ylabel("Frequency MHZ", fontsize=20)
        axlist_dynspec[pol].set_xlim(lim_left, lim_right_trim)

        axlist_timeseries[pol].plot(np.arange(lim_left, lim_right),
                                    ts_search[lim_left:lim_right], c="k", alpha=0.7)
        axlist_timeseries[pol].plot(np.arange(lim_left, lim_right),
                                    ts_snr[lim_left:lim_right], c="navy", alpha=0.7)
        axlist_timeseries[pol].set_ylabel("S/N", fontsize=10)
        axlist_timeseries[pol].get_xaxis().set_visible(False)
        axlist_timeseries[pol].set_xlim(lim_left, lim_right_trim)
        axlist_timeseries[pol].axvline(x=burst_location, linestyle="--", c="crimson")
        axlist_timeseries[pol].axhline(y=search_snr, linestyle="--", c="darkgray", linewidth=1)
        axlist_timeseries[pol].axhline(y=-search_snr, linestyle="--", c="darkgray")

        cb = fig.colorbar(im, cax=fig.add_subplot(gs[1 + pol * 3, 1]))
        cb.set_label(label="Intenstiy (Jy)", size=20)
        cb.ax.tick_params(labelsize=20)

        axlist_timeseries[pol].set_title("Sweep straight; Stokes " + stokesparam[pol], fontsize=20)
        if pol == 3:
            axlist_dynspec[pol].set_xlabel("Time (min)", fontsize=20)

        # ---- column 3 (gs col 5): swept ("sweep present") ----
        swept_ds = ds[pol]
        swept_var_spectrum = var_spectrum[pol]

        sum_1_over_variance = np.nansum(1 / swept_var_spectrum, axis=0)
        sum_1_over_variance[sum_1_over_variance == 0] = np.nan
        ts = np.nansum(swept_ds / swept_var_spectrum, axis=0) / sum_1_over_variance
        ts_std = np.sqrt(1 / sum_1_over_variance)
        ts_snr = robust_zscore_1d(ts / ts_std)

        ts_search = convolve(ts_snr, np.ones(burst_width), mode="same") / (burst_width ** 0.5)
        lim_left, lim_right = np.clip([burst_location - win, burst_location + win],
                                      0, swept_ds.shape[1])
        swept_flip = np.flip(swept_ds, axis=0)
        lim_right_trim = _trim_window(lim_left, lim_right, tdown)

        spec_win = swept_flip[:, lim_left:lim_right_trim]
        dynspec_down = downsamp(np.nan_to_num(spec_win.astype(np.float32)), tdown=tdown, fdown=1)

        im = axlist_sweptdynspec[pol].imshow(
            np.nan_to_num(dynspec_down), aspect="auto", cmap="viridis",
            vmin=np.nanpercentile(dynspec_down, 5), vmax=np.nanpercentile(dynspec_down, 99),
            interpolation="nearest", extent=[lim_left, lim_right_trim, 0, chans])
        axlist_sweptdynspec[pol].tick_params(axis="both", which="major", labelsize=20)
        axlist_sweptdynspec[pol].set_yticks(np.linspace(0, chans, 5))
        axlist_sweptdynspec[pol].set_yticklabels(np.array(np.linspace(f_min, f_max, 5), dtype="int"))
        xt = np.linspace(lim_left, lim_right_trim, 6)
        axlist_sweptdynspec[pol].set_xticks(xt)
        axlist_sweptdynspec[pol].set_xticklabels(np.round((xt - lim_left) * 4 / 60, 2))
        axlist_sweptdynspec[pol].set_ylabel("Frequency MHZ", fontsize=20)
        axlist_sweptdynspec[pol].set_xlim(lim_left, lim_right_trim)

        axlist_swepttimeseries[pol].plot(np.arange(lim_left, lim_right),
                                         ts_search[lim_left:lim_right], c="k", alpha=0.8)
        axlist_swepttimeseries[pol].plot(np.arange(lim_left, lim_right),
                                         ts_snr[lim_left:lim_right], c="navy", alpha=0.7)
        axlist_swepttimeseries[pol].set_ylabel("S/N", fontsize=10)
        axlist_swepttimeseries[pol].get_xaxis().set_visible(False)
        axlist_swepttimeseries[pol].set_xlim(lim_left, lim_right_trim)
        axlist_swepttimeseries[pol].axvline(x=burst_location, linestyle="--", c="crimson")
        axlist_swepttimeseries[pol].axhline(y=search_snr, linestyle="--", c="darkgray")
        axlist_swepttimeseries[pol].axhline(y=-search_snr, linestyle="--", c="darkgray")
        axlist_swepttimeseries[pol].set_title("Sweep present; Stokes " + stokesparam[pol], fontsize=20)
        if pol == 3:
            axlist_sweptdynspec[pol].set_xlabel("Time (min)", fontsize=20)

    # ---- middle column: I and V sweep maps ----
    ts_len = ds[0].shape[1]
    lim_sweep_left, lim_sweep_right = np.clip([burst_location - win, burst_location + win], 0, ts_len)
    drift_limm = np.int64(np.clip(100, burst_drift * 1.2, 451))

    sweep_list_plot = np.linspace(-drift_limm, drift_limm, 2 * drift_limm + 1)
    swept_time_I = np.zeros((len(sweep_list_plot), ds[0].shape[1]), dtype=np.float32)
    swept_time_V = np.zeros((len(sweep_list_plot), ds[3].shape[1]), dtype=np.float32)

    dynspec_V = ds[3].astype(np.float64)
    dynspec_I = ds[0].astype(np.float64)

    for ii, sweep in enumerate(sweep_list_plot):
        unswept_ds_I = unsweep_ds(dynspec_I, sweep, f_min, f_max, chans)
        unswept_var_I = unsweep_ds(var_spectrum[0], sweep, f_min, f_max, chans)
        sum_inv_I = np.nansum(1 / unswept_var_I, axis=0)
        sum_inv_I[sum_inv_I == 0] = np.nan
        ts_I = (np.nansum(unswept_ds_I / unswept_var_I, axis=0) / sum_inv_I) * np.sqrt(sum_inv_I)
        swept_time_I[ii, :] = robust_zscore_1d(ts_I)

        unswept_ds_V = unsweep_ds(dynspec_V, sweep, f_min, f_max, chans)
        unswept_var_V = unsweep_ds(var_spectrum[3], sweep, f_min, f_max, chans)
        sum_inv_V = np.nansum(1 / unswept_var_V, axis=0)
        sum_inv_V[sum_inv_V == 0] = np.nan
        ts_V = (np.nansum(unswept_ds_V / unswept_var_V, axis=0) / sum_inv_V) * np.sqrt(sum_inv_V)
        swept_time_V[ii, :] = robust_zscore_1d(ts_V)

    downsampled_I = np.zeros_like(swept_time_I)
    downsampled_V = np.zeros_like(swept_time_V)
    for ii in range(len(sweep_list_plot)):
        downsampled_I[ii, :] = convolve(swept_time_I[ii], np.ones(burst_width), mode="same") / (burst_width ** 0.5)
        downsampled_V[ii, :] = convolve(swept_time_V[ii], np.ones(burst_width), mode="same") / (burst_width ** 0.5)

    def _sweep_ax(ax, data, title=None, xlabel=None):
        ax.imshow(data, aspect="auto", interpolation="none", cmap="RdBu_r",
                  vmin=-search_snr, vmax=search_snr)
        ax.set_ylabel("Sweep (minutes)", fontsize=20)
        if title:
            ax.set_title(title, fontsize=20)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=20)
        ax.set_yticks(np.linspace(0, len(sweep_list_plot), 7))
        ax.set_yticklabels(np.int64(np.linspace(-drift_limm, drift_limm, 7) * 4 / 60))
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.axvline(x=burst_location, linestyle=(0, (5, 5)), c="k", linewidth=1, alpha=0.8)
        ax.axhline(y=np.argmin(np.abs(sweep_list_plot - burst_drift)),
                   linestyle=(0, (5, 5)), c="k", linewidth=1, alpha=0.8)
        ax.axhline(y=len(sweep_list_plot) // 2, linestyle="-", c="k", linewidth=1, alpha=0.25)
        ax.set_xlim(lim_sweep_left, lim_sweep_right)

    _sweep_ax(fig.add_subplot(gs[1, 3]), swept_time_I)
    _sweep_ax(fig.add_subplot(gs[4, 3]), downsampled_I, title="Downsampled I")
    _sweep_ax(fig.add_subplot(gs[7, 3]), downsampled_V, title="Downsampled V")
    _sweep_ax(fig.add_subplot(gs[10, 3]), swept_time_V, xlabel="Time (timestep)")

    plt.savefig(save_path, bbox_inches="tight", dpi=cfg.plot_dpi)
    plt.close(fig)
    return save_path
