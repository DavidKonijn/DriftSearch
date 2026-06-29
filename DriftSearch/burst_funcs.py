import numpy as np

from numba import njit
from scipy.signal import convolve
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.ndimage import label as nd_label

def watershed_merge_labels(S_s: np.ndarray,Th: float,Tl: float,min_distance: int = 5,merge_drop: float = 2.0):
    """
    Parameters:
                S_s (2D float array) : SNR map (higher = more signal).
                Th (float)           : Peak threshold for markers.
                Tl (float)           : Lower threshold for mask connectivity.
                min_distance (int)   : peak_local_max min_distance (over-seeding is fine; we'll merge).
                merge_drop (float)   : Merge if min(peakA, peakB) - saddle < merge_drop.

    Returns
        labels_merged (2D int array) : 0 background, 1-N merged regions (bursts)
        peak_val (1D float array)    : Peak value per original watershed region id (index=rid); for merged labels you can recompute if needed.
    """
    S_s = S_s.astype(np.float32, copy=False)
    mask = S_s >= Tl

    coords = peak_local_max(S_s,threshold_abs=Th,labels=mask.astype(np.uint8),min_distance=min_distance,exclude_border=False)

    if coords.size == 0:
        return np.zeros_like(S_s, dtype=np.int32), np.array([0], dtype=np.float32)

    markers = np.zeros_like(S_s, dtype=np.int32)

    for i, (yy, xx) in enumerate(coords, start=1):
        markers[yy, xx] = i

    markers, _ = nd_label(markers > 0)
    labels_ws = watershed(-S_s, markers=markers, mask=mask, compactness=0)
    nreg = int(labels_ws.max())

    if nreg == 0:
        return labels_ws.astype(np.int32, copy=False), np.array([0], dtype=np.float32)

    # peak per region
    peak_val = np.full(nreg + 1, -np.inf, dtype=np.float32)
    for rid in range(1, nreg + 1):
        m = labels_ws == rid
        if np.any(m):
            peak_val[rid] = float(np.max(S_s[m]))

    # saddle estimation on 4-neighborhood boundaries: (a,b) -> max boundary level
    saddle = {}
    H, W = labels_ws.shape

    # right neighbors
    a = labels_ws[:, :-1]
    b = labels_ws[:, 1:]
    m = (a != b) & (a > 0) & (b > 0)
    if np.any(m):
        aa = a[m].ravel()
        bb = b[m].ravel()
        svals = np.maximum(S_s[:, :-1][m], S_s[:, 1:][m]).ravel()
        for ai, bi, sv in zip(aa, bb, svals):
            if ai > bi:
                ai, bi = bi, ai
            key = (int(ai), int(bi))
            prev = saddle.get(key, -np.inf)
            if sv > prev:
                saddle[key] = float(sv)

    # down neighbors
    a = labels_ws[:-1, :]
    b = labels_ws[1:, :]
    m = (a != b) & (a > 0) & (b > 0)
    if np.any(m):
        aa = a[m].ravel()
        bb = b[m].ravel()
        svals = np.maximum(S_s[:-1, :][m], S_s[1:, :][m]).ravel()
        for ai, bi, sv in zip(aa, bb, svals):
            if ai > bi:
                ai, bi = bi, ai
            key = (int(ai), int(bi))
            prev = saddle.get(key, -np.inf)
            if sv > prev:
                saddle[key] = float(sv)

    # union-find for merging
    parent = np.arange(nreg + 1, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # sort merges by shallowest separation first
    pairs = []
    for (ra, rb), sadd in saddle.items():
        drop = min(peak_val[ra], peak_val[rb]) - sadd
        pairs.append((drop, ra, rb))
    pairs.sort(key=lambda t: t[0])

    for drop, ra, rb in pairs:
        if drop < merge_drop:
            union(ra, rb)

    # relabel to contiguous ids
    root = np.array([find(i) for i in range(nreg + 1)], dtype=np.int32)
    uniq = {r: i for i, r in enumerate(sorted(set(root[1:])), start=1)}

    labels_merged = np.zeros_like(labels_ws, dtype=np.int32)
    for rid in range(1, nreg + 1):
        labels_merged[labels_ws == rid] = uniq[root[rid]]

    return labels_merged, peak_val

def _regions_as_flat_indices(labels_merged: np.ndarray):
    """
    Maps multiple watershed index regions as a dictionary, to later remove overlapping larger widths bursts
    """
    flat = labels_merged.ravel()
    nz = np.flatnonzero(flat)
    if nz.size == 0:
        return {}

    labs = flat[nz]
    order = np.argsort(labs, kind="mergesort")
    nz_sorted = nz[order]
    labs_sorted = labs[order]

    uniq, start_idx, counts = np.unique(labs_sorted, return_index=True, return_counts=True)
    regions = {}
    for rid, s, c in zip(uniq, start_idx, counts):
        regions[int(rid)] = nz_sorted[s:s + c]
    return regions


def dedup_smallest_width_wins_by_label_overlap(per_width_results):
    """
    Because we run our watershed for multiple boxcar widths (basically downsampling), 
    this function prioritises overlapping smaller widths bursts, 
    to not over-downsample bursts that are already reaching the SNR_threshold 
    """
    per_width_results = sorted(per_width_results, key=lambda d: d["width"])
    H, W = per_width_results[0]["labels_merged"].shape
    owner = np.zeros(H * W, dtype=np.int32)

    unique_indices = []
    kept_id = 0

    for res in per_width_results:
        labels = res["labels_merged"]
        S_s = res["S_s"]
        width = res["width"]

        if labels.shape != (H, W):
            raise ValueError("All labels_merged must have identical shape")

        regions = _regions_as_flat_indices(labels)

        for rid, flat_idx in regions.items():
            # overlap test: any shared pixel with a kept region
            if np.any(owner[flat_idx] != 0):
                continue

            # keep
            kept_id += 1
            owner[flat_idx] = kept_id

            yy, xx = np.unravel_index(flat_idx, (H, W))

            region_vals = S_s[yy, xx]
            imax = int(np.argmax(region_vals))
            y_peak = int(yy[imax])
            x_peak = int(xx[imax])
            snr_peak = float(region_vals[imax])

            unique_indices.append((x_peak, int(width), y_peak, snr_peak))

    return unique_indices, owner.reshape(H, W)


def robust_zscore_1d(x,eps=1e-6,clip_q=95.0):
    """
    Renormalises the 1D freq-integrated array such that the average of a spectrum will not increase substantially over a candidate burst width
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.nanmedian(x)
    x0 = x - m

    # winsorize large excursions (reduces burst impact on sigma)
    absx = np.abs(x0)
    thr = np.nanpercentile(absx, clip_q)
    if np.isfinite(thr) and thr > 0:
        xw = np.clip(x0, -thr, thr)
    else:
        xw = x0

    sig = np.nan

    if not np.isfinite(sig) or sig < eps:
        mad = np.nanmedian(np.abs(xw))
        sig = 1.4826 * mad

    if not np.isfinite(sig) or sig < eps:
        sig = np.nanstd(xw)

    if not np.isfinite(sig) or sig < eps:
        sig = 1.0

    return (x0 / sig).astype(np.float32)
    

def remove_brightest_channels_reject(dynspec_I, var_ds, burst_drift, burst_location, burst_width,f_min, f_max, chans, search_snr, top_k=None):
    """
    After dropping brightest k channels, SNR should still > (SNR_threshold - 1.5),
    
    Return: the dropped_chan_snr, and True if snr_drop < (SNR_threshold - 1.5)
    """
    if top_k is None:
        top_k = max(1, np.int32(np.round(chans/100)))

    unswept_I   = unsweep_ds(dynspec_I, burst_drift, f_min, f_max, chans)
    unswept_var = unsweep_ds(var_ds,    burst_drift, f_min, f_max, chans)

    inv_var = np.where(np.isfinite(unswept_var) & (unswept_var > 0), 1.0 / unswept_var, np.nan)

    D = np.nansum(inv_var, axis=0)
    D[D == 0] = np.nan

    num = unswept_I * inv_var
    ts_mean = np.nansum(num, axis=0) / D
    ts_std  = np.sqrt(1.0 / D)
    ts      = ts_mean / ts_std
    ts      = robust_zscore_1d(ts)

    snr_full = (convolve(ts, np.ones(burst_width), mode='same') / np.sqrt(burst_width))[burst_location]    
    nfreq, ntime = num.shape

    half = max(1, int(burst_width // 2))
    W0 = int(np.clip(burst_location - half, 0, ntime - 1))
    W1 = int(np.clip(burst_location + half, 1, ntime))
    if W1 <= W0:
        W0 = max(0, burst_location - 1)
        W1 = min(ntime, burst_location + 1)

    w_t = 1.0 / np.sqrt(D)
    C_f = np.nansum(num[:, W0:W1] * w_t[W0:W1], axis=1) 

    # Pick top-k channels by C_f (ignore NaN/-inf)
    k = min(int(top_k), nfreq - 1)
    C_safe = np.where(np.isfinite(C_f), C_f, -np.inf)
    top_idx = np.argpartition(C_safe, -k)[-k:]

    keep = np.ones(nfreq, dtype=bool)
    keep[top_idx] = False

    D_d = np.nansum(inv_var[keep, :], axis=0)
    D_d[D_d == 0] = np.nan

    ts_mean_d = np.nansum(num[keep, :], axis=0) / D_d
    ts_std_d  = np.sqrt(1.0 / D_d)
    ts_d      = ts_mean_d / ts_std_d
    ts_d      = robust_zscore_1d(ts_d)

    snr_drop = (convolve(ts_d, np.ones(burst_width), mode='same') / np.sqrt(burst_width))[burst_location]
    return snr_drop, (not np.isfinite(snr_drop)) or (np.abs(snr_drop) < snr_full-1.5)


@njit
def downsamp(ds,tdown=1,fdown=1):
    tdown=int(tdown)
    fdown=int(fdown)

    if fdown!=1:
        ds=ds.reshape(ds.shape[0]//fdown, fdown,ds.shape[-1]).sum(axis=1)
    if tdown!=1:
        ds=ds.reshape(ds.shape[0], ds.shape[-1]//tdown, tdown).sum(axis=2)
    return ds


@njit
def unsweep_ds(data, sweep_steps, f_min, f_max, chans):

    freqs = np.linspace(f_min, f_max, chans)
    x = sweep_steps/(1/f_min - 1/f_max) * (1/freqs - 1/f_max)

    delay_bins = -np.rint(x)
    unswept_data = 1*data

    for i in range(unswept_data.shape[0]):
        unswept_data[i] = np.roll(unswept_data[i], int(delay_bins[i]))

    return unswept_data


def matched_filter_snr(dynspec, var_spectrum, sweep_steps, f_min, f_max, chans):
    unswept = unsweep_ds(dynspec, sweep_steps, f_min, f_max, chans)
    unswept_var = unsweep_ds(var_spectrum, sweep_steps, f_min, f_max, chans)

    sum_inv = np.nansum(1.0 / unswept_var, axis=0)
    sum_inv[sum_inv == 0] = np.nan

    ts = np.nansum(unswept / unswept_var, axis=0) / sum_inv
    ts_std = np.sqrt(1.0 / sum_inv)
    return ts / ts_std
