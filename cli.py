"""
Command-line interface for DriftSearch.

Usage:
    python -m DriftSearch INPUT_DIR OUTPUT_DIR [options]
"""

import argparse

from .config import DriftSearchConfig
from .search import DriftSearch


def build_parser():
    p = argparse.ArgumentParser(
        prog="DriftSearch",
        description="Search for drifting broadband radio bursts.",
    )
    p.add_argument("input_dir", help="Directory of target dynamic-spectrum FITS files")
    p.add_argument("output_dir", help="Directory to write candidate PDFs")
    p.add_argument("--off-dir", default=None, help="Off-target dir (default: input_dir)")
    p.add_argument("--file-glob", default="*.fits")

    p.add_argument("--search-snr", type=float, default=9.0)
    p.add_argument("--search-stokes", default="I", help="Stokes param to search (I/Q/U/V)")
    p.add_argument("--weight-stokes", default="V", help="Stokes param used to flag bad spectra")

    p.add_argument("--drift-min", type=float, default=0.0)
    p.add_argument("--drift-max", type=float, default=450.0)
    p.add_argument("--drift-steps", type=int, default=451)

    p.add_argument("--bad-spectra-frac", type=float, default=0.10)
    p.add_argument("--merge-drop", type=float, default=2.5)
    p.add_argument("--reject-top-k-frac", type=float, default=0.01)

    p.add_argument("--no-plots", action="store_true", help="Find candidates without writing PDFs (almost never worth it)")
    p.add_argument("--plot-window", type=int, default=200)
    p.add_argument("--plot-dpi", type=int, default=128)
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = DriftSearchConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        off_dir=args.off_dir,
        file_glob=args.file_glob,
        search_snr=args.search_snr,
        search_stokes=args.search_stokes,
        weight_stokes=args.weight_stokes,
        drift_min=args.drift_min,
        drift_max=args.drift_max,
        drift_steps=args.drift_steps,
        bad_spectra_frac=args.bad_spectra_frac,
        merge_drop=args.merge_drop,
        reject_top_k_frac=args.reject_top_k_frac,
        make_plots=not args.no_plots,
        plot_window=args.plot_window,
        plot_dpi=args.plot_dpi)
    
    candidates = DriftSearch(cfg).run()
    print(f"Found {len(candidates)} candidate(s).")

    return candidates

if __name__ == "__main__":
    main()
