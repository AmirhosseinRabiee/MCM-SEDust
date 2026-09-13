"""
plot_both.py  --  overlay Model C and Model M MCMC fits on one figure.

Loads the two saved chains (for example star1_C_chain.npy and star1_M_chain.npy),
takes the median parameter set from each, runs DUSTY once for each composition
to produce the model curve, solves the analytic vertical shift, and plots both
curves on top of the star's photometry -- one figure, both models, like the
reference figures from the literature.

Usage:
    python3 plot_both.py --star star1 --data star1.dat
    python3 plot_both.py --star STAR --data STAR.dat \
        --c-chain STAR_C_chain.npy --m-chain STAR_M_chain.npy \
        --template my.inp --dusty-dir /path/to/dusty

Prerequisite: you must have already run the MCMC twice, once with --comp C and
once with --comp M, so the two chain files exist.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dusty_kernel import DustyKernel
from sed_likelihood import load_photometry, load_photometry_fnu_jy, SEDLikelihood
import mcmc_dusty as core   # for auto_ylim (shared axis-limit logic)


def median_params(chain_path):
    """Return median (Tstar, Tdust, tau) from a saved flat chain."""
    chain = np.load(chain_path)
    if chain.ndim != 2 or chain.shape[1] != 3:
        raise ValueError("unexpected chain shape %s in %s"
                         % (chain.shape, chain_path))
    med = np.median(chain, axis=0)
    return float(med[0]), float(med[1]), float(med[2])


def main():
    ap = argparse.ArgumentParser(
        description="Overlay Model C and Model M MCMC fits on one figure.")
    ap.add_argument("--star", required=True,
                    help="star name (used to find chains and label the plot)")
    ap.add_argument("--data", required=True,
                    help="photometry file (txt or csv)")
    ap.add_argument("--c-chain", default=None,
                    help="Model C chain (default: <STAR>_C_chain.npy)")
    ap.add_argument("--m-chain", default=None,
                    help="Model M chain (default: <STAR>_M_chain.npy)")
    ap.add_argument("--dusty-dir", default=".",
                    help="directory where the DUSTY executable runs")
    ap.add_argument("--template", default="template.inp",
                    help="DUSTY input template (.inp)")
    ap.add_argument("--exec", default="dusty", help="DUSTY executable name")
    # photometry file layout
    ap.add_argument("--format", choices=["lamflam", "fnu_jy"], default="lamflam",
                    help="'lamflam' = lambda + lambda*F_lambda (plain .dat files); "
                         "'fnu_jy' = lambda, F_nu[Jy], lambda err, F_nu frac err "
                         "(a Jy-based catalog CSV format)")
    ap.add_argument("--lam-col", type=int, default=0)
    ap.add_argument("--flam-col", type=int, default=1,
                    help="lambdaFlambda column index (format=lamflam only)")
    ap.add_argument("--fnu-col", type=int, default=1,
                    help="F_nu column index (format=fnu_jy only)")
    ap.add_argument("--lam-err-col", type=int, default=2,
                    help="lambda error column index (format=fnu_jy only)")
    ap.add_argument("--fnu-err-col", type=int, default=3,
                    help="F_nu error column index (format=fnu_jy only)")
    ap.add_argument("--fnu-err-absolute", action="store_true",
                    help="set if the F_nu error column is absolute Jy, not fractional")
    ap.add_argument("--err-col", type=int, default=None,
                    help="error column index (format=lamflam only)")
    ap.add_argument("--err-is-log", action="store_true")
    ap.add_argument("--sigma", type=float, default=0.05,
                    help="assumed error in dex if no error column (default 0.05)")
    ap.add_argument("--out", default=None,
                    help="output figure path (default: <STAR>_both.png)")
    args = ap.parse_args()

    c_path = args.c_chain or ("%s_C_chain.npy" % args.star)
    m_path = args.m_chain or ("%s_M_chain.npy" % args.star)
    out_path = args.out or ("%s_both.png" % args.star)

    have_c = os.path.isfile(c_path)
    have_m = os.path.isfile(m_path)
    if not (have_c or have_m):
        print("ERROR: neither chain found.")
        print("  expected %s and/or %s" % (c_path, m_path))
        print("  run the MCMC first:  python3 mcmc_dusty.py --comp C ...")
        sys.exit(1)
    if not have_c:
        print("note: %s missing -- plotting Model M only." % c_path)
    if not have_m:
        print("note: %s missing -- plotting Model C only." % m_path)

    # load data + build kernel/likelihood (shared by both compositions)
    if args.format == "fnu_jy":
        is_csv = args.data.lower().endswith(".csv")
        data = load_photometry_fnu_jy(
            args.data, lam_col=args.lam_col, fnu_col=args.fnu_col,
            lam_err_col=args.lam_err_col, fnu_err_col=args.fnu_err_col,
            fnu_err_is_fractional=not args.fnu_err_absolute,
            delimiter="," if is_csv else None)
    else:
        data = load_photometry(args.data, sigma_log=args.sigma,
                               lam_col=args.lam_col, flam_col=args.flam_col,
                               err_col=args.err_col, err_is_log=args.err_is_log)
    kernel = DustyKernel(dusty_dir=args.dusty_dir,
                         template_inp=args.template, exec_name=args.exec)
    like = SEDLikelihood(data, kernel=kernel)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    summaries = []
    curves = []          # (log10 lambda, log10 lamFlam) of each plotted model
    for comp, chain_path, color, style, label_prefix in [
        ("C", c_path, "red",   "--", "Model C"),
        ("M", m_path, "green", "-",  "Model M"),
    ]:
        if not os.path.isfile(chain_path):
            continue
        Tstar, Tdust, tau = median_params(chain_path)
        print("running DUSTY at %s median: Tstar=%.0f Tdust=%.0f tau=%.3f ..."
              % (label_prefix, Tstar, Tdust, tau))
        lam_m, ftot = kernel.run(Tstar, Tdust, tau, composition=comp)
        if lam_m is None:
            print("  WARNING: DUSTY failed for %s median; skipping curve." % label_prefix)
            continue
        res = like.evaluate_model(lam_m, ftot)
        if res is None:
            print("  WARNING: model doesn't cover data wavelengths; skipping.")
            continue
        shift, chi2 = res["shift"], res["chi2"]
        summaries.append((label_prefix, Tstar, Tdust, tau, shift, chi2))
        g = ftot > 0
        curve_x = np.log10(lam_m[g])
        curve_y = np.log10(ftot[g]) + shift
        curves.append((curve_x, curve_y))   # kept for the y-limit calculation
        ax.plot(curve_x, curve_y,
                color=color, ls=style, lw=2,
                label="%s  Tstar=%.0f Tdust=%.0f tau=%.3f  shift=%.3f"
                      % (label_prefix, Tstar, Tdust, tau, shift))

    # raw fractional error bars (a common plotting convention). Display only --
    # the fit always uses the correct dex error internally.
    yerr = like.flam_frac if getattr(like, "flam_frac", None) is not None \
        else like.sigma_log
    xerr = like.lam_err if getattr(like, "lam_err", None) is not None else None
    ax.errorbar(np.log10(like.lam), like.logflam,
                xerr=xerr, yerr=yerr,
                fmt="ko", ms=7, capsize=0, label="%s observed" % args.star,
                zorder=5)
    ax.set_xlim(-0.5, 1.5)
    # y-limits adapt to this star: span the data and BOTH model curves
    if curves:
        all_x = np.concatenate([c[0] for c in curves])
        all_y = np.concatenate([c[1] for c in curves])
    else:
        all_x = all_y = None
    ax.set_ylim(*core.auto_ylim(like.logflam, yerr=yerr,
                                model_logflam=all_y, model_logx=all_x,
                                xlim=(-0.5, 1.5)))
    ax.set_xlabel(r"$\log_{10}\lambda\ [\mu m]$")
    ax.set_ylabel(r"$\log_{10}(\lambda F_\lambda)$")
    ax.legend(loc="lower center", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("%s : MCMC median fits (Model C vs Model M)" % args.star)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("saved %s" % out_path)

    if summaries:
        print("\nSummary:")
        print("  model   Tstar   Tdust    tau    shift    chi2")
        for s in summaries:
            print("  %-5s  %6.0f  %6.0f  %6.3f  %7.3f  %7.2f" % s)
        if len(summaries) == 2:
            # whichever has lower chi2 is the better fit
            better = min(summaries, key=lambda r: r[5])
            diff = abs(summaries[0][5] - summaries[1][5])
            print("\n  >>> better fit: %s (chi2 lower by %.2f)" % (better[0], diff))


if __name__ == "__main__":
    main()
