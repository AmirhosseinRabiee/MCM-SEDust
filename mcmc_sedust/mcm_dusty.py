"""
mcm_dusty.py

Fits one star: samples the posterior of the three physical parameters

    Tstar  stellar blackbody temperature, K
    Tdust  dust temperature at the inner boundary, K
    tau    optical depth at the template's reference wavelength

with emcee, where every likelihood evaluation runs the actual DUSTY code
through DustyKernel. The vertical shift is not sampled -- it's solved
analytically inside the likelihood (see sed_likelihood.py), so the chain
really is just 3-dimensional.

Dust composition (C or M) is fixed for a given run; fit each one as a
separate chain and compare them afterwards (plot_both.py does this).

Run from the folder where DUSTY lives, with these modules and your data
alongside it:

    # time one DUSTY call and get a rough runtime estimate first
    python3 mcmc_dusty.py --comp M --mode quick

    # the real run
    python3 mcmc_dusty.py --comp M --mode full --nwalkers 24 --nburn 300 --nprod 1500

Outputs are named by star + composition, e.g. for star1 / M:
    star1_M_chain.npy      flat posterior samples
    star1_M_corner.png     corner plot (with a reference line marked, if
                           you've added one to KNOWN near the top of this file)
    star1_M_bestfit.png    median-model SED over the data

Needs numpy, matplotlib, emcee, corner.
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dusty_kernel import DustyKernel
from sed_likelihood import load_photometry, load_photometry_fnu_jy, SEDLikelihood

# defaults for a single-star run -- override any of these from the
# command line. BOUNDS is the prior box the walkers can't leave.
# KNOWN is empty by default; add your own reference values here if you
# have something to compare against (never used to seed the walkers,
# so it doesn't bias the fit -- see the manual for the format).
DUSTY_DIR    = "."
TEMPLATE_INP = "template.inp"
EXEC_NAME    = "dusty"
DATA_FILE    = "star1.dat"
STAR_NAME    = "star1"
ASSUMED_SIGMA_LOG = 0.05            # dex; placeholder until real errors exist

# Priors / search ranges -- SET THESE for your target galaxy and population.
BOUNDS = {
    "Tstar": (2000.0, 4000.0),
    "Tdust": (100.0, 1400.0),
    "tau":   (0.01, 2.0),
}

# Optional reference solutions, keyed by STAR NAME then composition, e.g.:
#   KNOWN = {"star1": {"C": dict(Tstar=2100.0, Tdust=100.0,  tau=0.06),
#                       "M": dict(Tstar=2100.0, Tdust=1100.0, tau=0.04)}}
# Leave empty if you have nothing to compare against.
KNOWN = {}
# ======================================================================= #


def build(composition):
    data = load_photometry(DATA_FILE, sigma_log=ASSUMED_SIGMA_LOG)
    kernel = DustyKernel(dusty_dir=DUSTY_DIR, template_inp=TEMPLATE_INP,
                         exec_name=EXEC_NAME)
    like = SEDLikelihood(data, kernel=kernel, bounds=BOUNDS,
                         composition=composition)
    return data, kernel, like


def init_walkers(nwalkers, rng):
    """Initialize walkers UNIFORMLY across the prior box (honest, unbiased).
    Walkers that start where DUSTY fails simply get -inf and are pulled in."""
    cols = []
    for name in ("Tstar", "Tdust", "tau"):
        lo, hi = BOUNDS[name]
        cols.append(rng.uniform(lo, hi, nwalkers))
    return np.column_stack(cols)


def time_one_call(kernel, composition):
    """Run DUSTY once at the centre of the box and report seconds."""
    mid = {k: 0.5 * (v[0] + v[1]) for k, v in BOUNDS.items()}
    t0 = time.time()
    lam, ftot = kernel.run(mid["Tstar"], mid["Tdust"], mid["tau"],
                           composition=composition)
    dt = time.time() - t0
    ok = lam is not None
    return dt, ok


# NOTE: emcee only evaluates half the ensemble at once (the stretch
# move updates one half using the other), so a single chain can never
# usefully use more than nwalkers/2 cores. For a whole catalog,
# parallelise over stars instead (fit_catalog.py --parallel-stars).
def run_chain(like, nwalkers, nburn, nprod, rng, nproc=None, dt_per_call=None,
              quiet=False):
    """Run burn-in + production.

    nproc > 1  : parallel walkers via a spawn Pool (single-star mode).
    nproc <= 1 : SERIAL, no Pool at all. This is required when run_chain is
                 itself called inside a worker of a star-level pool
                 (fit_catalog.py --parallel-stars), because a pool worker
                 cannot open its own nested pool.
    quiet      : suppress the heartbeat/step printing (used when many stars
                 run at once so the terminal isn't flooded).
    """
    import emcee
    import multiprocessing as mp

    ndim = 3
    p0 = init_walkers(nwalkers, rng)
    ncores = nproc if nproc is not None else _cpucount()

    # emcee's stretch move splits the ensemble in half: it must finish one half
    # before starting the other, so at most nwalkers/2 likelihood calls can run
    # simultaneously, NO MATTER how many cores are available. Cores beyond that
    # sit idle. The time estimates below use this true effective count.
    eff_cores = max(1, min(ncores, nwalkers // 2))
    if not quiet and eff_cores < ncores:
        print("\n  NOTE: %d walkers means emcee can only run %d DUSTY calls at "
              "once, so only %d of your %d cores will be used."
              % (nwalkers, eff_cores, eff_cores, ncores))
        print("        For full use of %d cores, use at least %d walkers "
              "(nwalkers >= 2 x ncores)." % (ncores, 2 * ncores))

    def _drive(sampler):
        total_calls_burn = nburn * nwalkers
        eta_burn = (total_calls_burn * dt_per_call / eff_cores) \
            if dt_per_call else None
        if not quiet:
            print("\n  --- burn-in: %d steps x %d walkers = %d DUSTY runs "
                  "on %d cores%s ---"
                  % (nburn, nwalkers, total_calls_burn, eff_cores,
                     ("  (est. %s)" % _fmt_dur(eta_burn)) if eta_burn else ""))
        state = _run_with_progress(sampler, p0, nburn, nwalkers,
                                   ncores, dt_per_call, "burn", quiet=quiet)
        sampler.reset()

        total_calls_prod = nprod * nwalkers
        eta_prod = (total_calls_prod * dt_per_call / eff_cores) \
            if dt_per_call else None
        if not quiet:
            print("\n  --- production: %d steps x %d walkers = %d DUSTY runs "
                  "on %d cores%s ---"
                  % (nprod, nwalkers, total_calls_prod, eff_cores,
                     ("  (est. %s)" % _fmt_dur(eta_prod)) if eta_prod else ""))
        _run_with_progress(sampler, state, nprod, nwalkers,
                           ncores, dt_per_call, "prod", quiet=quiet)
        return sampler

    if ncores <= 1:
        # serial: emcee evaluates walkers one at a time in this process.
        sampler = emcee.EnsembleSampler(nwalkers, ndim, like.log_prob)
        return _drive(sampler)

    # Use 'spawn' start method: each worker is a fresh Python interpreter.
    # This is essential for DUSTY because:
    #   1. Workers must not share file handles or DUSTY run directories.
    #   2. 'fork' can cause deadlocks when the parent has open sockets/threads.
    #   3. DustyKernel.__getstate__ resets _rundirs so each worker builds its
    #      own fresh run directory -- this is what makes parallelism actually work.
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=nproc) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, like.log_prob, pool=pool)
        return _drive(sampler)


def _run_with_progress(sampler, start_state, nsteps, nwalkers,
                       ncores, dt_per_call, label, quiet=False):
    """Iterate emcee step-by-step and print live progress with ETA + finish time.

    Two channels of feedback so the user always sees that DUSTY is running:
      - a 'step' line every time a step (= nwalkers DUSTY calls) completes
      - a background 'heartbeat' that prints elapsed seconds + a spinner every
        3 s, so the screen updates even mid-step on slow DUSTY calls
    quiet=True disables all of this (used when many stars run in parallel).
    """
    import datetime as _dt
    import threading

    t0 = time.time()
    state = start_state

    if quiet:
        # silent drive-through: no heartbeat, no per-step lines
        for state in sampler.sample(start_state, iterations=nsteps):
            pass
        return state

    print("    [%s] DUSTY running: starting %d steps x %d walkers ..."
          % (label, nsteps, nwalkers), flush=True)

    # ---- heartbeat thread ---------------------------------------------- #
    stop_flag = threading.Event()
    spinner = ["|", "/", "-", "\\"]

    def heartbeat():
        i = 0
        while not stop_flag.wait(3.0):  # tick every 3 s
            el = time.time() - t0
            print("    [%s] DUSTY running %s  elapsed %s"
                  % (label, spinner[i % 4], _fmt_dur(el)), flush=True)
            i += 1

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()
    # -------------------------------------------------------------------- #

    try:
        for i, state in enumerate(sampler.sample(start_state, iterations=nsteps), 1):
            now = time.time()
            done = i * nwalkers
            el = now - t0
            rate = done / max(0.001, el)
            steps_left = nsteps - i
            calls_left = steps_left * nwalkers
            eta_s = calls_left / max(0.001, rate)
            finish_at = _dt.datetime.now() + _dt.timedelta(seconds=eta_s)
            accept = float(np.mean(sampler.acceptance_fraction))
            # print a real status line on every step completion (no rate-limit)
            print("    [%s] step %d/%d   runs %d/%d   %.2f /s   "
                  "elapsed %s   remaining %s   ETA %s   accept %.2f"
                  % (label, i, nsteps, done, nsteps * nwalkers, rate,
                     _fmt_dur(el), _fmt_dur(eta_s),
                     finish_at.strftime("%H:%M:%S"), accept),
                  flush=True)
    finally:
        stop_flag.set()
        hb.join(timeout=4.0)

    return state


def _fmt_dur(s):
    """Format a duration in seconds as h:mm:ss or m:ss."""
    if s is None:
        return "?"
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return "%d:%02d:%02d" % (h, m, sec)
    return "%d:%02d" % (m, sec)


def auto_ylim(logflam_data, yerr=None, model_logflam=None, model_logx=None,
              xlim=(-0.5, 1.5), pad=0.45, min_height=1.6, legend_room=0.55):
    """Pick y-axis limits that frame THIS star's data instead of a fixed box.

    Uses the data points (extended by their error bars) plus the part of the
    model curve that is actually inside the plotted x-range, then pads a little.
    Extra room is left at the bottom for the legend.

    logflam_data : log10(lamFlam) of the photometry
    yerr         : half-length of the y error bars (same units), optional
    model_logflam, model_logx : the plotted model curve, optional
    min_height   : never produce an axis shorter than this (dex), so a star
                   whose points nearly coincide doesn't get absurdly zoomed.
    legend_room  : extra padding at the bottom for the legend box.
    """
    lo_vals = [np.asarray(logflam_data, float)]
    hi_vals = [np.asarray(logflam_data, float)]
    if yerr is not None:
        e = np.asarray(yerr, float)
        lo_vals.append(np.asarray(logflam_data, float) - e)
        hi_vals.append(np.asarray(logflam_data, float) + e)

    # include only the model points that fall inside the visible x-range
    if model_logflam is not None and model_logx is not None:
        mx = np.asarray(model_logx, float)
        my = np.asarray(model_logflam, float)
        vis = (mx >= xlim[0]) & (mx <= xlim[1]) & np.isfinite(my)
        if np.any(vis):
            lo_vals.append(my[vis])
            hi_vals.append(my[vis])

    lo = float(np.nanmin(np.concatenate([np.atleast_1d(v) for v in lo_vals])))
    hi = float(np.nanmax(np.concatenate([np.atleast_1d(v) for v in hi_vals])))

    lo -= pad + legend_room     # room for the legend at the bottom
    hi += pad

    # enforce a minimum height so near-flat SEDs are not over-zoomed
    if (hi - lo) < min_height:
        mid = 0.5 * (hi + lo)
        lo, hi = mid - min_height / 2.0, mid + min_height / 2.0
    return lo, hi


# turns a finished sampler into the chain/summary/corner/bestfit files
def summarize(sampler, like, kernel, composition, tag):
    # Derive this star's display name from the tag rather than the module
    # global STAR_NAME. fit_catalog.py calls summarize() directly (bypassing
    # main(), which is what sets STAR_NAME), so the global would otherwise stay
    # at its default and every catalog plot would show the wrong star name.
    # tag looks like "results/star1_M" (or "star1_M_quick") -> "star1".
    star_name = os.path.basename(tag)
    if star_name.endswith("_quick"):
        star_name = star_name[:-len("_quick")]
    if star_name.endswith("_" + str(composition)):
        star_name = star_name[:-(len(str(composition)) + 1)]
    labels = ["Tstar", "Tdust", "tau"]
    chain = sampler.get_chain(flat=True)
    np.save("%s_chain.npy" % tag, chain)
    print("\n  saved %s_chain.npy   (%d samples)" % (tag, chain.shape[0]))
    print("  mean acceptance fraction: %.3f" % np.mean(sampler.acceptance_fraction))
    try:
        ac = sampler.get_autocorr_time(tol=0)
        print("  autocorrelation times:", np.round(ac, 1),
              " (want chain >> these)")
    except Exception as e:
        print("  autocorr:", e)

    best = {}
    print("\n  === %s posterior (median, 16-84%%) ===" % tag)
    for i, l in enumerate(labels):
        lo, med, hi = np.percentile(chain[:, i], [16, 50, 84])
        best[l] = med
        print("    %-6s = %9.3f  (+%.3f / -%.3f)" % (l, med, hi - med, med - lo))

    lam_m, ftot = kernel.run(best["Tstar"], best["Tdust"], best["tau"],
                             composition=composition)
    res = like.evaluate_model(lam_m, ftot)
    print("    shift  = %9.5f   chi2 = %.2f" % (res["shift"], res["chi2"]))

    # M_dot etc. from the .out of the median-parameter run just above
    outres = getattr(kernel, "last_results", None)
    if outres is not None:
        print("    Mdot   = %9.3e Msun/yr   Ve = %.2f km/s"
              % (outres["Mdot"], outres["Ve"]))
        print("      (DUSTY assumes L = 1e4 Lsun; physical Mdot scales as "
              "sqrt(L/1e4 Lsun))")
    else:
        print("    Mdot   : not available (.out not parsed)")

    # write a plain-text summary file next to the chain
    stats = {}
    for i, l in enumerate(labels):
        lo, med, hi = np.percentile(chain[:, i], [16, 50, 84])
        stats[l] = (med, med - lo, hi - med)
    with open("%s_summary.txt" % tag, "w") as f:
        f.write("MCMC fit summary : %s\n" % tag)
        nme = getattr(like, "n_missing_err", 0)
        if nme:
            import sed_likelihood as _sl
            f.write("CAVEAT: %d photometry point(s) had NO error in the CSV; "
                    "an ASSUMED %.0f%% flux error was used for them.\n"
                    % (nme, 100.0 * _sl.MISSING_FLUX_ERR))
        f.write("samples = %d   acceptance = %.3f\n\n"
                % (chain.shape[0], float(np.mean(sampler.acceptance_fraction))))
        f.write("parameter   median      -err        +err\n")
        for l in labels:
            f.write("%-9s %10.4f  %10.4f  %10.4f\n"
                    % (l, stats[l][0], stats[l][1], stats[l][2]))
        f.write("\nshift = %.5f\nchi2  = %.3f\n" % (res["shift"], res["chi2"]))
        if outres is not None:
            f.write("Mdot  = %.4e Msun/yr  (DUSTY fiducial L=1e4 Lsun; "
                    "physical Mdot = this x sqrt(L/1e4 Lsun))\n" % outres["Mdot"])
            f.write("Ve    = %.3f km/s\nMmax  = %.3f Msun\n"
                    % (outres["Ve"], outres["Mmax"]))
        else:
            f.write("Mdot  = NA (.out not parsed)\n")
    print("  saved %s_summary.txt" % tag)

    # corner plot
    try:
        import corner
        truths = None
        # Only mark truth lines for a star you have added to KNOWN yourself
        # (SECTION 1). For any other star, no truth lines are drawn.
        star_known = KNOWN.get(star_name, {})
        if star_known.get(composition):
            truths = [star_known[composition][l] for l in labels]
        fig = corner.corner(chain, labels=labels, truths=truths,
                            quantiles=[0.16, 0.5, 0.84], show_titles=True)
        fig.savefig("%s_corner.png" % tag, dpi=130)
        plt.close(fig)
        print("  saved %s_corner.png" % tag)
    except Exception as e:
        print("  corner plot skipped:", e)

    # best-fit overlay
    g = ftot > 0
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(np.log10(lam_m[g]), np.log10(ftot[g]) + res["shift"], "g-", lw=2,
            label="median: Tstar=%.0f Tdust=%.0f tau=%.3f"
                  % (best["Tstar"], best["Tdust"], best["tau"]))
    # error bars: both axes use the raw fractional error (CSV cols 3 and 4),
    # matching a common plotting convention. Display only -- the fit
    # always uses the correct dex error internally.
    yerr = like.flam_frac if getattr(like, "flam_frac", None) is not None \
        else like.sigma_log
    xerr = like.lam_err if getattr(like, "lam_err", None) is not None else None
    ax.errorbar(np.log10(like.lam), like.logflam,
                xerr=xerr, yerr=yerr,
                fmt="ko", ms=7, capsize=0, label="%s observed" % star_name)
    ax.set_xlim(-0.5, 1.5)
    # y-limits adapt to this star (data + model), instead of a fixed box
    ax.set_ylim(*auto_ylim(like.logflam, yerr=yerr,
                           model_logflam=np.log10(ftot[g]) + res["shift"],
                           model_logx=np.log10(lam_m[g]),
                           xlim=(-0.5, 1.5)))
    ax.set_xlabel(r"$\log_{10}\lambda\ [\mu m]$")
    ax.set_ylabel(r"$\log_{10}(\lambda F_\lambda)$")
    ax.legend(loc="lower center", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("%s : MCMC median fit (model %s)" % (star_name, composition))
    fig.tight_layout()
    fig.savefig("%s_bestfit.png" % tag, dpi=130)
    plt.close(fig)
    print("  saved %s_bestfit.png" % tag)
    return best


# --mode defaults to 'quick' -- a short timing run, not a real fit
def main():
    global DATA_FILE, STAR_NAME, DUSTY_DIR, TEMPLATE_INP, EXEC_NAME, BOUNDS

    ap = argparse.ArgumentParser(
        description="MCMC SED fitting with the Fortran DUSTY code as the kernel.")
    ap.add_argument("--comp", choices=["C", "M"], required=True,
                    help="dust composition: C (carbon) or M (oxygen-rich)")
    ap.add_argument("--mode", choices=["quick", "full"], default="quick")
    # --- inputs that differ from machine to machine ---
    ap.add_argument("--data", default=DATA_FILE,
                    help="photometry file (txt or csv)")
    ap.add_argument("--star", default=STAR_NAME, help="star name (used in output filenames)")
    ap.add_argument("--dusty-dir", default=DUSTY_DIR,
                    help="directory where the DUSTY executable runs")
    ap.add_argument("--template", default=TEMPLATE_INP, help="template .inp file")
    ap.add_argument("--exec", default=EXEC_NAME, help="DUSTY executable name")
    ap.add_argument("--format", choices=["lamflam", "fnu_jy"], default="lamflam",
                    help="'lamflam' = lambda + lambda*F_lambda columns; "
                         "'fnu_jy' = lambda, F_nu[Jy], and error columns "
                         "(the Jy-style catalog format)")
    ap.add_argument("--lam-col", type=int, default=0, help="wavelength column index")
    ap.add_argument("--flam-col", type=int, default=1,
                    help="lambdaFlambda column index (format=lamflam only)")
    ap.add_argument("--fnu-col", type=int, default=1,
                    help="F_nu column index (format=fnu_jy only)")
    ap.add_argument("--lam-err-col", type=int, default=2,
                    help="lambda error column index (format=fnu_jy only)")
    ap.add_argument("--fnu-err-col", type=int, default=3,
                    help="F_nu error column index (format=fnu_jy only)")
    ap.add_argument("--fnu-err-absolute", action="store_true",
                    help="set if the F_nu error column is an ABSOLUTE Jy value "
                         "instead of a fractional/relative error (default: "
                         "fractional, the common catalog convention)")
    ap.add_argument("--err-col", type=int, default=None,
                    help="error column index (format=lamflam only, optional)")
    ap.add_argument("--err-is-log", action="store_true",
                    help="set if the error column is already in dex (format=lamflam only)")
    ap.add_argument("--sigma", type=float, default=ASSUMED_SIGMA_LOG,
                    help="assumed error in dex if no error column (default %(default)s)")
    # --- prior box ---
    ap.add_argument("--tstar", nargs=2, type=float, default=None,
                    metavar=("LO", "HI"), help="Tstar prior range (K)")
    ap.add_argument("--tdust", nargs=2, type=float, default=None,
                    metavar=("LO", "HI"), help="Tdust prior range (K)")
    ap.add_argument("--tau", nargs=2, type=float, default=None,
                    metavar=("LO", "HI"), help="tau prior range")
    # --- sampler (defaults None so we can prompt only if not set on CLI) ---
    ap.add_argument("--nwalkers", type=int, default=None)
    ap.add_argument("--nburn", type=int, default=None)
    ap.add_argument("--nprod", type=int, default=None)
    ap.add_argument("--nproc", type=int, default=None,
                    help="parallel processes (default: all cores)")
    ap.add_argument("--no-prompt", action="store_true",
                    help="never prompt; use defaults / CLI values only "
                         "(use this on servers / under nohup)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    allow = not args.no_prompt

    # --- 1) ask for the prior ranges first (Tstar, Tdust, tau) ---
    if allow:
        print("\nSet the prior ranges (press Enter to accept the defaults shown).")
    tstar = _ask_range("Tstar (K)", BOUNDS["Tstar"][0], BOUNDS["Tstar"][1],
                       allow, provided=args.tstar)
    tdust = _ask_range("Tdust (K)", BOUNDS["Tdust"][0], BOUNDS["Tdust"][1],
                       allow, provided=args.tdust)
    tau   = _ask_range("tau",       BOUNDS["tau"][0],   BOUNDS["tau"][1],
                       allow, provided=args.tau)
    BOUNDS = {"Tstar": tstar, "Tdust": tdust, "tau": tau}

    # override module globals from CLI so the rest of the code is unchanged
    DATA_FILE, STAR_NAME = args.data, args.star
    DUSTY_DIR, TEMPLATE_INP, EXEC_NAME = args.dusty_dir, args.template, args.exec

    rng = np.random.RandomState(args.seed)
    tag = "%s_%s" % (STAR_NAME, args.comp)

    if args.format == "fnu_jy":
        is_csv = args.data.lower().endswith(".csv")
        data = load_photometry_fnu_jy(
            args.data, lam_col=args.lam_col, fnu_col=args.fnu_col,
            lam_err_col=args.lam_err_col, fnu_err_col=args.fnu_err_col,
            fnu_err_is_fractional=not args.fnu_err_absolute,
            delimiter="," if is_csv else None)
        print("[%s] format=fnu_jy: converted F_nu[Jy] -> lambda*F_lambda" % tag)
    else:
        data = load_photometry(args.data, sigma_log=args.sigma,
                               lam_col=args.lam_col, flam_col=args.flam_col,
                               err_col=args.err_col, err_is_log=args.err_is_log)
    kernel = DustyKernel(dusty_dir=DUSTY_DIR, template_inp=TEMPLATE_INP,
                         exec_name=EXEC_NAME)
    like = SEDLikelihood(data, kernel=kernel, bounds=BOUNDS,
                         composition=args.comp)
    print("\n[%s] %d photometry points from %s" % (tag, like.lam.size, args.data))
    print("[%s] priors: Tstar %g-%g K  Tdust %g-%g K  tau %g-%g"
          % (tag, tstar[0], tstar[1], tdust[0], tdust[1], tau[0], tau[1]))

    # always time one call first
    print("[%s] timing one DUSTY call (composition %s) ..." % (tag, args.comp))
    dt, ok = time_one_call(kernel, args.comp)
    if not ok:
        print("  ERROR: DUSTY run failed. Check DUSTY_DIR / EXEC_NAME / template.")
        sys.exit(1)
    print("  one DUSTY call = %.2f s" % dt)

    if args.mode == "quick":
        nwalkers, nburn, nprod = 12, 20, 40
        print("\n[quick] tiny chain to confirm it runs (%d walkers, %d+%d steps)"
              % (nwalkers, nburn, nprod))
        ncores_max = _cpucount()
        if allow:
            print("\nThis machine has %d CPU cores available." % ncores_max)
        nproc = _ask_int("Number of CPU cores to use",
                         default=args.nproc or ncores_max,
                         lo=1, hi=ncores_max, allow_prompt=allow,
                         provided=args.nproc)
        for (w, b, p) in [(24, 300, 1500), (32, 500, 3000), (48, 800, 5000)]:
            calls = w * (b + p)
            eff = max(1, min(nproc, w // 2))   # emcee cap: nwalkers/2
            est = calls * dt / eff
            print("  estimate: %2d walkers x (%d+%d) = %d calls "
                  "-> ~%.0f s = %.1f h using %d of %d cores"
                  % (w, b, p, calls, est, est / 3600.0, eff, nproc))
        sampler = run_chain(like, nwalkers, nburn, nprod, rng, nproc, dt)
        summarize(sampler, like, kernel, args.comp, tag + "_quick")
        print("\n[quick] done. Tell me the per-call time above; then run --mode full.")
    else:
        # --- 2) ask for walkers / burn-in / production ---
        if allow:
            print("\nSet the sampler (press Enter to accept the defaults shown).")
        nwalkers = _ask_int("Number of walkers",
                            default=24, lo=6, hi=1024,
                            allow_prompt=allow, provided=args.nwalkers)
        nburn    = _ask_int("Number of burn-in (warm-up) steps",
                            default=300, lo=1, hi=100000,
                            allow_prompt=allow, provided=args.nburn)
        nprod    = _ask_int("Number of production steps",
                            default=1500, lo=1, hi=1000000,
                            allow_prompt=allow, provided=args.nprod)

        # --- 3) show available cores and ask how many to use ---
        ncores_max = _cpucount()
        if allow:
            print("\nThis machine has %d CPU cores available." % ncores_max)
        nproc    = _ask_int("Number of CPU cores to use",
                            default=ncores_max, lo=1, hi=ncores_max,
                            allow_prompt=allow, provided=args.nproc)

        calls = nwalkers * (nburn + nprod)
        # emcee can only run nwalkers/2 calls at once, however many cores exist
        eff = max(1, min(nproc, nwalkers // 2))
        print("\n[full] %d walkers x (%d+%d) = %d DUSTY calls "
              "-> ~%.1f h using %d of %d cores  (rough upper bound)"
              % (nwalkers, nburn, nprod, calls,
                 calls * dt / eff / 3600.0, eff, nproc))
        if eff < nproc:
            print("       (%d walkers caps emcee at %d simultaneous calls; "
                  "use --nwalkers %d or more to fill all %d cores)"
                  % (nwalkers, eff, 2 * nproc, nproc))
        sampler = run_chain(like, nwalkers, nburn, nprod, rng, nproc, dt)
        summarize(sampler, like, kernel, args.comp, tag)
        print("\n[full] done.")


def _ask_int(label, default, lo, hi, allow_prompt, provided):
    """Return `provided` if given on the CLI; otherwise prompt (or fall back to
    `default` when prompting is disabled). Validates against [lo, hi]."""
    if provided is not None:
        return provided
    if not allow_prompt:
        return default
    while True:
        raw = input("%s [%d, range %d-%d]: " % (label, default, lo, hi)).strip()
        if raw == "":
            return default
        try:
            v = int(raw)
        except ValueError:
            print("  please enter an integer.")
            continue
        if v < lo or v > hi:
            print("  out of range (%d to %d). try again." % (lo, hi))
            continue
        return v


def _ask_range(label, default_lo, default_hi, allow_prompt, provided=None):
    """Ask the user for a (lo, hi) range. Accepts 'lo hi' or 'lo,hi' on one line.
    Pressing Enter accepts the default. Returns (lo, hi) with lo < hi."""
    if provided is not None:
        return tuple(provided)
    if not allow_prompt:
        return (default_lo, default_hi)
    while True:
        raw = input("%s range [default %g %g]: "
                    % (label, default_lo, default_hi)).strip()
        if raw == "":
            return (default_lo, default_hi)
        parts = raw.replace(",", " ").split()
        if len(parts) != 2:
            print("  please enter two numbers separated by space or comma.")
            continue
        try:
            lo, hi = float(parts[0]), float(parts[1])
        except ValueError:
            print("  could not parse as numbers, try again.")
            continue
        if not (lo < hi):
            print("  lo must be strictly less than hi.")
            continue
        return (lo, hi)


def _cpucount():
    import os
    return os.cpu_count() or 1


if __name__ == "__main__":
    main()
