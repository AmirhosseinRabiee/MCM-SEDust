"""
fit_catalog.py  --  run the DUSTY MCMC fitter across a whole catalog of stars.

Loops over numbered CSV files (1.csv, 2.csv, ... N.csv) in a folder, each in
the fnu_jy format (lambda[um], F_nu[Jy], lambda error, F_nu fractional error),
and runs the MCMC for a chosen composition on every one. No interactive
prompts -- all sampler settings are fixed CLI numbers, since you can't sit
and answer prompts hundreds of times for a large catalog.

RESUMABLE: before fitting a star, it checks whether that star's chain file
already exists; if so it skips it (so a crashed or interrupted run can just be
re-launched and it continues where it left off). Use --redo to force re-fitting
everything anyway.

Examples:
  # Fit Model M for stars numbered 1 to N, files in <data-dir>/<n>.csv
  python3 fit_catalog.py --comp M --data-dir <data-dir> --first 1 --last N \
      --nwalkers 24 --nburn 300 --nprod 1500 --nproc 12

  # Fit both compositions in one pooled run across all available cores
  python3 fit_catalog.py --comp both --data-dir <data-dir> --first 1 --last N \
      --nwalkers 24 --nburn 300 --nprod 1500 --parallel-stars 64

  # Re-run a few specific stars only (e.g. they failed last time)
  python3 fit_catalog.py --comp M --data-dir <data-dir> --only 5,37,140 --redo

Outputs, per star (same as running mcmc_dusty.py directly):
  <star>_<comp>_chain.npy, _corner.png, _bestfit.png
plus ONE running summary CSV appended to as each star finishes:
  catalog_summary.csv
------------------------------------------------------------------------------
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")

from dusty_kernel import DustyKernel
from sed_likelihood import load_photometry_fnu_jy, SEDLikelihood
import mcmc_dusty as core   # reuse run_chain(), summarize(), _cpucount()

# one row per (star, composition), appended as each fit finishes.
# n_assumed_err tracks how many points used the placeholder error from
# sed_likelihood.MISSING_FLUX_ERR so that assumption stays visible.
SUMMARY_COLUMNS = [
    "star", "model", "npoints",
    "Tstar_med", "Tstar_lo", "Tstar_hi",
    "Tdust_med", "Tdust_lo", "Tdust_hi",
    "tau_med", "tau_lo", "tau_hi",
    "shift", "chi2", "Mdot", "acceptance", "n_assumed_err", "status", "seconds",
]


def parse_only(s):
    if not s:
        return None
    return set(int(x) for x in s.split(",") if x.strip())


def already_done(tag):
    return os.path.isfile("%s_chain.npy" % tag)


def append_summary_row(csv_path, row):
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(SUMMARY_COLUMNS)
        w.writerow(row)


def fit_one_star(star_id, data_path, args, bounds, star_seed=None, quiet=False,
                 nproc_override=None, comp=None):
    """Run the MCMC for one star + composition. Returns a summary row (list).
    All output files (chain, corner, bestfit) are written under args.out_dir.

    star_seed      : per-star random seed (so parallel stars are decorrelated).
    quiet          : suppress the chain's live progress printing.
    nproc_override : force the walker-level core count (1 in star-parallel mode,
                     because a star worker cannot open a nested pool).
    comp           : composition ('C' or 'M') for THIS fit; overrides comp
                     so a single run can cover both compositions.
    """
    comp = comp if comp is None else comp
    tag = os.path.join(args.out_dir, "%s_%s" % (star_id, comp))
    t0 = time.time()
    seed = args.seed if star_seed is None else star_seed
    nproc = args.nproc if nproc_override is None else nproc_override
    # emcee draws its internal proposal randomness from numpy's GLOBAL state
    # (np.random.get_state()), so seed that too -- otherwise chains are not
    # reproducible run-to-run even with the same per-star seed.
    np.random.seed(seed)

    try:
        data = load_photometry_fnu_jy(
            data_path, lam_col=args.lam_col, fnu_col=args.fnu_col,
            lam_err_col=args.lam_err_col, fnu_err_col=args.fnu_err_col,
            fnu_err_is_fractional=not args.fnu_err_absolute,
            delimiter="," if data_path.lower().endswith(".csv") else None)
    except Exception as e:
        print("  [%s] ERROR loading data: %s" % (tag, e))
        return [star_id, comp, 0] + [np.nan] * 9 + [np.nan, np.nan, np.nan, np.nan, 0,
                                                          "load_failed", 0.0]

    kernel = DustyKernel(dusty_dir=args.dusty_dir, template_inp=args.template,
                         exec_name=args.exec)
    like = SEDLikelihood(data, kernel=kernel, bounds=bounds,
                         composition=comp)

    # quick check that DUSTY runs at all for this star before committing to a chain
    dt, ok = core.time_one_call(kernel, comp)
    if not ok:
        print("  [%s] ERROR: DUSTY failed at test point; skipping." % tag)
        return [star_id, comp, like.lam.size] + [np.nan] * 9 + \
               [np.nan, np.nan, np.nan, np.nan, 0, "dusty_failed", time.time() - t0]

    rng = np.random.RandomState(seed)
    try:
        sampler = core.run_chain(like, args.nwalkers, args.nburn, args.nprod,
                                 rng, nproc, dt, quiet=quiet)
    except Exception as e:
        print("  [%s] ERROR during sampling: %s" % (tag, e))
        return [star_id, comp, like.lam.size] + [np.nan] * 9 + \
               [np.nan, np.nan, np.nan, np.nan, 0, "mcmc_failed", time.time() - t0]

    chain = sampler.get_chain(flat=True)
    np.save("%s_chain.npy" % tag, chain)
    labels = ["Tstar", "Tdust", "tau"]
    med = {}
    pct = {}
    for i, l in enumerate(labels):
        lo, m, hi = np.percentile(chain[:, i], [16, 50, 84])
        med[l] = m
        pct[l] = (lo, hi)

    lam_m, ftot = kernel.run(med["Tstar"], med["Tdust"], med["tau"],
                             composition=comp)
    if lam_m is not None:
        res = like.evaluate_model(lam_m, ftot)
        shift, chi2 = (res["shift"], res["chi2"]) if res else (np.nan, np.nan)
    else:
        shift, chi2 = np.nan, np.nan

    # Mdot from the .out of the median-parameter run just above
    # (DUSTY fiducial L=1e4 Lsun; physical Mdot = this x sqrt(L/1e4 Lsun))
    outres = getattr(kernel, "last_results", None)
    mdot = outres["Mdot"] if outres is not None else np.nan

    accept = float(np.mean(sampler.acceptance_fraction))
    elapsed = time.time() - t0

    # save corner + bestfit plots, same as single-star mode
    try:
        core.summarize(sampler, like, kernel, comp, tag)
    except Exception as e:
        print("  [%s] warning: plot/summary failed: %s" % (tag, e))

    print("  [%s] done in %.0f s : Tstar=%.0f Tdust=%.0f tau=%.3f shift=%.3f "
          "chi2=%.2f Mdot=%.2e"
          % (tag, elapsed, med["Tstar"], med["Tdust"], med["tau"], shift, chi2,
             mdot))

    return [star_id, comp, like.lam.size,
           med["Tstar"], pct["Tstar"][0], pct["Tstar"][1],
           med["Tdust"], pct["Tdust"][0], pct["Tdust"][1],
           med["tau"], pct["tau"][0], pct["tau"][1],
           shift, chi2, mdot, accept,
           getattr(like, "n_missing_err", 0), "ok", elapsed]


# has to be a top-level function so 'spawn' can import it in each
# worker. Runs one fit, start to finish, serially on one core.
def _star_worker(task):
    """Top-level worker for the star-level pool (must be importable by 'spawn').

    task = (key, star_id, data_path, args, bounds, star_seed, comp)
    Runs ONE (star, composition) chain SERIALLY (nproc_override=1, since a pool
    worker cannot open a nested pool) and quietly (no heartbeat spam).
    Returns (key, summary_row). 'key' uniquely identifies the task for
    resumability/ordering (it encodes both star number and composition).
    """
    key, star_id, data_path, args, bounds, star_seed, comp = task
    row = fit_one_star(star_id, data_path, args, bounds,
                       star_seed=star_seed, quiet=True, nproc_override=1,
                       comp=comp)
    return key, row


def main():
    ap = argparse.ArgumentParser(description="Batch-fit a catalog of stars with DUSTY+MCMC")
    ap.add_argument("--comp", choices=["C", "M", "both"], required=True,
                    help="composition to fit: C, M, or 'both' (fits each star "
                         "for both C and M in one run, filling all cores)")
    ap.add_argument("--data-dir", required=True,
                    help="folder containing <n>.csv files")
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=221)
    ap.add_argument("--only", default=None,
                    help="comma-separated star numbers to (re)run, "
                         "overrides --first/--last")
    ap.add_argument("--ext", default="csv", help="file extension (default csv)")
    ap.add_argument("--prefix", default="", help="star name prefix, e.g. gal_")
    ap.add_argument("--redo", action="store_true",
                    help="re-fit stars even if a chain file already exists")
    ap.add_argument("--no-prompt", action="store_true",
                    help="never ask questions; use CLI values or defaults "
                         "(needed for unattended/nohup server runs)")
    ap.add_argument("--out-dir", default="results",
                    help="subfolder (inside the current directory) where all "
                         "chains, plots, and the summary CSV are written "
                         "(default: %(default)s)")
    # DUSTY / data layout
    ap.add_argument("--dusty-dir", default=".")
    ap.add_argument("--template", default="template.inp")
    ap.add_argument("--exec", default="dusty")
    ap.add_argument("--lam-col", type=int, default=0)
    ap.add_argument("--fnu-col", type=int, default=1)
    ap.add_argument("--lam-err-col", type=int, default=2)
    ap.add_argument("--fnu-err-col", type=int, default=3)
    ap.add_argument("--fnu-err-absolute", action="store_true")
    # priors -- default None so we know whether the user passed them on the CLI
    ap.add_argument("--tstar", nargs=2, type=float, default=None,
                    metavar=("LO", "HI"))
    ap.add_argument("--tdust", nargs=2, type=float, default=None,
                    metavar=("LO", "HI"))
    ap.add_argument("--tau", nargs=2, type=float, default=None,
                    metavar=("LO", "HI"))
    # sampler -- default None so we know whether the user passed them on the CLI
    ap.add_argument("--nwalkers", type=int, default=None)
    ap.add_argument("--nburn", type=int, default=None)
    ap.add_argument("--nprod", type=int, default=None)
    ap.add_argument("--nproc", type=int, default=None)
    ap.add_argument("--parallel-stars", type=int, default=None,
                    help="run this many fits at once, one per core "
                         "(recommended for catalogs -- try your core count). "
                         "1 = old walker-parallel behaviour.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--summary", default="catalog_summary.csv")
    args = ap.parse_args()

    # --- make and move into the output folder; everything below is written there ---
    os.makedirs(args.out_dir, exist_ok=True)
    args.summary = os.path.join(args.out_dir, args.summary)
    print("All chains, plots, and the summary CSV will be written to: %s/"
          % os.path.abspath(args.out_dir))

    # --- ask for anything not given on the CLI (once, before the loop) ---
    allow = not args.no_prompt
    default_bounds = {"Tstar": (2000.0, 4000.0), "Tdust": (100.0, 1400.0),
                      "tau": (0.01, 2.0)}
    if allow:
        print("\nSet the prior ranges (press Enter to accept the defaults shown).")
        print("(These apply to EVERY star in this batch.)")
    tstar = core._ask_range("Tstar (K)", *default_bounds["Tstar"], allow, args.tstar)
    tdust = core._ask_range("Tdust (K)", *default_bounds["Tdust"], allow, args.tdust)
    tau   = core._ask_range("tau",       *default_bounds["tau"],   allow, args.tau)
    bounds = {"Tstar": tstar, "Tdust": tdust, "tau": tau}

    if allow:
        print("\nSet the sampler (press Enter to accept the defaults shown). "
              "(Applies to every star.)")
    args.nwalkers = core._ask_int("Number of walkers", default=24, lo=6, hi=1024,
                                  allow_prompt=allow, provided=args.nwalkers)
    args.nburn    = core._ask_int("Number of burn-in (warm-up) steps", default=300,
                                  lo=1, hi=100000, allow_prompt=allow,
                                  provided=args.nburn)
    args.nprod    = core._ask_int("Number of production steps", default=1500,
                                  lo=1, hi=1000000, allow_prompt=allow,
                                  provided=args.nprod)
    ncores_max = core._cpucount()
    if allow:
        print("\nThis machine has %d CPU cores available." % ncores_max)
        print("Parallel stars = how many STARS run at once (one per core).")
        print("This is the fastest mode for a catalog. Enter 1 for the old")
        print("behaviour (one star at a time with parallel walkers).")
    args.parallel_stars = core._ask_int(
        "Number of stars to run in parallel", default=ncores_max,
        lo=1, hi=ncores_max, allow_prompt=allow, provided=args.parallel_stars)
    if args.parallel_stars <= 1:
        # sequential star loop -> walker-level parallelism, so ask about cores
        args.nproc = core._ask_int("Number of CPU cores to use (per star)",
                                   default=ncores_max, lo=1, hi=ncores_max,
                                   allow_prompt=allow, provided=args.nproc)
    else:
        args.nproc = 1   # each star runs serially inside its own worker

    only = parse_only(args.only)
    if only:
        ids = sorted(only)
    else:
        ids = list(range(args.first, args.last + 1))

    comps = ["C", "M"] if args.comp == "both" else [args.comp]

    print("Catalog fit: composition=%s  stars=%d  fits=%d  (%s)"
          % (args.comp, len(ids), len(ids) * len(comps),
             ids[:5] if len(ids) > 5 else ids))
    if args.parallel_stars > 1:
        print("Mode: STAR-PARALLEL -- %d fits at a time, each fit's chain "
              "runs serially on its own core" % args.parallel_stars)
        print("Sampler per fit: %d walkers x (%d burn + %d prod)"
              % (args.nwalkers, args.nburn, args.nprod))
    else:
        print("Sampler: %d walkers x (%d burn + %d prod) on %d cores"
              % (args.nwalkers, args.nburn, args.nprod, args.nproc))

    n_ok, n_skip, n_fail = 0, 0, 0
    t_start = time.time()

    # ---- first pass: resolve skips / missing files, build the task list ----
    # One task per (star, composition). 'key' is a label used only for progress
    # printing; it encodes both so the two compositions never collide.
    tasks = []
    for comp in comps:
        for n in ids:
            star_id = "%s%d" % (args.prefix, n)
            key = "%d_%s" % (n, comp)
            tag = os.path.join(args.out_dir, "%s_%s" % (star_id, comp))
            data_path = os.path.join(args.data_dir, "%d.%s" % (n, args.ext))

            if not args.redo and already_done(tag):
                print("%s : already done, skipping (use --redo to re-fit)" % tag)
                n_skip += 1
                continue
            if not os.path.isfile(data_path):
                print("%s : ERROR file not found: %s" % (tag, data_path))
                append_summary_row(args.summary,
                                   [star_id, comp, 0] + [np.nan] * 9 +
                                   [np.nan, np.nan, np.nan, np.nan, 0, "file_missing", 0.0])
                n_fail += 1
                continue
            # per-(star,comp) seed: decorrelates walker init across the catalog
            # AND between a star's C and M chains (offset M by 100000).
            seed = args.seed + n + (100000 if comp == "M" else 0)
            tasks.append((key, star_id, data_path, args, bounds, seed, comp))

    # ---- second pass: run the remaining tasks ----
    if args.parallel_stars > 1 and len(tasks) > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        nsimul = min(args.parallel_stars, len(tasks))

        # ---- upfront estimate for the WHOLE catalog -------------------------
        # Time one DUSTY call here in the main process so we can predict the
        # total. Each fit does a known, fixed number of calls, and nsimul fits
        # run at once, each on its own core.
        calls_per_fit = args.nwalkers * (args.nburn + args.nprod)
        total_calls = calls_per_fit * len(tasks)
        print("\ntiming one DUSTY call to estimate the total run time ...")
        probe = DustyKernel(dusty_dir=args.dusty_dir,
                            template_inp=args.template, exec_name=args.exec)
        dt, ok = core.time_one_call(probe, comps[0])
        if ok and dt:
            secs_per_fit = calls_per_fit * dt
            waves = int(np.ceil(len(tasks) / float(nsimul)))
            total_secs = secs_per_fit * len(tasks) / float(nsimul)
            eta_clock = time.strftime(
                "%a %H:%M", time.localtime(time.time() + total_secs))
            print("  one DUSTY call        = %.1f s" % dt)
            print("  calls per fit         = %d  (%d walkers x (%d burn + %d prod))"
                  % (calls_per_fit, args.nwalkers, args.nburn, args.nprod))
            print("  total DUSTY calls     = %d  (%d fits)"
                  % (total_calls, len(tasks)))
            print("  one fit takes         ~ %s  (on its own core)"
                  % core._fmt_dur(secs_per_fit))
            print("  running %d at a time  -> %d waves" % (nsimul, waves))
            print("  ESTIMATED TOTAL       ~ %s   (finishes around %s)"
                  % (core._fmt_dur(total_secs), eta_clock))
            print("  (rough: DUSTY is slower at high tau, so treat as a guide)")
        else:
            dt = None
            print("  WARNING: the timing DUSTY call failed -- no estimate "
                  "available. Check your template .inp / dusty executable.")

        print("\nlaunching %d fits, %d at a time ..." % (len(tasks), nsimul))
        print("(workers are quiet, so the heartbeat below is how you know it is "
              "alive; the first fit will not finish for a while)")

        # ---- heartbeat: proof of life + live DUSTY activity -----------------
        import threading
        import subprocess as _sp
        _stop = threading.Event()
        _t0 = time.time()
        _done = [0]

        def _count_dusty():
            """How many DUSTY processes are running right now (proof it works)."""
            try:
                out = _sp.check_output(["pgrep", "-c", "-f", args.exec],
                                       stderr=_sp.DEVNULL)
                return int(out.strip())
            except Exception:
                return -1   # pgrep unavailable; just don't report it

        def _heartbeat():
            while not _stop.wait(30.0):
                el = time.time() - _t0
                ndusty = _count_dusty()
                done = _done[0]
                msg = "    [alive] %s elapsed | %d/%d fits done" \
                      % (core._fmt_dur(el), done, len(tasks))
                if ndusty >= 0:
                    msg += " | %d DUSTY running now" % ndusty
                # once at least one fit has finished, we can extrapolate
                if done > 0:
                    rate = done / max(1e-6, el)
                    msg += " | est. remaining %s" \
                        % core._fmt_dur((len(tasks) - done) / rate)
                elif dt:
                    # nothing finished yet: fall back to the upfront estimate
                    msg += " | est. total %s" % core._fmt_dur(
                        calls_per_fit * dt * len(tasks) / float(nsimul))
                print(msg, flush=True)

        hb = threading.Thread(target=_heartbeat)
        hb.daemon = True
        hb.start()

        done_count = 0
        with ctx.Pool(processes=nsimul) as pool:
            # imap_unordered: rows arrive as fits finish (fast ones first);
            # ONLY this main process writes the summary CSV, so no file clashes.
            for key, row in pool.imap_unordered(_star_worker, tasks):
                done_count += 1
                _done[0] = done_count      # so the heartbeat can report progress
                append_summary_row(args.summary, row)
                if row[-2] == "ok":
                    n_ok += 1
                else:
                    n_fail += 1
                el = time.time() - t_start
                rate = done_count / max(1e-3, el)
                remain = (len(tasks) - done_count) / max(1e-6, rate)
                print("[%d/%d done] %s finished (%s)   elapsed %s   "
                      "est. remaining %s"
                      % (done_count, len(tasks), key, row[-2],
                         core._fmt_dur(el), core._fmt_dur(remain)), flush=True)
        _stop.set()   # shut the heartbeat down
    else:
        for k, task in enumerate(tasks, 1):
            key, star_id, data_path, _, _, star_seed, comp = task
            print("\n[%d/%d] fitting %s_%s  (%s) ..."
                  % (k, len(tasks), star_id, comp, data_path))
            row = fit_one_star(star_id, data_path, args, bounds,
                               star_seed=star_seed, comp=comp)
            append_summary_row(args.summary, row)
            if row[-2] == "ok":
                n_ok += 1
            else:
                n_fail += 1

    elapsed = time.time() - t_start
    print("\n=== catalog fit done in %s ===" % core._fmt_dur(elapsed))
    print("  ok: %d   skipped (already done): %d   failed: %d"
          % (n_ok, n_skip, n_fail))
    print("  summary written to %s" % args.summary)


if __name__ == "__main__":
    main()
