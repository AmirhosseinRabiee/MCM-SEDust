"""
sed_likelihood.py

Loads photometry and defines the likelihood that compares a DUSTY model SED
against it.

A few things worth knowing about how this is set up:

  Everything is compared in log10(lambda*F_lambda) space, since that's the
  axis the data gets plotted on and it makes the vertical "shift" a simple
  additive constant.

  The shift itself (which stands in for the source's distance and
  luminosity) isn't a sampled parameter. For a given model shape it has a
  closed-form best value -- the inverse-variance-weighted mean of
  (data - model) in log space -- so it's solved directly at every evaluation
  instead of being fit. Keeps the MCMC down to three parameters and avoids
  a nasty degeneracy with distance.

  The model gets interpolated onto each filter's wavelength in log-log
  space rather than convolved with the actual filter bandpass. Good enough
  for broadband photometry; convolving properly would be a refinement, not
  a requirement.

Works with either a live DustyKernel or a precomputed (lambda, fTot) pair,
which makes it easy to test without DUSTY installed.

------------------------------------------------------------------------------
Photometry file format (lamflam mode):  column 1 = wavelength [micron],
                                         column 2 = lambda*F_lambda
------------------------------------------------------------------------------
"""


import numpy as np

# ---------------------------------------------------------------------------
# Placeholder uncertainty for photometry that has NO error in the CSV.
#
# Some catalog rows (often the longest-wavelength bands) carry 'nan' in
# the flux-error column. Those are exactly the wavelengths that constrain the
# dust temperature, so they must not be thrown away. They are given this
# fractional flux error instead, which is deliberately LARGER than the ~8-11%
# of the measured filters: it lets them inform the fit without dominating it.
#
# THIS IS AN ASSUMPTION, NOT DATA. Every affected run says so on screen and in
# the per-star summary. Set this back to None-like handling, or simply update
# the CSVs, once the real uncertainties are available.
# ---------------------------------------------------------------------------
MISSING_FLUX_ERR = 0.15   # 15 per cent


# two loaders: one for plain lambda*F_lambda files, one for the
# Jy-based catalog CSVs (which also carries error columns)
def load_photometry(path, frac_err=0.10, sigma_log=None,
                    lam_col=0, flam_col=1, err_col=None, err_is_log=False):
    """Load a star's SED photometry from a text/CSV file.

    Returns a dict with:
        lam       : wavelength [micron]
        logflam   : log10(lambda*F_lambda)
        sigma_log : 1-sigma uncertainty in dex (log10 units)

    Flexible parsing so it works on different file layouts:
      lam_col   : index of the wavelength column   (default 0)
      flam_col  : index of the lambda*F_lambda col  (default 1)
      err_col   : index of an error column, or None (default None)
      err_is_log: True if that error column is already in dex; False if it is a
                  linear flux error on lambda*F_lambda (converted to dex).

    Delimiters: auto-detects commas (CSV) vs whitespace. Header lines beginning
    with '#' are ignored; a non-numeric first header row is skipped too.

    Errors, in priority order:
      1. an err_col in the file,
      2. an explicit sigma_log argument (scalar dex or array),
      3. otherwise a constant fractional flux error `frac_err` -> dex.
         (Replace with real errors when available; affects posterior WIDTHS.)
    """
    # sniff delimiter
    with open(path, "r") as fh:
        sample = ""
        for line in fh:
            if line.strip() and not line.lstrip().startswith("#"):
                sample = line
                break
    delim = "," if sample.count(",") >= 1 else None  # None -> any whitespace

    # try numeric load; if the first row is a header, skip it
    try:
        d = np.loadtxt(path, delimiter=delim, comments="#")
    except ValueError:
        d = np.loadtxt(path, delimiter=delim, comments="#", skiprows=1)
    d = np.atleast_2d(d)

    lam = d[:, lam_col].astype(float)
    flam = d[:, flam_col].astype(float)

    raw_err = d[:, err_col].astype(float) if err_col is not None else None

    order = np.argsort(lam)
    lam, flam = lam[order], flam[order]
    logflam = np.log10(flam)

    if raw_err is not None:
        raw_err = raw_err[order]
        if err_is_log:
            sig = raw_err
        else:
            # linear flux error -> dex:  sigma_log ~ err / (flam * ln10)
            sig = raw_err / (flam * np.log(10.0))
    elif sigma_log is not None:
        sig = np.atleast_1d(np.asarray(sigma_log, float))
        sig = np.full_like(lam, sig.item()) if sig.size == 1 else sig[order]
    else:
        sig = np.full_like(lam, frac_err / np.log(10.0))

    return {"lam": lam, "logflam": logflam, "sigma_log": sig}


def load_photometry_fnu_jy(path, lam_col=0, fnu_col=1, lam_err_col=2,
                          fnu_err_col=3, fnu_err_is_fractional=True,
                          delimiter=","):
    """Load photometry given as F_nu in Jansky (a common catalog format:
    col0=lambda[um], col1=F_nu[Jy], col2=lambda error, col3=F_nu error).

    Converts to the same internal representation as load_photometry():
        lam       : wavelength [micron]            (unchanged)
        logflam   : log10(lambda * F_lambda)        [W/m^2], derived from F_nu
        sigma_log : 1-sigma uncertainty in dex

    Conversion:  lambda*F_lambda = c * F_nu / lambda
                 (F_nu in Jy is first converted to SI: 1 Jy = 1e-26 W/m^2/Hz)

    fnu_err_col          : column index of the F_nu error.
    fnu_err_is_fractional: True  -> that column is a RELATIVE error (e.g. 0.089
                                     means 8.9%).
                           False -> that column is an ABSOLUTE error in Jy.
    lam_err_col           : column index of the wavelength error (kept for
                            plotting horizontal error bars; NOT used in the
                            likelihood, since DUSTY has no concept of an
                            x-uncertainty -- each filter is evaluated at one
                            wavelength).

    Returns the same dict as load_photometry(), PLUS 'lam_err' (the raw
    FRACTIONAL wavelength error from the file; absolute error in micron is
    lam * lam_err) for plotting, and 'fnu_jy' (the original flux density).
    """
    C_LIGHT = 2.998e8       # m/s
    JY_TO_SI = 1e-26        # W/m^2/Hz per Jy

    d = np.loadtxt(path, delimiter=delimiter, comments="#")
    d = np.atleast_2d(d)

    lam = d[:, lam_col].astype(float)              # micron
    fnu_jy = d[:, fnu_col].astype(float)            # Jy
    lam_err = d[:, lam_err_col].astype(float) if lam_err_col is not None else None
    fnu_err_raw = d[:, fnu_err_col].astype(float) if fnu_err_col is not None else None

    order = np.argsort(lam)
    lam = lam[order]
    fnu_jy = fnu_jy[order]
    if lam_err is not None:
        lam_err = lam_err[order]
    if fnu_err_raw is not None:
        fnu_err_raw = fnu_err_raw[order]

    lam_m = lam * 1e-6
    lamflam = C_LIGHT * (fnu_jy * JY_TO_SI) / lam_m    # W/m^2
    logflam = np.log10(lamflam)

    if fnu_err_raw is not None:
        if fnu_err_is_fractional:
            # a relative error transfers directly:
            # sigma(lamFlam)/lamFlam = sigma(Fnu)/Fnu
            frac = fnu_err_raw
        else:
            frac = fnu_err_raw / fnu_jy
        # convert a fractional LINEAR error to a dex (log10) error:
        # d(log10 x) = (1/ln10) * dx/x   <-- this is the CORRECT y error bar
        sigma_log = frac / np.log(10.0)
        flam_frac = frac    # raw fraction, kept only for the alternate plot style
    else:
        sigma_log = np.full_like(lam, 0.05)   # placeholder if no error column
        flam_frac = None

    # ---- handle points with a missing (nan) error -----------------------
    # Some catalog rows carry 'nan' in the flux-error column: the filter WAS
    # measured, but its uncertainty is missing from the CSV. Those points are
    # the mid-IR ones that constrain the dust temperature, so dropping them
    # (or excluding them from the chi2) badly degrades Tdust.
    #
    # PLACEHOLDER: they are given a stated fractional flux error of
    # MISSING_FLUX_ERR below and then treated like any other point. This is an
    # ASSUMPTION, not data -- it is deliberately larger than the ~8-11% typical
    # of the measured filters, so these points inform the fit without
    # dominating it. Every run prints which points were substituted, and
    # 'n_missing_err' is reported downstream so the assumption stays visible.
    #
    # >>> REPLACE THIS once the real uncertainties are added to the CSVs. <<<
    #
    # Rows with a non-finite wavelength or flux are genuinely unusable: dropped.
    usable = np.isfinite(lam) & np.isfinite(logflam)
    if not np.all(usable):
        nbad = int((~usable).sum())
        print("  NOTE: dropped %d row(s) with an invalid wavelength or flux"
              % nbad)
        lam, logflam, fnu_jy = lam[usable], logflam[usable], fnu_jy[usable]
        sigma_log = sigma_log[usable]
        if lam_err is not None:
            lam_err = lam_err[usable]
        if flam_frac is not None:
            flam_frac = flam_frac[usable]

    missing = ~(np.isfinite(sigma_log) & (sigma_log > 0))
    n_missing_err = int(missing.sum())
    if n_missing_err:
        sigma_log = np.where(missing, MISSING_FLUX_ERR / np.log(10.0),
                             sigma_log)
        if flam_frac is not None:
            flam_frac = np.where(missing, MISSING_FLUX_ERR, flam_frac)
        print("  NOTE: %d of %d points had NO flux error in the file "
              "(lambda = %s um)."
              % (n_missing_err, lam.size,
                 ", ".join("%.3f" % x for x in np.atleast_1d(lam)[missing])))
        print("        ASSUMED %.0f%% error for them so they can be fitted. "
              "This is a placeholder, not measured data."
              % (100.0 * MISSING_FLUX_ERR))

    if lam.size < 4:
        raise ValueError(
            "only %d photometry points in %s -- too few to constrain "
            "3 parameters" % (lam.size, path))

    return {"lam": lam, "logflam": logflam, "sigma_log": sigma_log,
           "lam_err": lam_err, "fnu_jy": fnu_jy, "flam_frac": flam_frac,
           "n_missing_err": n_missing_err}


def interp_model_to_data(lam_model, fTot_model, lam_data):
    """Interpolate the model log10(fTot) onto the data wavelengths, in log-log.

    Returns (logmodel_at_data, ok_mask). ok_mask is False for any data
    wavelength outside the model's positive-flux coverage (no extrapolation).
    """
    g = fTot_model > 0
    lm = np.log10(lam_model[g])
    ym = np.log10(fTot_model[g])
    xd = np.log10(lam_data)
    inside = (xd >= lm.min()) & (xd <= lm.max())
    out = np.full_like(xd, np.nan)
    out[inside] = np.interp(xd[inside], lm, ym)
    return out, inside


def analytic_shift(logflam_data, logmodel, sigma_log):
    """Closed-form best vertical shift (dex): inverse-variance weighted mean of
    (data - model)."""
    w = 1.0 / sigma_log**2
    return np.sum(w * (logflam_data - logmodel)) / np.sum(w)


# ties the data to the kernel -- emcee calls like.log_prob(theta) on this
class SEDLikelihood:
    def __init__(
        self,
        data,
        kernel=None,
        bounds=None,
        composition=None,
    ):
        """
        data        : dict from load_photometry(...)
        kernel      : a DustyKernel instance (real MCMC use). May be None if you
                      only call .evaluate_model(...) with a precomputed model.
        bounds      : dict of (lo, hi) priors for 'Tstar','Tdust','tau'.
        composition : None -> template as-is; 'C' carbon; 'M' oxygen-rich.
                      The kernel runs with this dust composition every call.
        """
        self.lam = data["lam"]
        self.logflam = data["logflam"]
        self.sigma_log = data["sigma_log"]
        # fractional wavelength error (column 3 of the fnu_jy CSVs), if the
        # loader provided it; used only for x error bars in plots, never in
        # the likelihood.
        self.lam_err = data.get("lam_err", None)
        # raw fractional flux error (CSV col 4), kept only so plots can
        # optionally reproduce a raw-fraction plotting style; never used in the fit.
        self.flam_frac = data.get("flam_frac", None)
        # how many points got the MISSING_FLUX_ERR placeholder instead of a
        # measured error. Kept so the fit reports can flag the assumption.
        self.n_missing_err = int(data.get("n_missing_err", 0) or 0)
        self.kernel = kernel
        self.composition = composition
        self.bounds = bounds or {
            "Tstar": (2000.0, 4000.0),   # cool evolved-star photospheres
            "Tdust": (100.0, 1500.0),    # inner-boundary dust temperature
            "tau":   (0.01, 50.0),       # optical depth at lambda0
        }
        self.param_names = ("Tstar", "Tdust", "tau")

    # ---- prior ------------------------------------------------------- #
    def log_prior(self, theta):
        for name, val in zip(self.param_names, theta):
            lo, hi = self.bounds[name]
            if not (lo <= val <= hi):
                return -np.inf
        return 0.0

    # ---- core comparison given an explicit model --------------------- #
    def evaluate_model(self, lam_model, fTot_model):
        """Compare a precomputed model to the data.

        Returns dict(shift, chi2, loglike, logmodel, mask) or None if the model
        does not cover the data wavelengths.
        """
        ym, mask = interp_model_to_data(lam_model, fTot_model, self.lam)
        if not mask.all():
            return None
        # Every point now has a sigma (measured, or the MISSING_FLUX_ERR
        # placeholder set by the loader), so all points enter the fit.
        shift = analytic_shift(self.logflam, ym, self.sigma_log)
        resid = self.logflam - (ym + shift)
        chi2 = np.sum((resid / self.sigma_log) ** 2)
        # Gaussian log-likelihood (constant term kept for completeness)
        loglike = -0.5 * chi2 - np.sum(
            np.log(self.sigma_log * np.sqrt(2 * np.pi)))
        return {
            "shift": shift,
            "chi2": chi2,
            "loglike": loglike,
            "logmodel": ym,
            "mask": mask,
        }

    # ---- likelihood from parameters (runs the kernel) ---------------- #
    def log_likelihood(self, theta):
        if self.kernel is None:
            raise RuntimeError("No kernel set; use evaluate_model() instead.")
        Tstar, Tdust, tau = theta
        lam_model, fTot_model = self.kernel.run(Tstar, Tdust, tau,
                                                composition=self.composition)
        if lam_model is None:
            return -np.inf
        res = self.evaluate_model(lam_model, fTot_model)
        if res is None:
            return -np.inf
        return res["loglike"]

    # ---- full posterior probability for emcee (step 3) --------------- #
    def log_prob(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll
