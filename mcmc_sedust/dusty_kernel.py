"""
dusty_kernel.py

Wraps the compiled DUSTY V2 executable so it can be called like an ordinary
Python function:

    lam, sed = kernel.run(Tstar, Tdust, tau)

Each call writes the three parameters into a fresh copy of your template
.inp, runs DUSTY in an isolated directory (so parallel calls never clobber
each other), and parses the resulting .stb spectrum. The vertical shift
that maps the model shape onto an observed flux level is NOT applied here --
that's solved analytically in sed_likelihood.py.

    from dusty_kernel import DustyKernel
    kernel = DustyKernel(dusty_dir="/path/to/dusty", template_inp="template.inp")
    lam, sed = kernel.run(Tstar=2900, Tdust=100, tau=0.565)

If DUSTY fails or doesn't converge, run() returns (None, None) so the
likelihood can just treat that point as -inf and move on.

You can sanity check the .stb parser alone, without running DUSTY, on any
existing .stb file:
    python dusty_kernel.py some_model.stb
"""



import os
import re
import shutil
import subprocess
import tempfile
import sys

import numpy as np


# wraps the compiled DUSTY executable. One instance is reused for
# thousands of calls; each process keeps its own private run directory
# so parallel workers never step on each other's files.
class DustyKernel:
    def __init__(
        self,
        dusty_dir: str,
        template_inp: str = "template.inp",
        exec_name: str = "dusty",
        model_name: str = "mcmc",
        timeout: float = 120.0,
        link_support_files: bool = True,
    ):
        """
        dusty_dir          : the directory where DUSTY runs successfully today.
                             Must contain the executable, lambda_grid.dat, the
                             data/ folder with optical constants, etc.
        template_inp       : an .inp file (inside dusty_dir) used as the template.
        exec_name          : name of the compiled DUSTY executable in dusty_dir.
        model_name         : base name used for the per-call input/output files.
        timeout            : seconds before a hung DUSTY run is killed.
        link_support_files : symlink support files into each temp dir (fast).
                             Set False to copy them instead (slower, more portable).
        """
        self.dusty_dir = os.path.abspath(dusty_dir)
        self.exec_path = os.path.join(self.dusty_dir, exec_name)
        self.model_name = model_name
        self.timeout = timeout
        self.link = link_support_files
        self._rundirs = {}   # pid -> persistent run directory
        self.last_results = None   # dict from the last run's .out (Mdot etc.)

        template_path = os.path.join(self.dusty_dir, template_inp)
        if not os.path.isfile(template_path):
            raise FileNotFoundError("template .inp not found: %s" % template_path)
        with open(template_path, "r") as f:
            self.template = f.read()

        if not os.path.isfile(self.exec_path):
            raise FileNotFoundError("DUSTY executable not found: %s" % self.exec_path)

    def __getstate__(self):
        """Called when Python pickles this object to send to a worker process.
        We deliberately drop _rundirs so every worker starts with an empty dict
        and builds its own fresh run directory rather than inheriting the main
        process's paths (which it can't safely share)."""
        state = self.__dict__.copy()
        state["_rundirs"] = {}
        return state

    def __setstate__(self, state):
        """Called when a worker unpickles the object. Restore everything as-is;
        _rundirs is already empty from __getstate__, so the worker will create
        its own run directory on its first DUSTY call."""
        self.__dict__.update(state)

    # ------------------------------------------------------------------ #
    #  Parameter substitution into the .inp template
    # ------------------------------------------------------------------ #
    # Abundance lines for the supported grain types, in the fixed DUSTY order:
    #   Sil-Ow  Sil-Oc  Sil-DL  grf-DL  amC-Hn  SiC-Pg
    # 'C' = carbon star  (amorphous carbon + silicon carbide)
    # 'M' = oxygen-rich/M-type star (Draine & Lee astronomical silicate)
    COMPOSITIONS = {
        "C": "0.00    0.00    0.00    0.00    0.85    0.15",
        "M": "0.00    0.00    1.00    0.00    0.00    0.00",
    }

    # fills in the template with this call's Tstar/Tdust/tau and composition
    def _make_input(self, Tstar: float, Tdust: float, tau: float,
                    composition=None) -> str:
        """Return a copy of the template with the 3 parameters substituted.

        Anchored by text patterns (not line numbers), so it survives small
        edits to the template. Raises if any pattern is not found exactly once,
        which protects you from silently fitting the wrong number.

        composition : None  -> leave the template's composition line unchanged
                      'C'   -> carbon star  (amC-Hn 0.85 + SiC-Pg 0.15)
                      'M'   -> oxygen-rich   (Sil-DL 1.00)
                      or a raw 6-number abundance string for full control.
        """
        text = self.template

        # 1) Stellar blackbody temperature:  "Temperature = 2900 K"
        text, n1 = re.subn(
            r"(Temperature\s*=\s*)[0-9.eE+-]+(\s*K)",
            rf"\g<1>{Tstar:g}\g<2>",
            text,
        )
        # 2) Inner-boundary dust temperature:  "- temperature = 100 K"
        text, n2 = re.subn(
            r"(-\s*temperature\s*=\s*)[0-9.eE+-]+(\s*K)",
            rf"\g<1>{Tdust:g}\g<2>",
            text,
        )
        # 3) Optical depth:  "tau(min) = 0.565; tau(max) = 0.2"
        #    With number of models = 1, DUSTY uses tau(min); we set both equal.
        text, n3 = re.subn(
            r"(tau\(min\)\s*=\s*)[0-9.eE+-]+(\s*;\s*tau\(max\)\s*=\s*)[0-9.eE+-]+",
            rf"\g<1>{tau:g}\g<2>{tau:g}",
            text,
        )

        if not (n1 == 1 and n2 == 1 and n3 == 1):
            raise ValueError(
                "Parameter substitution failed "
                f"(stellar T matches={n1}, dust T matches={n2}, tau matches={n3}). "
                "Check that the template .inp still has the expected lines."
            )

        # 4) Optional composition switch: the abundance line that begins "x ="
        #    Anchored to the start of a line so it does NOT match "index = 1".
        if composition is not None:
            abund = self.COMPOSITIONS.get(composition, composition)
            text, n4 = re.subn(
                r"(?m)^(\s*x\s*=\s*)[0-9.][0-9.\s]*?(\n)",
                lambda mm, a=abund: mm.group(1) + a + mm.group(2),
                text,
                count=1,
            )
            if n4 != 1:
                raise ValueError(
                    "Composition substitution failed: could not find the "
                    "'x = ...' abundance line in the template."
                )
        return text

    # ------------------------------------------------------------------ #
    #  Build an isolated run directory
    # ------------------------------------------------------------------ #
    # each process works in its own scratch directory -- see __getstate__
    def _populate_rundir(self, rundir):
        """Symlink/copy the support files into rundir ONCE. Called a single time
        per run directory, not per DUSTY call."""
        for name in os.listdir(self.dusty_dir):
            if name.endswith(".inp") or name == "dusty.inp":
                continue  # we write our own each call
            src = os.path.join(self.dusty_dir, name)
            dst = os.path.join(rundir, name)
            try:
                if self.link:
                    os.symlink(src, dst)
                else:
                    if os.path.isdir(src):
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
            except FileExistsError:
                pass

    def _get_rundir(self):
        """Return this process's persistent run directory, creating it the first
        time. Keyed by PID so parallel emcee workers never share a directory.

        The heavy step (linking the whole DUSTY folder) happens ONCE per process;
        subsequent calls reuse the directory and only rewrite the small .inp."""
        pid = os.getpid()
        rundir = self._rundirs.get(pid)
        if rundir is None or not os.path.isdir(rundir):
            rundir = tempfile.mkdtemp(prefix="dusty_%d_" % pid)
            self._populate_rundir(rundir)
            # the master file dusty.inp never changes -> write it once
            with open(os.path.join(rundir, "dusty.inp"), "w") as f:
                f.write(self.model_name + "\n")
            self._rundirs[pid] = rundir
        return rundir

    # ------------------------------------------------------------------ #
    #  Parse a .stb file -> (lambda, fTot)
    # ------------------------------------------------------------------ #
    @staticmethod
    # pull the spectrum, plus Mdot/Ve/Mmax, out of DUSTY's output files
    def parse_stb(stb_path: str):
        """Return (lambda_micron, fTot) for rows with positive flux.

        Column layout (DUSTY V2 .stb):
          0:lambda  1:fTot  2:xAtt  3:xDs  4:xDe  5:fInp  6:tauT  7:albedo
        fTot is the normalized lambda*F_lambda (bolometric integral = 1).
        """
        data = np.loadtxt(stb_path, comments="#")
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"unexpected .stb shape in {stb_path}")
        lam = data[:, 0]
        fTot = data[:, 1]
        good = fTot > 0
        return lam[good], fTot[good]

    # ------------------------------------------------------------------ #
    #  The public call
    # ------------------------------------------------------------------ #
    # the function the likelihood actually calls on every MCMC step
    def run(self, Tstar: float, Tdust: float, tau: float, composition=None):
        """Run DUSTY for one parameter set.

        composition : None -> use template as-is; 'C' carbon; 'M' oxygen-rich.

        Returns (lambda_micron, lambdaFlambda) on success,
        or (None, None) if the run failed / produced no usable spectrum.

        Uses a persistent per-process run directory: the DUSTY support files are
        linked in only once, and each call just rewrites the small <model>.inp,
        so the per-call overhead is a single tiny file write.
        """
        try:
            inp_text = self._make_input(Tstar, Tdust, tau, composition=composition)
        except ValueError:
            # Bad template -- re-raise, this is a setup bug, not a bad proposal.
            raise

        rundir = self._get_rundir()
        self.last_results = None   # cleared so a failed run leaves no stale Mdot
        try:
            # rewrite only the model input file each call
            with open(os.path.join(rundir, "%s.inp" % self.model_name), "w") as f:
                f.write(inp_text)

            # clear a stale .stb so we never read a previous call's output
            stb_path = os.path.join(rundir, "%s.stb" % self.model_name)
            if os.path.isfile(stb_path):
                os.remove(stb_path)

            proc = subprocess.run(
                [self.exec_path],
                cwd=rundir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.timeout,
            )
            if proc.returncode != 0:
                return None, None
            if not os.path.isfile(stb_path) or os.path.getsize(stb_path) == 0:
                return None, None

            lam, fTot = self.parse_stb(stb_path)
            if lam.size < 5:
                return None, None

            # parse the .out RESULTS row for derived quantities (Mdot etc.);
            # non-fatal if it fails -- the spectrum is still valid.
            out_path = os.path.join(rundir, "%s.out" % self.model_name)
            self.last_results = self.parse_out_results(out_path)

            return lam, fTot

        except (subprocess.TimeoutExpired, ValueError, OSError):
            return None, None

    @staticmethod
    def parse_out_results(out_path):
        """Parse the RESULTS row of a DUSTY V2 .out file.

        The .out contains a block like:
          RESULTS:
          --------
         ###   tau0   F1(W/m2)  r1(cm) r1/rc theta1 Td(Y) err  Mdot     Ve      M>
           1 4.00E-02 1.25E+05 1.57E+14 2.98E+00 1.17E+00 154  0  6.38E-07 3.18E+01 1.18E+01

        Legend: (8) Mass-loss rate (Msun/yr), (9) terminal velocity (km/s),
        (10) stellar mass upper limit (Msun).

        NOTE: DUSTY computes these for a fiducial luminosity L = 1e4 Lsun.
        The physical Mdot scales as sqrt(L / 1e4 Lsun), so a correction using
        the star's real luminosity (i.e. distance) is needed downstream.

        Returns dict with keys 'Mdot', 'Ve', 'Mmax' (floats), or None if the
        row could not be found/parsed.
        """
        try:
            with open(out_path, "r") as f:
                lines = f.readlines()
        except (IOError, OSError):
            return None
        in_results = False
        for line in lines:
            if "RESULTS" in line:
                in_results = True
                continue
            if not in_results:
                continue
            parts = line.split()
            # model row: '1' + 10 numeric columns
            if len(parts) >= 11 and parts[0] == "1":
                try:
                    vals = [float(p) for p in parts[1:11]]
                except ValueError:
                    continue
                return {"tau0": vals[0], "F1": vals[1], "r1": vals[2],
                        "Mdot": vals[7], "Ve": vals[8], "Mmax": vals[9]}
        return None

    def cleanup(self):
        """Remove this process's run directory. Call at program end if you like;
        otherwise the OS clears temp dirs on reboot."""
        for rundir in self._rundirs.values():
            shutil.rmtree(rundir, ignore_errors=True)
        self._rundirs = {}


# ---------------------------------------------------------------------- #
#  Standalone parser check (no DUSTY required):  python dusty_kernel.py X.stb
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python dusty_kernel.py <file.stb>")
        sys.exit(1)
    lam, fTot = DustyKernel.parse_stb(sys.argv[1])
    logy = np.log10(fTot)
    print(f"parsed {lam.size} positive-flux points")
    print(f"  lambda range : {lam.min():.4g} - {lam.max():.4g} micron")
    print(f"  log10(fTot)  : peak {logy.max():.4f} at lambda={lam[np.argmax(logy)]:.4g} um")
