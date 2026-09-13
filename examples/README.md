# Example data

`demo_data/example_source_1.csv` is a **synthetic** photometric SED, made up
purely to demonstrate the expected file format — it is not a real
measurement of any star. Each row is one photometric band:

```
wavelength[um], F_nu[Jy], fractional_error_on_wavelength, fractional_error_on_F_nu
```

This is the `fnu_jy` format read by `load_photometry_fnu_jy()`. If your own
catalog already gives you `lambda * F_lambda` directly instead of `F_nu`, use
the `lamflam` format instead (see the manual, Section 4).

## Trying it out

This demo file only exercises the data-loading and likelihood code — you
still need your own compiled DUSTY executable and a template `.inp` file to
actually run a fit (see the manual for where to obtain DUSTY and how the
template file works). Once you have those in a working directory:

```bash
cd /path/to/your/dusty/working/dir
python3 /path/to/MCMC_SEDust/mcmc_sedust/mcmc_dusty.py \
    --comp M --mode full \
    --data /path/to/MCMC_SEDust/examples/demo_data/example_source_1.csv \
    --star demo1 --format fnu_jy \
    --tstar 2000 4000 --tdust 100 1400 --tau 0.01 2 \
    --nwalkers 16 --nburn 100 --nprod 300
```

Replace the prior ranges with whatever is appropriate for your own target
population — the numbers above are just a reasonable starting point for a
dusty evolved star, not a recommendation for any specific galaxy.
