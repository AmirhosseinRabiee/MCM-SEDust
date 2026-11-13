# MCM_SEDust

**MCM_SEDust** fits dust radiative-transfer models to the observed spectral
energy distributions (SEDs) of dusty evolved stars, using the Fortran
**DUSTY V2** code as a live forward model inside an MCM sampler
([`emcee`](https://emcee.readthedocs.io)) rather than a pre-computed grid.
It was built for a photometric catalog of one galaxy, but nothing in the
fitting method is specific to that galaxy — point it at a different
photometry catalog and different priors, and it works the same way.

Full documentation, including a complete step-by-step guide and an
explanation of every module, lives in **[`docs/MCM_SEDust_manual.pdf`](docs/MCM_SEDust_manual.pdf)**.
This README is just a quick orientation.

## What it does

For each source, three parameters are sampled: the stellar temperature
(`Tstar`), the dust temperature at the inner edge of the envelope
(`Tdust`), and the envelope optical depth (`tau`). A fourth quantity, the
overall flux normalisation, is solved for algebraically at every step rather
than sampled, which removes a dimension from the problem at no extra cost.
Missing flux uncertainties in the input catalog are handled explicitly
rather than silently dropped or guessed at. Large catalogs are fit by
parallelising over *sources* (one complete fit per CPU core) rather than
over sampler walkers, which is what actually lets the code scale to many
cores — see the manual for why.

## Repository layout

```
MCM_SEDust/
├── MCM_sedust/          the five modules -- keep them together
│   ├── dusty_kernel.py       wraps the DUSTY executable
│   ├── sed_likelihood.py     photometry loading + likelihood
│   ├── MCM_dusty.py         fit ONE source (also the shared library)
│   ├── fit_catalog.py        fit MANY sources, in parallel, resumably
│   └── plot_both.py          compare two dust compositions for one source
├── DUSTY_TEST/           DUSTY's own source + support data + a compile check
│   └── README.md             what's here and how to compile it
├── examples/
│   ├── demo_data/            a small synthetic photometry file
│   └── README.md
├── docs/
│   └── MCM_SEDust_Manual.pdf   <-- start here
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

`DUSTY_TEST/` has everything needed to compile DUSTY (see the manual, Section
2.2) but deliberately **not** a compiled binary — dynamically linked Fortran
binaries don't reliably run across different machines, so compile it fresh
wherever you're going to run fits.

**Still yours to supply**: a DUSTY input template for your own dust model,
and your photometry catalog. See the manual for what these need to look
like.

## Installation

```bash
git clone https://github.com/AmirhosseinRabiee/MCM-SEDust.git
cd MCM_SEDust
pip install -r requirements.txt
```

You also need a working DUSTY V2 installation (Ivezić & Elitzur 1997;
Elitzur & Ivezić 2001) — `DUSTY_TEST/` has the source and everything needed
to compile it; MCM_SEDust itself runs the resulting executable as a
subprocess.

## Quick start

```bash
cd /path/to/your/dusty/working/directory   # has the dusty executable + template
python3 /path/to/MCM_SEDust/MCM_sedust/MCM_dusty.py \
    --comp M --mode full \
    --data /path/to/your_star.csv --star star1 --format fnu_jy \
    --tstar 2000 4000 --tdust 100 1400 --tau 0.01 2 \
    --nwalkers 16 --nburn 100 --nprod 300
```

For a whole catalog:

```bash
python3 /path/to/MCM_SEDust/MCM_sedust/fit_catalog.py \
    --comp both --data-dir /path/to/your/catalog --first 1 --last N \
    --out-dir results --nwalkers 16 --nburn 100 --nprod 300 \
    --parallel-stars 64
```

The manual walks through every step, every flag, every output file, and
every module of the code, in the order you'd actually use them.

## Citing

See `CITATION.cff`. Please also cite DUSTY and emcee, which this code
depends on and wraps.

## License

MIT for the code in this repository (see `LICENSE`). DUSTY itself is
separate third-party software with its own license and citation
requirements.
