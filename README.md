# Reference-cohort model-X knockoffs in neuroimaging — analysis archive

Code, results and manuscript for:

> **Feature geometry, not the reference cohort, is the binding constraint on
> model-X knockoffs in neuroimaging**

## Verify every number in the manuscript

```bash
python scripts/verify_claims.py
```

`scripts/claims.csv` registers 175 numeric claims, each mapped to a specific
column and row-selector of a specific result file. The script checks them all
and exits non-zero on any mismatch. It has a second layer that scans the
manuscript for numeric literals not covered by any claim — this is what
catches a number written into the text that no computation produced. The two
layers point in opposite directions and neither is sufficient alone: the
first cannot tell whether a claim is still used by the manuscript, and the
second says nothing about whether the claimed values are right.

`scripts/audit_claims.py` closes the first of those gaps: it checks the
opposite direction, that each registered value is actually used by the
manuscript. Most claims appear verbatim or rounded; a group does not, and the
script lists them for triage, because "absent" has three different meanings —
the manuscript may express the quantity in words, the value may be a table
cell or a plotted point, or the analysis may have dropped out of the paper.
The last category is the one worth acting on.

## Regenerate the figures without rerunning experiments

```bash
pip install -r requirements.txt
python scripts/make_figures.py            # all ten figures
python scripts/make_figures.py --list     # list them
python scripts/make_figures.py --only fig_draws,fig_shift
```

Run it **from the repository root** (the script asserts that `paper/` and
`results_c0.05/` are present, and writes to `paper/figs/` by relative path).
Each figure is written twice — `<stem>.png` at 300 dpi for submission systems
and `<stem>.pdf` as vector art with editable text — alongside
`paper/figs/source_data/<stem>.csv`, which holds the exact numbers the figure
draws. `results*/` holds the numbers each figure is drawn from, so this runs
in seconds and needs no imaging data.

Two scripts regenerate cohort descriptions and controls rather than figures,
and both read external inputs documented in their docstrings:

```bash
python scripts/exp_adhd200_description.py        # -> results_v3/
python scripts/exp_multienv_positive_control.py --reps 60
```

The second is the positive control for the multi-environment arm: it injects
a signal that is consistent across environments by construction and reports
how much of it the filter recovers, so the arm's empty selection on the real
response can be read for what it is.

The PNGs regenerate bit-for-bit. The PDFs differ from the committed copies in
one field only—matplotlib embeds a `/CreationDate`—so `git status` lists them
as modified after a re-run even though the drawings are identical, at
identical file sizes. `git checkout -- paper/figs` restores them.

## Two things to know before reusing the code

**1. The penalty constant `c` is not free.** The fixed-λ lasso statistic has
one tuning parameter and it must satisfy two constraints at once: small
enough that a real signal survives it, and large enough that coordinate
descent converges on the real (weak) response. On this data only `c=0.05`
satisfies both. `results_c0.05/penalty_sweep.csv` and
`penalty_convergence_real.csv` measure the two sides. Every experiment
reports `frac_converged`; rows below 0.90 should be discarded, not
interpreted.

**2. Selections are distributions.** Knockoff generation is random, so
`P(S_hat != empty)` and the distribution of the threshold ratio are the
reportable quantities (`results_c0.05/draw_distribution.csv`), not any
single run.

**3. Where the error numbers come from.** On real data there is no ground
truth, so no false discovery proportion is computable. Every error figure in
the manuscript comes from an injected signal (truth constructed) or a global
null (everything null by construction). Both are in
`results_c0.05/positive_control_v3.csv` and
`results_matrixpow_step02/matrixpow_globalnull.csv`.

## Layout

| path | contents |
|---|---|
| `src/refcohort.py` | shared utilities: covariance, spectral diagnostics, the knockoff pipeline, provenance, lasso convergence tracking |
| `src/features.py` | the four ROI summary definitions |
| `src/build_features_v2.py` | one-pass feature construction for all four definitions |
| `src/exp_*.py` | the experiments behind each results section |
| `src/knockoffs.py` | knockoff constructions (equicorrelated / SDP / ASDP / MRC) |
| `results*/` | result tables; each has a sibling `.json` with arguments, library versions and input hashes |
| `scripts/verify_claims.py`, `scripts/claims.csv` | the claim-to-artifact map |
| `paper/manuscript_v3.tex`, `paper/figs/` | the manuscript and its figures |

## Data

Raw imaging data are public and are **not** redistributed here; see
`data/README.md` for sources and access requirements.

Three cohorts, in two data families:

| cohort | disorder | parcellation | n | build |
|---|---|---|---|---|
| ABIDE I | autism | AAL116 / CC200 / HO, 2 pipelines | 852 | `src/build_features_v2.py` |
| ABIDE II | autism | Power2011 (CIFTI) | 956 | `src/build_features_v2.py` |
| ADHD-200 | ADHD | AAL116 | 664 | `src/build_features_adhd200_v2.py` |

ADHD-200 is the external-validation cohort (`src/exp_third_cohort.py`). It is a
different disorder with different sites and protocols, and it is what tests
whether the feature-geometry ordering generalises: it does, while the severity
does not. Its time series are AFNI `3dROIstats` output — a `File`/`Sub-brick`
header then values, ROI codes in the same `#2001` scheme — so its reader is
separate from the ABIDE one.

**Matching reference-cohort size matters when comparing effective ranks.**
A sample correlation matrix at `n_ref/p` close to 1 is nearly singular by
construction, which inflates the apparent degeneracy. ADHD-200's reference
cohort is 121 against p=116, so its raw effective rank is not comparable to
ABIDE I's without matching; `src/exp_third_cohort.py` does the matching.
The same caution applies to the cohort-mismatch measure, which is why that
script also computes a sampling-noise floor.
