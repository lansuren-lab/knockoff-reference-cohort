# Data sources

Raw imaging data are **not** redistributed in this archive. Both cohorts are
public.

## ABIDE I
- Source: FCP-INDI public archive,
  `https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative`
- Branches used: `cpac/filt_noglobal` and `niak/filt_noglobal`
- Access: no registration, no data use agreement
- Fetcher: `src/download_abide_roi.py` (AAL116 parcel time series, `.1D`)
- Phenotype: `Phenotypic_V1_0b_preprocessed1.csv` from the same archive

## ABIDE II
- Imaging: same archive, `Derivatives/DCAN`, Power2011 CIFTI parcel time
  series (`.ptseries.nii`). No registration.
- Phenotype: NITRC Neuro Bureau release area,
  `https://www.nitrc.org/frs/?group_id=383`. Requires a NITRC account and
  agreement to the 1000 Functional Connectomes Project terms.
- Fetcher: `src/download_abide2.py`

## Derived features
`src/build_features_v2.py` turns parcel time series into subject-level
feature matrices. For AAL116 the ROI order follows the standard AAL116
sequence; the numeric codes shipped with ABIDE I are **not** the usual AAL
numbering, so the mapping is by position and is checked against 11 anatomical
anchors at load time (hard failure on mismatch).

Constant parcel time series (incomplete brain coverage) are marked missing
rather than imputed; the resulting missingness drives the `>20 subjects` ROI
filter. Affected: 68 of 885 ABIDE I subjects (7.7%), concentrated by site.
