# Pion-versus-electromagnetic analysis plan

## What the first classifier should decide

Define the classification target before training:

- **Reconstructable charged-pion track:** a charged pion above Cherenkov
  threshold with enough visible path for a meaningful track fit.
- **Electromagnetic topology:** a primary electron/photon shower, including
  conversions and (if relevant to the selected sample) photons from
  neutral-pion decay.
- **Other/unreconstructable:** sub-threshold charged pions, pion interactions
  with too little clean track, multi-particle final states, cosmic/beam
  backgrounds, and pathological readout events.

A binary pion/shower label that silently folds the third category into either
class will give an attractive aggregate ROC curve but a poorly defined physics
selection.  Start with three truth categories, then decide which third-category
events belong in the background for the intended cross-section measurement.

For water with refractive index 1.344, a charged pion begins direct Cherenkov
emission at roughly 69 MeV kinetic energy.  A produced pion below threshold is
not a failed fit; it is a different reconstruction regime.  Also note that the
free-nucleon single-pion photoproduction thresholds are about 145 MeV for
`gamma p -> pi0 p` and 151 MeV for `gamma p -> pi+ n`; nuclear binding and
Fermi motion smear this boundary.  Treat the 100--threshold region separately
when defining training samples and reporting performance.

## Recommended reconstruction sequence

1. Apply beam-tagger, data-quality, fiducial, and event-time-cluster
   requirements.  Keep these independent of the later topology score.
2. Calculate inexpensive event features before fitting:
   hit PMTs, pulse count, total PE, PE per hit PMT, charge quantiles and
   concentration, raw/ToF-corrected time spread, beam-angle moments, forward
   charge fractions, angular concentration, and charge-weighted spatial
   moments.
3. Fit the pion track in **both** `cz` hemispheres.  Retain the lower NLL and
   store the hemisphere NLL difference.  The original package only used
   positive `cz`; this update exposes both.
4. Calculate track residual features: Poisson charge deviance, timing residual
   RMS/pulls, ring residual width, observed/predicted total PE ratio, and charge
   shape distance.
5. Fit the deterministic longitudinal shower cone. Compare track and shower NLL, AIC,
   and BIC only when the fits use exactly the same active-PMT mask and the same
   charge/time likelihood conventions.
6. Train an interpretable logistic-regression baseline.  Add a
   histogram-gradient-boosted model only after feature distributions and
   correlations are understood.
7. Choose the score threshold from the physics goal (for example, maximum
   shower rejection at 80% pion efficiency), not from accuracy on a balanced
   training set.
8. Run the pion parameter fit only on the selected sample, but retain failed
   and low-quality fits for efficiency accounting.

## Feature guidance

The strongest first-pass candidates are expected to be:

- hit-PMT and total-PE scale, conditioned on tagged photon energy;
- mean PE per hit PMT and the fraction of charge in the hottest 10% of PMTs;
- forward charge fraction and reconstructed-axis angle to the beam;
- angular spread and PCA eigenvalue fractions of PMT viewing directions;
- ToF-corrected prompt fraction, RMS, IQR, and MAD;
- track charge deviance per degree of freedom and timing-pull RMS;
- track-ring residual width and charge fraction near the fitted Cherenkov ring;
- track-versus-shower likelihood/AIC/BIC differences.

`beam_transverse_radius_*` and `charge_weighted_spatial_rms_mm` are calculated
from PMT locations.  They are useful detector observables, but they are not a
literal transverse shower size.  Detector wall geometry, dead channels, and
vertex position can dominate them.  Prefer angular variables from a nominal or
fitted vertex and validate every spatial feature versus vertex.

Do not feed tagged photon energy to a classifier without deciding whether the
analysis wants energy-dependent discrimination.  A safe comparison is:

1. train without photon energy;
2. train with energy and reweight signal/background to the same energy
   spectrum;
3. quote performance in narrow photon-energy bins for both.

This exposes whether the classifier learned topology or only differing sample
spectra.

## Training and validation

Use independent simulation production or run identifiers as cross-validation
groups.  Random event splitting can leak common calibration, generator, or
production details.  Keep a final untouched production campaign for closure.

Report at least:

- ROC AUC and precision-recall AUC;
- pion efficiency, EM misidentification, and pion purity at the chosen cut;
- performance versus tagged photon energy, pion kinetic energy, pion angle,
  vertex, visible length, interaction channel, and total PE;
- classifier calibration and score stability across runs;
- pion vertex/direction/energy resolution and bias after the topology cut;
- selection efficiency migrations under detector and interaction-model
  variations.

Important detector variations include charge scale/resolution, timing
offset/resolution, inactive PMTs/mPMTs, dark noise, water absorption/scattering,
and beam/target position.  Important physics variations include pion elastic
scattering, absorption, charge exchange, secondary production, and EM shower
modeling.

Data control samples should anchor the input features before unblinding a
signal region.  Tagged electron/photon-like samples test the EM side; available
charged-particle beam samples test ring width, timing, and track-fit residuals.
Plot data/MC agreement for every classifier input and avoid using badly modeled
features solely because they improve simulated separation.

## What “energy” means in the two fits

For a full-length LicketyFit track, kinetic energy is inferred from the range to
Cherenkov threshold.  That interpretation is biased when a pion scatters or
interacts hadronically before slowing to threshold.

The absorption parameterization separates visible length from the full range,
but the initial-energy/full-range parameter can be weakly identified and
correlated with vertex, direction, and light-yield modeling.  Demonstrate MC
closure versus pion energy and interaction mode before treating it as an
unbiased energy reconstruction.  Tagged photon energy is useful as a constraint
only within an explicit reaction/kinematic hypothesis.

The shower fitter reports total **detected** PE. Convert that to MeV
only with a position-, direction-, and run-dependent calibration derived from
simulation and control data.  Its fitted vertex is an effective light-emission
point, not automatically the photon conversion vertex.

## Longitudinal shower model

The default `pdg_longitudinal` model sums fuzzy Cherenkov emission over fixed
quadrature slices downstream of the conversion vertex. Slice weights follow a
PDG-inspired gamma profile in units of the 360.8 mm water radiation length;
the profile energy is fixed from the tagged photon energy, so no additional
shape parameter is floated. Arrival times include charged-particle propagation
to each slice and optical propagation from that slice to each PMT. Use
`emission_model="point"` only for legacy A/B comparisons. The longitudinal and
angular templates still require WCSim calibration in the 100--600 MeV regime.

## When to build a fuller shower fitter

The included fuzzy-cone model is enough to test whether an explicit competing
hypothesis improves classification and to supply an effective direction/width.
Build a detector-specific shower fitter if:

- the shower vertex/direction/energy are physics outputs;
- the track-versus-shower likelihood ratio adds substantial rejection;
- a one-point cone shows systematic residuals versus energy or vertex; or
- neutral-pion two-shower events are an important background/category.

A production model should include a longitudinal gamma-distribution profile,
energy-dependent lateral/angular spread, photon propagation and PMT angular
response, calibrated timing, unhit PMTs, and (if needed) a two-shower
hypothesis.  Fit one- and two-shower models to simulated samples and decide
complexity with held-out likelihood and physics closure, not only training NLL.

## Minimal code path

```python
from fit_single_event import fit_track_both_directions, topology_features
from LicketyFit import ShowerFitter, compare_fit_hypotheses

track_scan = fit_track_both_directions(config, event=event)
track = track_scan["best"]
features = topology_features(track, beam_direction=(0, 0, 1))

shower_fitter = ShowerFitter(
    track["p_locations"],
    pmt_direction_zs=track["direction_zs"],
)
shower = shower_fitter.fit(
    track["obs_pes"],
    track["obs_ts"],
    vertex_seed_mm=(0, 0, 0),
    direction_seed=(0, 0, 1),
)
features.update(compare_fit_hypotheses(track, shower))
```

Train only after adding event identifiers, truth labels, tagged photon energy,
and independent production/run groups to the resulting feature table.
# Balanced pion-versus-shower validation sample

After producing `tagged_gamma_event_topologies.csv`, select 50 events in each
of five useful truth groups and write a compact NPZ containing only those 250
events:

```bash
python3 scripts/select_pion_shower_sample.py \
  outputs/tagged_gamma_truth_all/tagged_gamma_event_topologies.csv \
  --per-category 50 \
  --output-dir outputs/pion_shower_sample
```

The groups are pi+ decay, pi+ interaction, pi- interaction, no pion, and
pi0-only. Selection is reproducible and spread across source files. The compact
NPZ avoids reopening multi-gigabyte production files for every fit.

Before fitting, validate the charge/time inputs and the effect of the fitter's
prompt-time window:

```bash
python3 scripts/study_selected_observables.py \
  outputs/pion_shower_sample/fit_manifest.csv \
  --output-dir outputs/pion_shower_observable_checks
```

This writes per-event metrics and distributions by truth category. In the
current likelihood, track charge is normalized to the observed event total and
the shower total PE is floated, so total charge is a topology feature but not
part of the track-versus-shower absolute-yield comparison.

First run a small pilot and inspect its logs:

```bash
python3 scripts/run_pion_shower_sample.py \
  outputs/pion_shower_sample/fit_manifest.csv \
  --max-events 5 \
  --pilot-seeds \
  --output-dir outputs/pion_shower_fits_pilot
```

Then omit `--max-events` for all 250 events. The runner is resumable: it writes
each event immediately and skips completed events. It compares a forward
PDG-longitudinal shower fit against unrestricted full-length and absorption
pion fits. Positive `delta_nll_shower_minus_pion` favors the best pion fit.
The score is emitted only when the shower fit and at least one pion fit are
valid; command completion alone is not treated as fit validity.

WCSim digit times are passed through without a peak-time cut by default, which
matches the production WCSim driver. The historical single-event one-sided cut
can be reproduced for diagnostics with `--apply-wcsim-peak-window` when running
`run_pion_shower_smoke_test.py` directly.

Summarize the completed study with:

```bash
python3 scripts/analyze_pion_shower_results.py \
  outputs/pion_shower_fits/fit_results.csv
```

The balanced sample is for conditional separation studies. It does not reflect
the very low pion prior in the tagged-gamma beam, so efficiency and background
rejection should be reported separately from expected beam purity.
