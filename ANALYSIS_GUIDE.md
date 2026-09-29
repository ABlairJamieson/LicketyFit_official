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

## Delayed Michel-electron pilot

`scripts/study_delayed_clusters.py` searches raw digit times for a prompt burst
and later bursts using a sliding 50 ns window. It counts distinct PMTs and
reports every candidate in `clusters.csv` plus the strongest candidate in
`events.csv`. No fitter, geometry file, or truth labels are required. This is
an exploratory pi+ decay tag; it is not an all-pion selection.

On CERN, from the LicketyFit repository with its Python environment active:

```bash
NPZDIR=/eos/experiment/wcte/MC_Production/v1.5.1/tagged_gamma/converted_npz
python3 scripts/study_delayed_clusters.py \
  "$NPZDIR/mdt_wCDS_pi+_Uniform_0_800MeV_0001.npz" \
  "$NPZDIR/mdt_wCDS_pi-_Uniform_0_800MeV_0000.npz" \
  --max-events-per-file 1000 \
  --output-dir outputs/delayed_pion_pilot
```

Check the actual filenames first; the pi+ and pi- production file numbers may
differ. The default requires 10 distinct PMTs in a 50 ns window from 200 ns
to 10 us after the prompt cluster. Try `--width-ns 20` and a range of
`--min-pmts` values after inspecting the pilot distributions. The prompt
cluster is selected near the first 10-PMT coincidence (`--prompt-min-pmts`),
ignoring isolated early noise hits. Candidate
`delta_t_ns` uses the median digit time in each cluster.

In `--time-mode auto`, digit times are placed on a common clock by adding
`trigger_time[digi_hit_trigger]` when both arrays are present. This follows
the WCSim digit-time definition (relative to the trigger header). If metadata
is absent, the output marks `raw_missing_trigger_metadata`; inspect this
before interpreting microsecond delays. `--time-mode raw` is provided for
an explicit comparison. The CSV records the timing mode, number of triggers,
and observed digit span for each event.
The default also rejects times before -1 ms, because the pion pilot contained
digits near -214,748,368 ns that formed false prompt anchors. The number of
rejected digits is reported per event; use `--min-time-ns` to change this
boundary when the input has a different clock convention.

First inspect the `delta_t_ns` distribution and the rate of candidate bursts
in each pion sample. Then compare with no-pion tagged-gamma events to choose
a PMT threshold from measured accidental/background rates. A mu+ Michel
population should show a delay scale near 2.2 us; pi- can also be absorbed or
produce mu- that captures on oxygen. A candidate is evidence for a delayed
muon decay, not by itself proof that a pion was produced. Confirm that WCSim
retained hits over the intended delayed search interval before using the
absence of a candidate as a negative tag.

Plot the strongest delayed candidate per event using 30 bins between 100
and 6100 ns (200 ns per bin), with separate panels for pi+ and pi-:

```bash
python3 scripts/plot_delayed_times.py \
  outputs/delayed_pion_pilot_v3/events.csv \
  --output outputs/delayed_pion_pilot_v3/delayed_time_histogram.png
```

The plot title for each panel gives the total number of candidates and the
number falling inside the displayed range. This is a candidate timing plot,
not a truth-matched decay measurement. The pi+ panel also shows an unbinned
truncated-exponential plus flat-background fit over 500–6100 ns. Its printed
profile-likelihood interval includes statistical uncertainty only. The
result is an *apparent* delay constant until the trigger/readout efficiency
versus delay and non-Michel backgrounds are measured. Pass `--bins 60` for
100 ns bins, `--fit-min-ns` to adjust the fitted interval, or `--no-fit` to
show the histograms alone.

Inspect the size of the best delayed burst with separate distributions of
digit hits, distinct hit PMTs, and total digit charge:

```bash
python3 scripts/plot_delayed_cluster_hits.py \
  outputs/delayed_pion_pilot_v3/events.csv \
  --output outputs/delayed_pion_pilot_v3/delayed_cluster_hits.png
```

The hit and charge distributions can guide a possible high-light cut, but
their relation to Michel energy also depends on the event position and PMT
coverage. Keep any cut exploratory until a no-pion control is measured.

To estimate the effective light origin of each best delayed cluster, first
try 20 candidates, then omit `--max-candidates` for all candidates:

```bash
python3 scripts/reconstruct_delayed_vertices.py \
  outputs/delayed_pion_pilot_v3/clusters.csv \
  --max-candidates 20 \
  --output-dir outputs/delayed_pion_pilot_v3/vertex_pilot

python3 scripts/reconstruct_delayed_vertices.py \
  outputs/delayed_pion_pilot_v3/clusters.csv \
  --output-dir outputs/delayed_pion_pilot_v3/vertices_all
```

This uses the existing point multilateration seed on digits inside the saved
cluster window. The PMT coordinates come from `tables/wcsim_wcte_mapping.txt`
in WCSim centimetres, converted to millimetres; the separate design-geometry
file has a different origin and would bias these WCSim vertex estimates.
`vertices.csv` records failures and timing residuals. The reco-only PNG,
`delayed_vertices_tank_ry.png`, shows
the tank's cylindrical radius `R=sqrt(x^2+z^2)` versus vertical `y`, plus an
`x,z` cross-section. The second PNG, `delayed_vertices_truth_overlay.png`,
marks the production point of a truth muon-decay electron or positron with an
`x` and links it to the reconstructed point. Matches require the daughter
track time relative to the earliest primary track to agree with the measured
prompt-to-delayed-cluster separation within 100 ns (adjustable with
`--truth-match-tolerance-ns`). Unmatched and missing-truth events are reported
in the CSV and excluded from the overlay. The lines show projected
differences; `reco_truth_distance_3d_mm` gives the actual three-dimensional
distance. These point-source estimates can be biased by the electron's finite
track; no containment or reconstruction-quality cut is applied yet.

The vertex run also writes `delayed_vertex_residuals_xyz.png`: three signed
histograms of reconstructed minus matched-truth `x`, `y`, and `z` in cm. Each
panel reports its median and RMS. Only successfully fitted, time-matched
muon-decay daughters enter the plot; the title gives that denominator. To
replot an existing `vertices.csv` without repeating reconstruction:

```bash
python3 scripts/plot_delayed_vertex_residuals.py \
  outputs/delayed_pion_pilot_v3/vertex_pilot/vertices.csv
```

Your `delayed_pion_pilot_v3` clustering already processed all 1,000 events
from each of the cited pion files. Thus the fastest full-file follow-up is
to reconstruct every candidate from those existing clusters, then draw
separate residual panels for each primary type:

```bash
python3 scripts/reconstruct_delayed_vertices.py \
  outputs/delayed_pion_pilot_v3/clusters.csv \
  --output-dir outputs/delayed_pion_pilot_v3/vertices_all
python3 scripts/plot_delayed_vertex_residuals.py \
  outputs/delayed_pion_pilot_v3/vertices_all/vertices.csv --primary-pid 211
python3 scripts/plot_delayed_vertex_residuals.py \
  outputs/delayed_pion_pilot_v3/vertices_all/vertices.csv --primary-pid -211
```

To repeat clustering on the full 1,000-event pion files independently, keep
their outputs separate:

```bash
NPZDIR=/eos/experiment/wcte/MC_Production/v1.5.1/tagged_gamma/converted_npz
python3 scripts/study_delayed_clusters.py \
  "$NPZDIR/mdt_wCDS_pi+_Uniform_0_800MeV_0001.npz" \
  --max-events-per-file 1000 --output-dir outputs/delayed_pi_plus_full
python3 scripts/reconstruct_delayed_vertices.py \
  outputs/delayed_pi_plus_full/clusters.csv \
  --output-dir outputs/delayed_pi_plus_full/vertices

python3 scripts/study_delayed_clusters.py \
  "$NPZDIR/mdt_wCDS_pi-_Uniform_0_800MeV_0000.npz" \
  --max-events-per-file 1000 --output-dir outputs/delayed_pi_minus_full
python3 scripts/reconstruct_delayed_vertices.py \
  outputs/delayed_pi_minus_full/clusters.csv \
  --output-dir outputs/delayed_pi_minus_full/vertices
```

For a tagged-gamma control pilot, use the existing 100-event skim first:

```bash
python3 scripts/study_delayed_clusters.py \
  "$NPZDIR/mdt_e1000MeV_gamma_cyl_HD5_events00000-00099.npz" \
  --max-events-per-file 100 \
  --output-dir outputs/delayed_gamma_hd5_100
python3 scripts/reconstruct_delayed_vertices.py \
  outputs/delayed_gamma_hd5_100/clusters.csv \
  --output-dir outputs/delayed_gamma_hd5_100/vertices
```

If the gamma skim has no delayed candidate, the reconstruction command writes
empty tables and annotated plots; that is an informative count. The full
gamma files contain 25,000 events each. The current NPZ readers decompress
whole object-array fields even with `--max-events-per-file`, so a full gamma
file should run as a memory-sized batch job or first be converted into smaller
NPZ event chunks; do not assume the 100-event limit makes a 25,000-event file
cheap to open interactively on lxplus. The 100-event control is too small to
measure the rare pion-production population or a reliable false-tag rate.

### Full tagged-gamma NPZ files on CERN batch

The repository provides one batch job per complete `mdt_e1000MeV_gamma_cyl_HD*.npz`
file, excluding the 100-event `_events...` skim. Start from the LicketyFit
repository on EOS with the working virtual environment active. Prepare a
single-file pilot first:

```bash
python3 scripts/prepare_delayed_gamma_batch.py \
  --max-files 1 --memory-gb 64 \
  --repo-dir /eos/user/a/ajamieso/SWAN_projects/LicketyFitTutorial_Sep2026/LicketyFit_official \
  --output-dir outputs/delayed_tagged_gamma_batch_pilot
condor_submit outputs/delayed_tagged_gamma_batch_pilot/delayed_gamma.sub
condor_q "$USER"
```

Use the same CERN **EosSubmit** schedd configuration that worked for the
earlier conversion jobs; a standard schedd rejects direct `/eos` paths in
this submit file. The generated file and all logs/output paths live on EOS.
The job uses the currently active `python3` path, so check that this Python
has NumPy, SciPy, and Matplotlib and is accessible on worker nodes. Check
`outputs/delayed_tagged_gamma_batch_pilot/logs/` for stdout and stderr.
Successful jobs create `analysis.done` inside their per-file output directory,
alongside `clusters/events.csv`, `clusters/clusters.csv`, and `vertices/`.
Only trust outputs with that marker: an interrupted job may leave partial CSVs.

If the pilot finishes within the requested memory and the results are
sensible, regenerate the submit list in the same output directory for all
files. The completed pilot file will be skipped because it has `analysis.done`:

```bash
python3 scripts/prepare_delayed_gamma_batch.py \
  --memory-gb 64 \
  --repo-dir /eos/user/a/ajamieso/SWAN_projects/LicketyFitTutorial_Sep2026/LicketyFit_official \
  --output-dir outputs/delayed_tagged_gamma_batch_pilot
condor_submit outputs/delayed_tagged_gamma_batch_pilot/delayed_gamma.sub
```

The generator prints the job count; at the time of writing, 13 full tagged
gamma NPZ files were present. `--memory-gb` can be raised if a pilot job is
killed for memory use. Each worker runs clustering with `--all-events` and
then vertex reconstruction in a *separate Python process*. This releases the
clustering arrays before reconstruction, but does **not** stream individual
events out of an NPZ member: NumPy still decompresses each requested
object-array field in full. Thus memory is bounded per file/job, not per
event. Do not run all full gamma files concurrently on an lxplus head node.
