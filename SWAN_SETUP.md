# LicketyFit pion/shower analysis on CERN SWAN

This overlay is designed for the analytical fitter in WCTE/LicketyFit pull
request 1 (`jakobrim121:replace-with-licketyfit-official`). It should not be
applied to the older WCTE `main` tree.

## 1. Start a SWAN session

Choose a recent Python 3 configuration. Open a SWAN terminal. SWAN projects
live under `$CERNBOX_HOME/SWAN_projects`, which persists in CERNBox.

```bash
echo "$CERNBOX_HOME"
mkdir -p "$CERNBOX_HOME/SWAN_projects"
cd "$CERNBOX_HOME/SWAN_projects"
```

## 2. Create the correct Git working tree

Clone your existing fork, add WCTE as `upstream`, and make the analysis branch
directly from pull request 1:

```bash
git clone https://github.com/ABlairJamieson/LicketyFit.git
cd LicketyFit
git remote add upstream https://github.com/WCTE/LicketyFit.git
git fetch upstream pull/1/head:collaborator-update
git switch -c pion-shower-analysis-v2 collaborator-update
git log -1 --oneline
```

The last command should show commit `b58353f` or the corresponding pull-request
head commit, with message `Replace LicketyFit with current analytical fitter
implementation`.

If `upstream` already exists, do not add it again:

```bash
git remote -v
git fetch upstream pull/1/head:collaborator-update
```

## 3. Apply this overlay

Upload `LicketyFit-pion-shower-SWAN-overlay.zip` to the `LicketyFit` directory
using the SWAN/CERNBox file browser. From the repository root:

```bash
unzip -l LicketyFit-pion-shower-SWAN-overlay.zip
unzip -o LicketyFit-pion-shower-SWAN-overlay.zip -d .
git status --short
git diff --stat
git diff --check
```

This modifies or adds only the analysis-related files. It does not commit or
push anything.

## 4. Make the Geometry package available

Place the WCTE Geometry repository beside LicketyFit so these paths exist:

```text
SWAN_projects/
  LicketyFit/
  Geometry/
    Geometry/Device.py
    examples/wcte_bldg157.geo
```

If Geometry is already elsewhere in CERNBox or EOS, pass its location with
`--geometry-dir` rather than copying it.

## 5. Check and install Python dependencies

Use SWAN's supplied packages where possible. Check first:

```bash
python - <<'PY'
from importlib.metadata import version

for package in ("numpy", "iminuit", "numba", "matplotlib", "pandas", "uproot", "awkward"):
    try:
        print(f"{package:12s} {version(package)}")
    except Exception:
        print(f"{package:12s} MISSING")
PY
```

Install only missing packages into CERNBox:

```bash
python -m pip install --user iminuit numba matplotlib pandas uproot awkward
```

When starting the SWAN session, enable user-installed Python packages. If a
newly installed package is not visible, restart the kernel/session.

## 6. Verify the package

From the LicketyFit repository root:

```bash
python scripts/check_setup.py
python -m pytest tests/test_analysis.py tests/test_shower_fitter.py -v
```

The added test suite contains nine tests.

## 7. Run the included proton smoke test

```bash
python scripts/run_pion_shower_smoke_test.py --ncall 2000
```

For the longer track and shower comparison:

```bash
python scripts/run_pion_shower_smoke_test.py \
  --ncall 15000 \
  --shower \
  --shower-ncall 5000
```

Results are written to a timestamped directory under `outputs/` as
`report.json` and `diagnostic_arrays.npz`.

With `--shower`, the default shower-vertex seed is obtained from the prompt
PMTs by robust time-of-flight multilateration. This estimates an effective
prompt-light origin and is only a starting point for the full shower fit. To
compare against the track-fit vertex as the seed, add `--shower-seed track`.

The current WCSim reader supplies one charge-weighted time per PMT. If the ROOT
files contain individual photon or pulse times, the converter should also save
the earliest time at each PMT; that is the preferred input for the prompt seed.

## 8. Run a pion WCSim event

For a simulated pion with known kinetic energy, fixing the energy is useful for
a first mechanical check:

```bash
python scripts/run_pion_shower_smoke_test.py \
  --input /eos/PATH/TO/PION_FILE.npz \
  --event-index 0 \
  --particle pion \
  --energy-mev 300 \
  --fit-mode absorption \
  --both-directions \
  --shower
```

Adjust `--fixed-z-mm` if the simulated vertex differs, or pass `--free-z` to
float it. For a first mechanical test, fixing the known simulated vertex makes
failures easier to diagnose.

For analysis-like reconstruction, do not fix the pion energy. The seed values
are pion kinetic energies:

```bash
python scripts/run_pion_shower_smoke_test.py \
  --input /eos/PATH/TO/PION_FILE.npz \
  --event-index 0 \
  --particle pion \
  --energy-mev 300 \
  --fit-mode absorption \
  --free-z \
  --free-energy \
  --ke-seeds-mev 100,150,200,300,450,600 \
  --visible-length-seeds-mm 50,100,200,400,700,1000 \
  --both-directions \
  --shower
```

## 9. Run a gamma WCSim event

There is no `gamma` track model. For a gamma event, fit the competing **pion**
track hypothesis with free pion energy, then compare it with the shower model:

```bash
python scripts/run_pion_shower_smoke_test.py \
  --input /eos/PATH/TO/GAMMA_FILE.npz \
  --event-index 0 \
  --particle pion \
  --energy-mev 300 \
  --fit-mode absorption \
  --free-z \
  --free-energy \
  --ke-seeds-mev 100,150,200,300,450,600 \
  --visible-length-seeds-mm 50,100,200,400,700,1000 \
  --both-directions \
  --shower
```

Here `--energy-mev` is metadata describing the simulated/tagged sample. It is
not used to fix pion energy when `--free-energy` is present.

Do not interpret one event's likelihood difference as classifier performance.
Collect matched pion and gamma/electron samples over photon energy, vertex, and
direction before choosing a cut.
