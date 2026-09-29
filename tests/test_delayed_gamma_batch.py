import subprocess
import sys
from pathlib import Path


def test_submit_generator_excludes_small_skims_and_can_limit_pilot(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in (
        "mdt_e1000MeV_gamma_cyl_HD1.npz",
        "mdt_e1000MeV_gamma_cyl_HD2.npz",
        "mdt_e1000MeV_gamma_cyl_HD1_events00000-00099.npz",
    ):
        (inputs / name).write_bytes(b"test")
    output = tmp_path / "batch"
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_delayed_gamma_batch.py"
    subprocess.run([sys.executable, str(script), "--npz-dir", str(inputs),
                    "--output-dir", str(output), "--repo-dir", "/eos/user/a/test/LicketyFit_official",
                    "--max-files", "1", "--memory-gb", "80"],
                   check=True, capture_output=True, text=True)
    jobs = (output / "jobs.txt").read_text(encoding="utf-8").splitlines()
    submit = (output / "delayed_gamma.sub").read_text(encoding="utf-8")
    assert len(jobs) == 1
    assert "HD1.npz" in jobs[0]
    assert "_events" not in jobs[0]
    assert "request_memory = 80 GB" in submit
    assert "queue input_path,output_path from" in submit
    assert "executable = " in submit
    assert "run_delayed_npz_one.sh" in submit
    assert "arguments = $(input_path)" in submit
    assert "test\\LicketyFit_official" in submit or "test/LicketyFit_official" in submit
    assert "home-i01" not in submit
