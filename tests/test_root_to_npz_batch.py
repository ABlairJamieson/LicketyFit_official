import subprocess
import sys
from pathlib import Path


def test_condor_generator_scans_root_directory(tmp_path):
    root_dir = tmp_path / "root"
    root_dir.mkdir()
    (root_dir / "b.root").write_bytes(b"")
    (root_dir / "a.root").write_bytes(b"")
    (root_dir / "ignore.txt").write_text("x")
    data_tools = tmp_path / "DataTools"
    (data_tools / "root_utils").mkdir(parents=True)
    (data_tools / "root_utils/event_dump.py").write_text("# test")
    output = tmp_path / "npz"
    submit_dir = tmp_path / "submit"
    script = Path(__file__).resolve().parents[1] / "scripts/root_to_npz_batch.py"
    subprocess.run([sys.executable, str(script), str(root_dir), str(output),
                    "--data-tools", str(data_tools), "--container", "/eos/test.sif",
                    "--prepare-condor", "--submit-dir", str(submit_dir)],
                   check=True, capture_output=True, text=True)
    assert len((submit_dir / "root_files.txt").read_text().splitlines()) == 2
    submit = (submit_dir / "root_to_npz.sub").read_text()
    assert "executable = " in submit
    assert "queue root_file,output_dir from" in submit
