#!/usr/bin/env python3
"""Prepare one CERN HTCondor job per full tagged-gamma WCSim NPZ file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Do not call resolve() here.  On CERN EOS, /eos/user/... can resolve through
# a /eos/home-* alias which is not accepted by the EosSubmit parser even
# though it refers to the same storage.  Preserve the path spelling used by
# the caller (normally /eos/user/<initial>/<account>/...).
ROOT = Path(__file__).absolute().parents[1]
DEFAULT_NPZ_DIR = Path("/eos/experiment/wcte/MC_Production/v1.5.1/tagged_gamma/converted_npz")


def safe_path(path: Path) -> str:
    """HTCondor's simple argument/queue syntax here requires whitespace-free paths."""
    result = str(path.absolute())
    if any(character.isspace() for character in result):
        raise ValueError(f"Batch paths cannot contain whitespace: {result}")
    return result


def prepare(npz_dir: Path, output_dir: Path, *, python_bin: Path,
            memory_gb: int, max_files: int | None = None) -> tuple[Path, int]:
    if memory_gb < 1 or (max_files is not None and max_files < 1):
        raise ValueError("memory_gb and max_files must be positive")
    files = sorted(path for path in npz_dir.glob("mdt_e1000MeV_gamma_cyl_HD*.npz")
                   if "_events" not in path.stem)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise ValueError(f"No full tagged-gamma NPZ files found in {npz_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logs = output_dir / "logs"
    logs.mkdir(exist_ok=True)
    manifest = output_dir / "jobs.txt"
    submit = output_dir / "delayed_gamma.sub"
    repo = safe_path(ROOT)
    python_path = safe_path(python_bin)
    worker = safe_path(ROOT / "scripts/run_delayed_npz_one.sh")
    with manifest.open("w", encoding="utf-8") as handle:
        for source in files:
            destination = output_dir / source.stem
            handle.write(f"{safe_path(source)} {safe_path(destination)}\n")
    submit.write_text(
        "\n".join([
            "# Submit on a CERN EosSubmit schedd: all paths below are on EOS.",
            "universe = vanilla",
            "executable = /bin/bash",
            f"arguments = {worker} $(input_path) $(output_path) {repo} {python_path}",
            f"output = {safe_path(logs)}/$(ClusterId).$(ProcId).out",
            f"error = {safe_path(logs)}/$(ClusterId).$(ProcId).err",
            f"log = {safe_path(logs)}/$(ClusterId).log",
            "should_transfer_files = NO",
            "request_cpus = 1",
            f"request_memory = {memory_gb} GB",
            "request_disk = 4 GB",
            f"queue input_path,output_path from {safe_path(manifest)}",
            "",
        ]), encoding="utf-8",
    )
    return submit, len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/delayed_tagged_gamma_batch")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--memory-gb", type=int, default=64)
    parser.add_argument("--max-files", type=int, default=None,
                        help="Prepare just the first N full files for a batch pilot")
    args = parser.parse_args()
    submit, count = prepare(args.npz_dir, args.output_dir, python_bin=args.python,
                            memory_gb=args.memory_gb, max_files=args.max_files)
    print(f"Prepared {count} per-file jobs: {submit}")
    print("Submit from an EosSubmit-configured CERN session with:")
    print(f"  condor_submit {submit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
