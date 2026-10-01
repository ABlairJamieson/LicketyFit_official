#!/usr/bin/env python3
"""Convert a directory of WCSim ROOT files to NPZ, locally or via Condor.

The actual ROOT reader is WatChMaL/DataTools ``root_utils/event_dump.py``.
This wrapper supplies the generic directory-to-directory workflow and creates
one EOS-compatible Condor job per ROOT file when ``--prepare-condor`` is used.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def find_root_files(root_dir: Path) -> list[Path]:
    return sorted(path for path in root_dir.glob("*.root") if path.is_file())


def convert_one(root_file: Path, output_dir: Path, data_tools: Path,
                container: Path | None) -> None:
    event_dump = data_tools / "root_utils/event_dump.py"
    if not event_dump.is_file():
        raise FileNotFoundError(f"Missing DataTools converter: {event_dump}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if container is None:
        command = [sys.executable, str(event_dump), str(root_file), "-d", str(output_dir)]
    else:
        command = ["apptainer", "exec", "--bind", "/eos:/eos", str(container),
                   "python", str(event_dump), str(root_file), "-d", str(output_dir)]
    subprocess.run(command, check=True)


def prepare_condor(root_dir: Path, output_dir: Path, data_tools: Path,
                   container: Path, submit_dir: Path, memory_gb: int) -> Path:
    files = find_root_files(root_dir)
    if not files:
        raise ValueError(f"No ROOT files found in {root_dir}")
    submit_dir.mkdir(parents=True, exist_ok=True)
    manifest = submit_dir / "root_files.txt"
    logs = submit_dir / "logs"
    logs.mkdir(exist_ok=True)
    worker = Path(__file__).absolute()
    # EOS submit requires the executable and all output/error paths to be EOS.
    with manifest.open("w", encoding="utf-8") as handle:
        for root_file in files:
            handle.write(f"{root_file} {output_dir}\n")
    submit = submit_dir / "root_to_npz.sub"
    submit.write_text("\n".join([
        "universe = vanilla",
        f"executable = {worker}",
        f"arguments = --worker $(root_file) $(output_dir) {data_tools} {container}",
        f"output = {logs}/$(ClusterId).$(ProcId).out",
        f"error = {logs}/$(ClusterId).$(ProcId).err",
        f"log = {logs}/$(ClusterId).log",
        "should_transfer_files = NO",
        "request_cpus = 1",
        f"request_memory = {memory_gb} GB",
        "request_disk = 4 GB",
        f"queue root_file,output_dir from {manifest}",
        "",
    ]), encoding="utf-8")
    print(f"Prepared {len(files)} jobs: {submit}")
    return submit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_dir", type=Path, nargs="?")
    parser.add_argument("output_dir", type=Path, nargs="?")
    parser.add_argument("--data-tools", type=Path, required=False,
                        help="DataTools checkout containing root_utils/event_dump.py")
    parser.add_argument("--container", type=Path, default=None,
                        help="WCTE Apptainer image; omit for a prepared local environment")
    parser.add_argument("--prepare-condor", action="store_true")
    parser.add_argument("--submit-dir", type=Path, default=None)
    parser.add_argument("--memory-gb", type=int, default=8)
    parser.add_argument("--worker", nargs=4, metavar=("ROOT", "OUTPUT", "DATATOOLS", "CONTAINER"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        root_file, output_dir, data_tools, container = map(Path, args.worker)
        convert_one(root_file, output_dir, data_tools, container)
        return 0
    if args.root_dir is None or args.output_dir is None or args.data_tools is None:
        parser.error("root_dir, output_dir, and --data-tools are required")
    if args.prepare_condor:
        if args.container is None:
            parser.error("--container is required with --prepare-condor")
        submit_dir = args.submit_dir or args.output_dir / "condor"
        prepare_condor(args.root_dir, args.output_dir, args.data_tools,
                       args.container, submit_dir, args.memory_gb)
    else:
        files = find_root_files(args.root_dir)
        for index, root_file in enumerate(files, 1):
            print(f"[{index}/{len(files)}] {root_file.name}", flush=True)
            convert_one(root_file, args.output_dir, args.data_tools, args.container)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
