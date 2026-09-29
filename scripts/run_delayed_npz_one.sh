#!/bin/bash
# Process one whole WCSim NPZ on a memory-sized batch worker.
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 INPUT_NPZ OUTPUT_DIR REPO_DIR PYTHON" >&2
    exit 2
fi

input_npz=$1
output_dir=$2
repo_dir=$3
python_bin=$4

if [[ ! -f "$input_npz" || ! -f "$repo_dir/scripts/study_delayed_clusters.py" || ! -x "$python_bin" ]]; then
    echo "Missing input NPZ, repository script, or executable Python: $input_npz $repo_dir $python_bin" >&2
    exit 2
fi

mkdir -p "$output_dir"
if [[ -f "$output_dir/analysis.done" ]]; then
    echo "Already completed: $output_dir"
    exit 0
fi

cd "$repo_dir"
echo "Input: $input_npz"
echo "Output: $output_dir"
"$python_bin" scripts/study_delayed_clusters.py "$input_npz" \
    --all-events --output-dir "$output_dir/clusters"
"$python_bin" scripts/reconstruct_delayed_vertices.py \
    "$output_dir/clusters/clusters.csv" --output-dir "$output_dir/vertices"

test -s "$output_dir/clusters/events.csv"
test -s "$output_dir/clusters/clusters.csv"
test -s "$output_dir/vertices/vertices.csv"
touch "$output_dir/analysis.done"
echo "Finished: $output_dir"
