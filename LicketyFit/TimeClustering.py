"""Small, geometry-free time-cluster finder for WCSim digitized hits.

All times are in ns.  A candidate is a group of distinct PMTs in a sliding
window; a fixed-bin histogram would split bursts that cross a bin boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TimeCluster:
    start_ns: float
    end_ns: float
    center_ns: float
    n_hits: int
    n_pmts: int
    charge: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def event_hit_times(event: dict, mode: str = "auto") -> tuple[np.ndarray, str]:
    """Return times on a common clock and the convention used.

    WCSim digit times are relative to their trigger header date.  DataTools
    exports that date as ``trigger_time`` and the digit's trigger index as
    ``digi_hit_trigger``.  In auto mode, use both fields when present.  If one
    is absent, raw times are returned and the caller should check the report.
    """
    if mode not in {"auto", "raw", "trigger-plus"}:
        raise ValueError("mode must be auto, raw, or trigger-plus")
    times = np.asarray(event["digi_hit_time"], dtype=float).reshape(-1)
    if mode == "raw":
        return times, "raw"
    has_metadata = "digi_hit_trigger" in event and "trigger_time" in event
    if not has_metadata:
        if mode == "trigger-plus":
            raise ValueError("trigger-plus requires digi_hit_trigger and trigger_time")
        return times, "raw_missing_trigger_metadata"
    indices = np.asarray(event["digi_hit_trigger"], dtype=int).reshape(-1)
    offsets = np.asarray(event["trigger_time"], dtype=float).reshape(-1)
    if indices.size != times.size:
        raise ValueError("digi_hit_trigger and digi_hit_time have different lengths")
    if times.size and (offsets.size == 0 or np.any(indices < 0) or np.any(indices >= offsets.size)):
        raise ValueError("digit trigger index is outside trigger_time")
    return times + offsets[indices], "trigger_plus"


def _best_window(
    times: np.ndarray, pmts: np.ndarray, charge: np.ndarray, width_ns: float,
) -> TimeCluster | None:
    if times.size == 0:
        return None
    order = np.argsort(times, kind="stable")
    t, p, q = times[order], pmts[order], charge[order]
    left = 0
    counts: dict[int, int] = {}
    best_left = best_right = 0
    best_pmts = -1
    best_charge = -1.0
    running_charge = 0.0
    for right in range(t.size):
        key = int(p[right])
        counts[key] = counts.get(key, 0) + 1
        running_charge += q[right]
        while t[right] - t[left] > width_ns:
            key = int(p[left])
            counts[key] -= 1
            if counts[key] == 0:
                del counts[key]
            running_charge -= q[left]
            left += 1
        if (len(counts), running_charge) > (best_pmts, best_charge):
            best_left, best_right = left, right + 1
            best_pmts, best_charge = len(counts), running_charge
    selected_t = t[best_left:best_right]
    return TimeCluster(
        start_ns=float(selected_t[0]),
        end_ns=float(selected_t[-1]),
        center_ns=float(np.median(selected_t)),
        n_hits=int(selected_t.size),
        n_pmts=int(best_pmts),
        charge=float(np.sum(q[best_left:best_right])),
    )


def _first_coincidence_end(
    times: np.ndarray, pmts: np.ndarray, width_ns: float, min_pmts: int,
) -> float | None:
    """Find the first sliding window with enough distinct PMTs."""
    order = np.argsort(times, kind="stable")
    t, p = times[order], pmts[order]
    left = 0
    counts: dict[int, int] = {}
    for right in range(t.size):
        key = int(p[right])
        counts[key] = counts.get(key, 0) + 1
        while t[right] - t[left] > width_ns:
            key = int(p[left])
            counts[key] -= 1
            if counts[key] == 0:
                del counts[key]
            left += 1
        if len(counts) >= min_pmts:
            return float(t[right])
    return None


def find_delayed_clusters(
    times_ns: np.ndarray,
    pmts: np.ndarray,
    charges: np.ndarray,
    *,
    width_ns: float = 50.0,
    min_pmts: int = 10,
    prompt_min_pmts: int = 10,
    search_start_ns: float = 200.0,
    search_end_ns: float = 10_000.0,
    prompt_search_ns: float = 200.0,
    max_clusters: int = 5,
) -> tuple[TimeCluster | None, list[TimeCluster]]:
    """Find the largest prompt burst, then delayed bursts relative to it.

    Delayed candidates are ranked by distinct PMTs, then charge.  The caller
    must check that the simulation readout actually covers the search interval.
    This method does not correct hit times for photon flight time.
    """
    if width_ns <= 0 or min_pmts < 1 or prompt_min_pmts < 1 or max_clusters < 1:
        raise ValueError("width_ns, min_pmts, prompt_min_pmts, and max_clusters must be positive")
    if search_start_ns < 0 or search_end_ns <= search_start_ns or prompt_search_ns <= 0:
        raise ValueError("invalid delayed search interval")
    times = np.asarray(times_ns, dtype=float).reshape(-1)
    ids = np.asarray(pmts, dtype=int).reshape(-1)
    charge = np.asarray(charges, dtype=float).reshape(-1)
    if not (times.size == ids.size == charge.size):
        raise ValueError("hit arrays have different lengths")
    good = np.isfinite(times) & np.isfinite(charge) & (charge > 0)
    times, ids, charge = times[good], ids[good], charge[good]
    # Isolated early noise hits cannot define t0. Anchor to the first actual
    # coincidence, then find the densest prompt window nearby. A Michel
    # electron can be brighter than its preceding pion/muon light.
    anchor = _first_coincidence_end(times, ids, width_ns, prompt_min_pmts)
    if anchor is None:
        return None, []
    early = (times >= anchor - width_ns) & (times <= anchor + prompt_search_ns)
    prompt = _best_window(times[early], ids[early], charge[early], width_ns)
    if prompt is None:
        return None, []
    delta = times - prompt.center_ns
    remaining = (delta >= search_start_ns) & (delta <= search_end_ns)
    candidates: list[TimeCluster] = []
    for _ in range(max_clusters):
        candidate = _best_window(times[remaining], ids[remaining], charge[remaining], width_ns)
        if candidate is None or candidate.n_pmts < min_pmts:
            break
        candidates.append(candidate)
        # One physical burst can produce several overlapping maximum windows.
        remaining &= (times < candidate.start_ns - width_ns) | (times > candidate.end_ns + width_ns)
    return prompt, candidates
