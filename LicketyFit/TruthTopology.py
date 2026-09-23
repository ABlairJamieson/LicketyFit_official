"""Conservative truth-topology summaries for DataTools WCSim NPZ files.

DataTools stores one object array per event for track-level information.  The
``track_parent`` field in the files used by WCTE commonly contains a parent
PDG code (and sentinel values), rather than a unique parent track identifier.
Consequently this module reports inferred topology signatures, not exact
Geant4 process labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PION_PDGS = frozenset((-211, 211))
EM_PDGS = frozenset((-11, 11, 22))
MUON_PDGS = frozenset((-13, 13))
NUCLEON_PDGS = frozenset((2112, 2212))


def _event_array(data: Any, key: str, event: int, dtype: Any = None) -> np.ndarray:
    array = np.asarray(data[key][event])
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array.reshape(-1)


def _particle_family(pid: int) -> str:
    pid = int(pid)
    if pid in PION_PDGS:
        return "charged_pion"
    if pid in (-11, 11):
        return "electron_positron"
    if pid == 22:
        return "gamma"
    if pid in MUON_PDGS:
        return "muon"
    if pid == 111:
        return "pi0"
    if pid == 2212:
        return "proton"
    if pid == 2112:
        return "neutron"
    if abs(pid) >= 1_000_000_000:
        return "nucleus"
    if pid == 0:
        return "optical_photon"
    return "other"


def _track_id_to_pid(track_ids: np.ndarray, track_pids: np.ndarray) -> tuple[dict[int, int], set[int]]:
    """Build an unambiguous track-ID lookup and identify conflicting IDs."""

    candidates: dict[int, set[int]] = defaultdict(set)
    for track_id, pid in zip(track_ids, track_pids):
        candidates[int(track_id)].add(int(pid))
    ambiguous = {track_id for track_id, pids in candidates.items() if len(pids) != 1}
    mapping = {
        track_id: next(iter(pids))
        for track_id, pids in candidates.items()
        if track_id not in ambiguous
    }
    return mapping, ambiguous


def infer_event_topology(data: Any, event: int) -> dict[str, Any]:
    """Infer a conservative physics topology and light-source composition."""

    required = ("pid", "energy", "track_pid", "track_parent", "track_id")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"NPZ is missing required truth fields: {', '.join(missing)}")

    primary_pid = int(data["pid"][event])
    primary_energy = float(data["energy"][event])
    track_pid = _event_array(data, "track_pid", event, int)
    track_parent = _event_array(data, "track_parent", event, int)
    track_id = _event_array(data, "track_id", event, int)
    if not (len(track_pid) == len(track_parent) == len(track_id)):
        raise ValueError(f"Event {event}: track truth arrays have inconsistent lengths")

    pion_parent = np.isin(track_parent, tuple(PION_PDGS))
    pion_parented_pid = track_pid[pion_parent]
    has_primary_pion = primary_pid in PION_PDGS
    has_pi0 = bool(np.any(track_pid == 111))
    has_pion_parented_pi0 = bool(np.any(pion_parented_pid == 111))
    has_pion_parented_nucleon = bool(np.any(np.isin(pion_parented_pid, tuple(NUCLEON_PDGS))))
    has_pion_parented_nucleus = bool(np.any(np.abs(pion_parented_pid) >= 1_000_000_000))
    has_pion_parented_gamma = bool(np.any(pion_parented_pid == 22))
    has_pion_decay_muon = bool(np.any(pion_parent & np.isin(track_pid, tuple(MUON_PDGS))))
    has_outgoing_charged_pion = bool(np.any(pion_parent & np.isin(track_pid, tuple(PION_PDGS))))
    hadronic_signature = bool(
        has_pion_parented_pi0
        or has_pion_parented_nucleon
        or has_pion_parented_nucleus
        or has_outgoing_charged_pion
    )

    if not has_primary_pion:
        topology = "no_primary_charged_pion"
    elif has_pion_parented_pi0:
        topology = "interacting_pion_with_pi0"
    elif hadronic_signature:
        topology = "interacting_pion_no_pi0"
    elif has_pion_decay_muon:
        topology = "pion_decay_candidate"
    else:
        topology = "clean_pion_candidate"

    pid_counts = Counter(int(pid) for pid in track_pid)
    result: dict[str, Any] = {
        "event_index": int(event),
        "event_id": int(data["event_id"][event]) if "event_id" in data else int(event),
        "primary_pid": primary_pid,
        "primary_energy_mev": primary_energy,
        "topology": topology,
        "has_primary_pion": has_primary_pion,
        "hadronic_signature": hadronic_signature,
        "has_pi0": has_pi0,
        "has_pion_parented_pi0": has_pion_parented_pi0,
        "has_pion_parented_nucleon": has_pion_parented_nucleon,
        "has_pion_parented_nucleus": has_pion_parented_nucleus,
        "has_pion_parented_gamma": has_pion_parented_gamma,
        "has_outgoing_charged_pion": has_outgoing_charged_pion,
        "has_pion_decay_muon": has_pion_decay_muon,
        "n_tracks": int(len(track_pid)),
        "n_pi0": int(pid_counts[111]),
        "n_charged_pion_tracks": int(pid_counts[-211] + pid_counts[211]),
        "n_electron_positron_tracks": int(pid_counts[-11] + pid_counts[11]),
        "n_gamma_tracks": int(pid_counts[22]),
        "n_proton_tracks": int(pid_counts[2212]),
        "n_neutron_tracks": int(pid_counts[2112]),
        "n_muon_tracks": int(pid_counts[-13] + pid_counts[13]),
    }

    light_counts: Counter[str] = Counter()
    if "true_hit_parent" in data:
        hit_parents = _event_array(data, "true_hit_parent", event, int)
        id_to_pid, ambiguous_ids = _track_id_to_pid(track_id, track_pid)
        for parent_id in hit_parents:
            parent_id = int(parent_id)
            if parent_id in ambiguous_ids:
                light_counts["ambiguous_track_id"] += 1
            elif parent_id in id_to_pid:
                light_counts[_particle_family(id_to_pid[parent_id])] += 1
            else:
                light_counts["unmapped"] += 1
        total_hits = int(len(hit_parents))
    else:
        total_hits = 0

    result["n_true_hits"] = total_hits
    families: Iterable[str] = (
        "charged_pion",
        "electron_positron",
        "gamma",
        "muon",
        "proton",
        "neutron",
        "nucleus",
        "other",
        "ambiguous_track_id",
        "unmapped",
    )
    for family in families:
        count = int(light_counts[family])
        result[f"true_hits_{family}"] = count
        result[f"true_hit_fraction_{family}"] = count / total_hits if total_hits else float("nan")
    result["true_hit_fraction_em"] = (
        (light_counts["electron_positron"] + light_counts["gamma"]) / total_hits
        if total_hits
        else float("nan")
    )
    return result


def infer_tagged_gamma_topology(data: Any, event: int) -> dict[str, Any]:
    """Infer charged-pion production and subsequent topology in a gamma event.

    The DataTools files available for this analysis encode parent information
    as PDG-like values rather than a complete, reliable track-ID ancestry tree.
    The returned labels are therefore conservative signatures, not generator
    process labels or exclusive reaction-channel identifications.
    """

    result = infer_event_topology(data, event)
    track_pid = _event_array(data, "track_pid", event, int)
    track_parent = _event_array(data, "track_parent", event, int)
    track_energy = (
        _event_array(data, "track_energy", event, float)
        if "track_energy" in data
        else np.full(len(track_pid), np.nan)
    )

    has_plus = bool(np.any(track_pid == 211))
    has_minus = bool(np.any(track_pid == -211))
    has_pi0 = bool(np.any(track_pid == 111))

    def pion_signature(pion_pid: int) -> dict[str, Any]:
        parent_mask = track_parent == pion_pid
        children = track_pid[parent_mask]
        has_parented_pi0 = bool(np.any(children == 111))
        has_parented_nucleon = bool(np.any(np.isin(children, tuple(NUCLEON_PDGS))))
        has_parented_nucleus = bool(np.any(np.abs(children) >= 1_000_000_000))
        has_outgoing_pion = bool(np.any(np.isin(children, tuple(PION_PDGS))))
        has_decay_muon = bool(np.any(np.isin(children, tuple(MUON_PDGS))))
        hadronic = bool(
            has_parented_pi0
            or has_parented_nucleon
            or has_parented_nucleus
            or has_outgoing_pion
        )

        # A pion whose parent is not itself is the best available proxy for a
        # newly produced pion.  Take the maximum when multiple candidates exist.
        produced_mask = (track_pid == pion_pid) & (track_parent != pion_pid)
        energies = track_energy[produced_mask]
        energies = energies[np.isfinite(energies)]
        return {
            "hadronic": hadronic,
            "with_pi0": has_parented_pi0,
            "decay": has_decay_muon,
            "max_produced_energy_mev": float(np.max(energies)) if energies.size else float("nan"),
        }

    plus = pion_signature(211)
    minus = pion_signature(-211)

    if has_plus and has_minus:
        production = "charged_pions_both_signs"
        topology = "multiple_sign_charged_pions"
    elif has_plus:
        production = "pi_plus_produced"
        if plus["with_pi0"]:
            topology = "pi_plus_interacting_with_pi0"
        elif plus["hadronic"]:
            topology = "pi_plus_interacting_no_pi0"
        elif plus["decay"]:
            topology = "pi_plus_decay_candidate"
        else:
            topology = "pi_plus_clean_candidate"
    elif has_minus:
        production = "pi_minus_produced"
        if minus["with_pi0"]:
            topology = "pi_minus_interacting_with_pi0"
        elif minus["hadronic"]:
            topology = "pi_minus_interacting_no_pi0"
        elif minus["decay"]:
            topology = "pi_minus_decay_candidate"
        else:
            topology = "pi_minus_clean_candidate"
    elif has_pi0:
        production = "pi0_only"
        topology = "pi0_without_charged_pion"
    else:
        production = "no_pion"
        topology = "no_pion"

    result.update(
        {
            "production_class": production,
            "tagged_gamma_topology": topology,
            "has_produced_charged_pion": has_plus or has_minus,
            "has_pi_plus": has_plus,
            "has_pi_minus": has_minus,
            "has_pi0_anywhere": has_pi0,
            "pi_plus_hadronic_signature": plus["hadronic"],
            "pi_minus_hadronic_signature": minus["hadronic"],
            "pi_plus_decay_signature": plus["decay"],
            "pi_minus_decay_signature": minus["decay"],
            "max_produced_pi_plus_energy_mev": plus["max_produced_energy_mev"],
            "max_produced_pi_minus_energy_mev": minus["max_produced_energy_mev"],
        }
    )
    return result


def analyze_truth_file(path: str | Path, first_event: int = 0, max_events: int | None = None) -> list[dict[str, Any]]:
    """Analyze a contiguous range of events in a DataTools NPZ file."""

    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        n_events = len(data["pid"])
        if first_event < 0 or first_event >= n_events:
            raise IndexError(f"first_event={first_event} outside file with {n_events} events")
        stop = n_events if max_events is None else min(n_events, first_event + max_events)
        return [infer_event_topology(data, event) for event in range(first_event, stop)]
