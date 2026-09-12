import numpy as np

from LicketyFit.TruthTopology import infer_event_topology


def object_events(*events):
    result = np.empty(len(events), dtype=object)
    result[:] = events
    return result


def test_interacting_pion_with_pi0_and_light_fractions():
    data = {
        "pid": np.array([-211]),
        "energy": np.array([661.0]),
        "event_id": np.array([7]),
        "track_id": object_events(np.array([1, 2, 3, 4])),
        "track_pid": object_events(np.array([-211, 111, 11, -11])),
        "track_parent": object_events(np.array([0, -211, 999, 999])),
        "true_hit_parent": object_events(np.array([1, 1, 3, 4])),
    }

    result = infer_event_topology(data, 0)

    assert result["topology"] == "interacting_pion_with_pi0"
    assert result["has_pion_parented_pi0"]
    assert result["true_hit_fraction_charged_pion"] == 0.5
    assert result["true_hit_fraction_em"] == 0.5


def test_clean_pion_candidate_without_interaction_signature():
    data = {
        "pid": np.array([211]),
        "energy": np.array([300.0]),
        "track_id": object_events(np.array([1, 2])),
        "track_pid": object_events(np.array([211, 11])),
        "track_parent": object_events(np.array([0, 999])),
    }

    result = infer_event_topology(data, 0)

    assert result["topology"] == "clean_pion_candidate"
    assert np.isnan(result["true_hit_fraction_em"])


def test_ambiguous_track_ids_are_not_misattributed():
    data = {
        "pid": np.array([-211]),
        "energy": np.array([200.0]),
        "track_id": object_events(np.array([0, 0])),
        "track_pid": object_events(np.array([-211, 0])),
        "track_parent": object_events(np.array([0, 0])),
        "true_hit_parent": object_events(np.array([0, 0])),
    }

    result = infer_event_topology(data, 0)

    assert result["true_hit_fraction_ambiguous_track_id"] == 1.0
    assert result["true_hit_fraction_charged_pion"] == 0.0
