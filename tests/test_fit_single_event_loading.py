from types import SimpleNamespace

import numpy as np

from scripts.fit_single_event import (
    SingleEventConfig,
    _load_wcsim_raw_event,
    _prepare_wcsim_observables,
)


def test_compact_wcsim_load_requests_only_fit_fields(tmp_path):
    requested = {}
    payload = {
        "digi_hit_pmt": np.asarray([np.asarray([1, 2])], dtype=object),
        "digi_hit_charge": np.asarray([np.asarray([3.0, 4.0])], dtype=object),
        "digi_hit_time": np.asarray([np.asarray([5.0, 6.0])], dtype=object),
    }

    def read_sim_data(path, *, fields=None):
        requested["path"] = path
        requested["fields"] = tuple(fields)
        return {name: payload[name] for name in fields}

    driver = SimpleNamespace(
        INPUT_FILE=str(tmp_path / "compact.npz"),
        FIT_FIELDS=("digi_hit_pmt", "digi_hit_charge", "digi_hit_time"),
        read_sim_data=read_sim_data,
    )
    event = _load_wcsim_raw_event(driver, SingleEventConfig(event_index=0))

    assert requested["fields"] == driver.FIT_FIELDS
    assert event["digi_hit_pmt"].tolist() == [1, 2]
    assert event["digi_hit_charge"].tolist() == [3.0, 4.0]
    assert event["digi_hit_time"].tolist() == [5.0, 6.0]


def test_wcsim_default_does_not_apply_peak_window(monkeypatch):
    raw = {
        "digi_hit_pmt": np.asarray([1, 2]),
        "digi_hit_charge": np.asarray([1.0, 1.0]),
        "digi_hit_time": np.asarray([10.0, 1000.0]),
    }
    monkeypatch.setattr(
        "scripts.fit_single_event._load_wcsim_raw_event",
        lambda driver, cfg, event=None: raw,
    )
    monkeypatch.setattr(
        "scripts.fit_single_event._apply_wcsim_peak_window",
        lambda *arrays: (_ for _ in ()).throw(
            AssertionError("default WCSim path must not apply the legacy window")
        ),
    )

    class Driver:
        WCD = object()
        P_LOCATIONS = np.zeros((2, 3))
        DIRECTION_ZS = np.ones((2, 3))
        MPMT_SLOTS = np.asarray([0, 0])
        RING_KEEP_MASK = np.asarray([True, True])
        ALL_RING = [0]
        RING_MASK_MODE = "none"

        @staticmethod
        def sim_to_event(sim_data, wcd, n_mpmt_total=106, pe_scale=1.0):
            return object(), np.asarray([1, 2])

        @staticmethod
        def build_observables_from_event(event, pe_scale=1.0):
            return raw["digi_hit_charge"], raw["digi_hit_time"]

        @staticmethod
        def apply_ring_mask_to_observables(charge, time, mask, mode="none"):
            return charge, time

    output = _prepare_wcsim_observables(
        Driver(), SingleEventConfig(apply_peak_time_window=None)
    )
    assert output["obs_ts"].tolist() == [10.0, 1000.0]
