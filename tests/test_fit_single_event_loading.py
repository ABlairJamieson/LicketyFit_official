from types import SimpleNamespace

import numpy as np

from scripts.fit_single_event import SingleEventConfig, _load_wcsim_raw_event


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
