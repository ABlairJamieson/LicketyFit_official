import numpy as np

from scripts.plot_delayed_times import expected_bin_counts, fit_delay_model


def test_truncated_exponential_plus_background_fit():
    rng = np.random.default_rng(20260929)
    start, stop, true_tau = 500.0, 6100.0, 2200.0
    length = stop - start
    uniform_draws = rng.random(5000)
    signal = start - true_tau * np.log1p(-uniform_draws * (1 - np.exp(-length / true_tau)))
    background = rng.uniform(start, stop, 700)
    fit = fit_delay_model(np.concatenate((signal, background)), start, stop)
    assert abs(fit["tau_ns"] - true_tau) < 300
    assert fit["low_ns"] < fit["tau_ns"] < fit["high_ns"]
    expected = expected_bin_counts(np.linspace(start, stop, 29), fit, start, stop)
    assert np.isclose(expected.sum(), fit["n_fit"])
