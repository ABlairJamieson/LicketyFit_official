import io

import numpy as np

from LicketyFit.pickle_compat import load_numpy_pickle


def test_numpy_2_private_core_module_is_remapped():
    # Protocol-0 GLOBAL reference using the private module name emitted by
    # NumPy 2. This remains constructible when the test itself runs on NumPy 1.
    payload = b"cnumpy._core.multiarray\nscalar\n."
    loaded = load_numpy_pickle(io.BytesIO(payload))
    assert loaded is np.core.multiarray.scalar
