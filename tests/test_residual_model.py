import numpy as np
import pytest


def test_residual_model_starts_as_identity():
    pytest.importorskip("tensorflow")
    from resolution import build_residual_model

    model = build_residual_model()
    image = np.random.default_rng(42).uniform(0, 1, (1, 80, 80, 3)).astype("float32")
    output = model.predict(image, verbose=0)
    assert output.shape == image.shape
    np.testing.assert_allclose(output, image, atol=1e-6)
