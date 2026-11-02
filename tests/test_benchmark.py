import numpy as np
import pytest

from benchmark import bicubic_baseline, psnr


def test_psnr_identical_is_infinite():
    image = np.ones((8, 8, 3), dtype=np.float32)
    assert psnr(image, image) == float("inf")


def test_psnr_known_error():
    reference = np.zeros((8, 8, 3))
    prediction = np.ones((8, 8, 3)) * 0.1
    assert psnr(reference, prediction) == pytest.approx(20)


def test_psnr_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        psnr(np.zeros((8, 8, 3)), np.zeros((7, 8, 3)))


def test_bicubic_baseline_preserves_shape():
    image = np.random.default_rng(1).random((81, 79, 3)).astype(np.float32)
    output = bicubic_baseline(image)
    assert output.shape == image.shape
    assert np.min(output) >= 0
    assert np.max(output) <= 1
