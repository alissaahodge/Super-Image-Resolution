import numpy as np
import pytest

from resolution import pixelate_image


@pytest.mark.parametrize("shape", [(80, 80, 3), (81, 79, 3), (1, 2, 3)])
def test_pixelation_preserves_dimensions(shape):
    image = np.random.default_rng(7).random(shape).astype(np.float32)
    output = pixelate_image(image)
    assert output.shape == shape
    assert np.isfinite(output).all()


def test_invalid_scale():
    with pytest.raises(ValueError):
        pixelate_image(np.zeros((80, 80, 3)), 0)


def test_invalid_image():
    with pytest.raises(ValueError):
        pixelate_image(np.zeros((80, 80)))
