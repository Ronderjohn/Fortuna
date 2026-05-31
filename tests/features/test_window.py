"""Tests for ``features/window.py``."""

from __future__ import annotations

import numpy as np
import pytest

from fortuna.features.window import WindowBuffer


def test_push_rolls_oldest_off():
    buf = WindowBuffer(n_bars=3, n_features=2)
    a = buf.push(np.array([1, 2], dtype=np.float32))
    b = buf.push(np.array([3, 4], dtype=np.float32))
    c = buf.push(np.array([5, 6], dtype=np.float32))
    d = buf.push(np.array([7, 8], dtype=np.float32))

    assert a.shape == (6,)
    # After 4 pushes the buffer holds the last 3 rows: [3,4], [5,6], [7,8].
    np.testing.assert_allclose(d, np.array([3, 4, 5, 6, 7, 8], dtype=np.float32))


def test_reset_zeroes_buffer():
    buf = WindowBuffer(n_bars=4, n_features=3)
    buf.push(np.array([1, 2, 3], dtype=np.float32))
    buf.push(np.array([4, 5, 6], dtype=np.float32))
    buf.reset()
    np.testing.assert_allclose(buf.snapshot(), np.zeros((4, 3), dtype=np.float32))


def test_push_validates_shape():
    buf = WindowBuffer(n_bars=2, n_features=2)
    with pytest.raises(ValueError):
        buf.push(np.array([1, 2, 3], dtype=np.float32))


def test_shape_property():
    buf = WindowBuffer(n_bars=5, n_features=7)
    assert buf.shape == (5, 7)
