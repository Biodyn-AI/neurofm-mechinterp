"""Tests for shared spectral-analysis helpers."""

from __future__ import annotations

import unittest

import numpy as np

from neurofm_mechinterp.analysis.common import (
    flatten_windowed_time,
    mean_sensor_relative_bandpower,
    relative_bandpower,
)


class MeanSensorRelativeBandpowerTests(unittest.TestCase):
    def test_avoids_sensor_cancellation(self) -> None:
        sample_rate = 256
        time = np.arange(sample_rate) / sample_rate
        sine = np.sin(2.0 * np.pi * 10.0 * time)
        anti_phase = -sine
        data = np.stack([sine, anti_phase], axis=0)[None, ...]

        collapsed = relative_bandpower(data.mean(axis=1), sample_rate=sample_rate)
        sensor_aware = mean_sensor_relative_bandpower(data, sample_rate=sample_rate, sensor_axis=1)

        self.assertLess(float(collapsed["alpha"][0]), 1e-3)
        self.assertGreater(float(sensor_aware["alpha"][0]), 0.9)

    def test_flatten_windowed_time_preserves_order(self) -> None:
        data = np.arange(24).reshape(1, 2, 3, 4)

        flattened = flatten_windowed_time(data)

        self.assertEqual(flattened.shape, (1, 2, 12))
        self.assertTrue(np.array_equal(flattened[0, 0], np.arange(12)))


if __name__ == "__main__":
    unittest.main()
