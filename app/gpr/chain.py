# Vendored from hectorvh/GPRadar at commit 1b052df:
#   PROCESSING_PERIODS, processing_windows_ns   from src/01c_scene_generator.py
#   FREQUENCY_HZ, processing_windows, process_scene_array, robust_to_uint8
#                                               from src/05a_render_synthetic.py
# Only these names are copied (05a imports matplotlib and the scene generator).
# `scenes_mod.processing_windows_ns` and `processing.` refer to the copies below.
# Licence: AGPL-3.0 (same as GPRadar). Do not edit here; change GPRadar and re-vendor.
"""Processing chain used to render the training images (time zero, dewow,
mean-trace removal, band-pass, AGC) and the robust 8-bit scaling."""

from types import SimpleNamespace

import numpy as np

from . import preprocessing as processing

# ---- from src/01c_scene_generator.py

# Three periods. The 2 ns dewow and AGC windows on the 1.5 GHz pipe are three cycles.
PROCESSING_PERIODS = 3.0


def processing_windows_ns(frequency_hz):
    """Dewow and AGC windows, in nanoseconds: three periods of the centre frequency."""

    period_ns = 1e9 / float(frequency_hz)
    window_ns = PROCESSING_PERIODS * period_ns
    return window_ns, window_ns


# 05a calls these through its `scenes_mod` alias.
scenes_mod = SimpleNamespace(processing_windows_ns=processing_windows_ns)

# ---- from src/05a_render_synthetic.py

FREQUENCY_HZ = 4.0e8


def processing_windows(frequency_hz=FREQUENCY_HZ):
    """Dewow and AGC windows in ns, and the band-pass edges in Hz."""

    dewow_ns, agc_ns = scenes_mod.processing_windows_ns(frequency_hz)
    return {
        "dewow_ns": float(dewow_ns),
        "agc_ns": float(agc_ns),
        "period_ns": float(1e9 / frequency_hz),
        "band_low_hz": 0.5 * float(frequency_hz),
        "band_high_factor": 2.0,
        "frequency_hz": float(frequency_hz),
    }


def process_scene_array(data, dt, frequency_hz=FREQUENCY_HZ):
    """Time zero, dewow, mean-trace removal, band-pass, AGC. No reference sub."""

    windows = processing_windows(frequency_hz)
    shifted, picks = processing.time_zero(data, dt)
    dewowed = processing.dewow(shifted, dt, windows["dewow_ns"])
    cleaned = processing.remove_background(dewowed)
    nyquist = 0.5 / float(dt)
    low, high = processing.default_band(frequency_hz, nyquist)
    filtered = processing.bandpass(cleaned, dt, low, high)
    gained = processing.gain_agc(filtered, dt, windows["agc_ns"])
    windows = dict(windows)
    windows["band_low_hz"] = float(low)
    windows["band_high_hz"] = float(high)
    return gained, picks, windows


def robust_to_uint8(data):
    """Symmetric robust scale to 8-bit greyscale (0 at mid grey)."""

    values = np.asarray(data, dtype=np.float64)
    limit = float(np.percentile(np.abs(values), 99))
    if limit <= 0:
        limit = 1.0
    scaled = np.clip(values / limit, -1.0, 1.0)
    return np.round((scaled + 1.0) * 0.5 * 255.0).astype(np.uint8)
