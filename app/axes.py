"""Metadata resolution (user > file > default) and the physical axes.

Precedence (plan 2.6): a value the user sends wins, so a wrong header can be
corrected; otherwise the value read from the file; otherwise the default.
When a user value replaces a different file value, a warning says so. Only
the velocity and the centre frequency have defaults. Without a time window or
trace spacing the axes stay in samples and traces, and no depth is given.
"""

from __future__ import annotations

import json
import math

from .errors import ApiError

FIELDS = ("frequency_mhz", "time_window_ns", "trace_spacing_m", "velocity_m_ns")
DEFAULTS = {
    "frequency_mhz": 400.0,  # the simulated training scenes (400 MHz)
    "time_window_ns": None,
    "trace_spacing_m": None,
    "velocity_m_ns": 0.10,  # typical damp soil; depth is approximate
}
# Accepted ranges (inclusive) for user values.
RANGES = {
    "frequency_mhz": (10.0, 3000.0),
    "time_window_ns": (1.0, 5000.0),
    "trace_spacing_m": (0.001, 5.0),
    "velocity_m_ns": (0.01, 0.3),
}


def parse_metadata(text, max_chars=2000):
    """Parse the ``metadata`` form field. Raises ApiError('invalid_metadata')."""

    if text is None or text.strip() == "":
        return {}
    if len(text) > max_chars:
        raise ApiError("invalid_metadata", f"metadata is longer than {max_chars} characters")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        raise ApiError("invalid_metadata", "metadata must be a JSON object") from None
    if not isinstance(payload, dict):
        raise ApiError("invalid_metadata", "metadata must be a JSON object")
    unknown = sorted(set(payload) - set(FIELDS))
    if unknown:
        raise ApiError("invalid_metadata", f"unknown metadata field(s): {', '.join(unknown)[:200]}")
    out = {}
    for name, value in payload.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ApiError("invalid_metadata", f"{name} must be a number")
        value = float(value)
        low, high = RANGES[name]
        if not math.isfinite(value) or not low <= value <= high:
            raise ApiError("invalid_metadata", f"{name} must be between {low:g} and {high:g}")
        out[name] = value
    return out


def resolve(file_values, user_values, warnings):
    """Merge into {field: {"value", "source"}} and append notes to ``warnings``."""

    resolved = {}
    for name in FIELDS:
        file_value = file_values.get(name)
        user_value = user_values.get(name)
        if user_value is not None:
            resolved[name] = {"value": float(user_value), "source": "user"}
            if file_value is not None and not math.isclose(user_value, file_value, rel_tol=1e-3):
                warnings.append(f"{name}: the value sent ({user_value:g}) replaced the file's ({file_value:g}).")
        elif file_value is not None:
            resolved[name] = {"value": float(file_value), "source": "file"}
        else:
            resolved[name] = {"value": DEFAULTS[name], "source": "default"}
    return resolved


def value(metadata, name):
    return metadata[name]["value"]


def build_axes(metadata, n_traces, n_samples):
    """Axis extents of the image edges (for drawing with ``box_norm``)."""

    spacing = value(metadata, "trace_spacing_m")
    window = value(metadata, "time_window_ns")
    velocity = value(metadata, "velocity_m_ns")
    x = (
        {"unit": "m", "min": 0.0, "max": _r(n_traces * spacing)}
        if spacing
        else {"unit": "trace", "min": 0.0, "max": float(n_traces)}
    )
    y = (
        {"unit": "ns", "min": 0.0, "max": _r(window)}
        if window
        else {"unit": "sample", "min": 0.0, "max": float(n_samples)}
    )
    depth = (
        {"unit": "m", "min": 0.0, "max": _r(velocity * window / 2.0), "velocity_m_ns": velocity}
        if window
        else None
    )
    return {"x": x, "y": y, "depth": depth}


def apex(box_px, image_w, image_h, n_traces, n_samples, metadata):
    """Top centre of the box, in traces/samples and, when known, metres and ns.

    twt_ns = y_min / height x time_window_ns; distance_m = x_centre_trace x
    trace_spacing_m; depth_m = velocity x twt_ns / 2 (approximate).
    """

    x_centre = 0.5 * (box_px[0] + box_px[2])
    trace = x_centre / image_w * n_traces
    sample = box_px[1] / image_h * n_samples
    out = {"trace": _r(trace), "sample": _r(sample)}
    spacing = value(metadata, "trace_spacing_m")
    window = value(metadata, "time_window_ns")
    if spacing:
        out["distance_m"] = _r(trace * spacing)
    if window:
        twt = box_px[1] / image_h * window
        out["twt_ns"] = _r(twt)
        out["depth_m"] = _r(value(metadata, "velocity_m_ns") * twt / 2.0)
    return out


def _r(number, digits=4):
    return round(float(number), digits)
