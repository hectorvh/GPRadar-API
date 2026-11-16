"""From a read input to the image the model sees, and the full /run job (plan 2.5).

Raw record (gprMax upload or bundled .npz), mode "full": the GPRadar chain
(time zero, dewow, mean-trace removal, band-pass, AGC; windows of three
periods of the centre frequency) on the whole record, robust 8-bit scaling
of the whole record, then a Pillow bilinear resize of that 8-bit image to
224 x 224, whatever the record's shape. This is exactly
GPRadar 07a ``processed_image`` (the rule that made golden.json), in the same
order: process -> robust_to_uint8 -> Image.fromarray(..., "L").resize((224,
224), BILINEAR). The training crops were also resized to 224 x 224 with
bilinear resampling (05a), so the model has only seen squeezed or stretched
aspect ratios like this. A gprMax record is therefore never tiled.

Raw record, mode "detector": robust 8-bit scaling only, with a warning.
Image, mode "detector": the 8-bit grey image as is (the default).
Image, mode "full": pixels mapped to amplitude (v - 128) / 128, dt from the
time window (default 50 ns), then dewow, background removal, band-pass and
AGC (no time zero), robust 8-bit scaling; allowed with a warning.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from . import axes
from .config import API_VERSION
from .detect import detect_tiled
from .errors import ApiError
from .gpr import preprocessing
from .gpr.chain import process_scene_array, processing_windows, robust_to_uint8
from .readers import ImageInput, read_path

RAW_IMAGE_SIZE = 224  # GPRadar 07a IMAGE_SIZE: raw records become 224 x 224
IMAGE_DEFAULT_WINDOW_NS = 50.0  # the Mendeley 400 MHz setting
DISPLAY_MAX_WIDTH = 2000

WARN_IMAGE_FULL = (
    "This looks like an image. Preprocessing an already processed image can degrade detection."
)
WARN_RAW_DETECTOR = "raw data without preprocessing; detections will be poor"
WARN_DEPTH = "Depth uses a constant velocity and is approximate."


@dataclass
class Prepared:
    model_image: np.ndarray
    raw_image: np.ndarray | None
    n_traces: int
    n_samples: int
    metadata: dict
    warnings: list = field(default_factory=list)


def chain_to_uint8(data, dt, frequency_hz):
    """GPRadar processing chain and robust 8-bit scaling; same shape as ``data``."""

    processed, _picks, windows = process_scene_array(np.asarray(data, dtype=np.float64), dt, frequency_hz)
    return robust_to_uint8(processed), windows


def image_chain_to_uint8(grey, dt, frequency_hz):
    """Full mode on an image: (v - 128) / 128, then the chain without time zero."""

    amplitude = (np.asarray(grey, dtype=np.float64) - 128.0) / 128.0
    windows = processing_windows(frequency_hz)
    dewowed = preprocessing.dewow(amplitude, dt, windows["dewow_ns"])
    cleaned = preprocessing.remove_background(dewowed)
    low, high = preprocessing.default_band(frequency_hz, 0.5 / dt)
    filtered = preprocessing.bandpass(cleaned, dt, low, high)
    gained = preprocessing.gain_agc(filtered, dt, windows["agc_ns"])
    return robust_to_uint8(gained)


def resize_to_model(grey):
    """8-bit record image -> 224 x 224, exactly as GPRadar 07a ``processed_image``."""

    image = Image.fromarray(grey, mode="L").resize(
        (RAW_IMAGE_SIZE, RAW_IMAGE_SIZE), resample=Image.Resampling.BILINEAR
    )
    return np.asarray(image, dtype=np.uint8)


def file_metadata(inp):
    """Metadata the file itself records."""

    if isinstance(inp, ImageInput):
        return {}
    scan = inp.scan
    out = {"time_window_ns": scan.data.shape[0] * scan.dt * 1e9}
    if scan.trace_spacing:
        out["trace_spacing_m"] = scan.trace_spacing
    if scan.frequency:
        out["frequency_mhz"] = scan.frequency / 1e6
    return out


def _check_band(frequency_hz, dt, source):
    nyquist = 0.5 / dt
    if not 0.5 * frequency_hz < nyquist:
        code = "bad_request" if source == "file" else "invalid_metadata"
        raise ApiError(
            code,
            f"centre frequency {frequency_hz / 1e6:g} MHz is too high for the sampling "
            f"(Nyquist {nyquist / 1e6:g} MHz)",
        )


def prepare(inp, mode, user_meta, max_elements):
    """Build the model image (and the raw image to show) for one input."""

    warnings = list(getattr(inp, "warnings", []))
    metadata = axes.resolve(file_metadata(inp), user_meta, warnings)
    frequency_hz = axes.value(metadata, "frequency_mhz") * 1e6
    freq_source = metadata["frequency_mhz"]["source"]

    if isinstance(inp, ImageInput):
        grey = inp.grey
        n_samples, n_traces = inp.source_shape
        if mode == "detector":
            model_image, raw = grey, None
        else:
            if grey.size > max_elements:
                raise ApiError(
                    "too_large",
                    f"full mode is limited to {max_elements} pixels; use detector mode for this image",
                )
            window = axes.value(metadata, "time_window_ns") or IMAGE_DEFAULT_WINDOW_NS
            dt = window * 1e-9 / grey.shape[0]  # decoded height (a large JPEG may be reduced)
            _check_band(frequency_hz, dt, freq_source)
            model_image = image_chain_to_uint8(grey, dt, frequency_hz)
            raw = grey
            warnings.append(WARN_IMAGE_FULL)
    else:
        scan = inp.scan
        n_samples, n_traces = scan.data.shape
        if mode == "full":
            _check_band(frequency_hz, scan.dt, freq_source)
            processed, _windows = chain_to_uint8(scan.data, scan.dt, frequency_hz)
            model_image = resize_to_model(processed)
            del processed
            raw = resize_to_model(robust_to_uint8(scan.data))
        else:
            model_image = resize_to_model(robust_to_uint8(scan.data))
            raw = None
            warnings.append(WARN_RAW_DETECTOR)

    if axes.value(metadata, "time_window_ns"):
        warnings.append(WARN_DEPTH)
    return Prepared(
        model_image=np.ascontiguousarray(model_image, dtype=np.uint8),
        raw_image=raw,
        n_traces=int(n_traces),
        n_samples=int(n_samples),
        metadata=metadata,
        warnings=warnings,
    )


def png_base64(grey):
    """PNG for display; images wider than 2000 px are downscaled (boxes use box_norm)."""

    grey = np.ascontiguousarray(grey, dtype=np.uint8)
    height, width = grey.shape
    if width > DISPLAY_MAX_WIDTH:
        new_h = max(1, round(height * DISPLAY_MAX_WIDTH / width))
        grey = cv2.resize(grey, (DISPLAY_MAX_WIDTH, new_h), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".png", grey)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def detections_payload(boxes, scores, prepared):
    height, width = prepared.model_image.shape
    out = []
    for index, (box, score) in enumerate(zip(boxes, scores), start=1):
        x0, y0, x1, y1 = (float(v) for v in box)
        out.append(
            {
                "id": index,
                "confidence": round(float(score), 4),
                "box_px": {"x_min": round(x0, 2), "y_min": round(y0, 2), "x_max": round(x1, 2), "y_max": round(y1, 2)},
                "box_norm": {
                    "x_min": round(x0 / width, 5),
                    "y_min": round(y0 / height, 5),
                    "x_max": round(x1 / width, 5),
                    "y_max": round(y1 / height, 5),
                },
                "apex": axes.apex(
                    (x0, y0, x1, y1), width, height, prepared.n_traces, prepared.n_samples, prepared.metadata
                ),
            }
        )
    return out


def size_class(n_values):
    """Coarse size label for logs (never the exact size or name)."""

    if n_values <= 250_000:
        return "small"
    if n_values <= 1_000_000:
        return "medium"
    return "large"


def run_job(path, settings, model, mode, user_meta, filename, run_id, default_mode=None, allow_npz=False):
    """Read, prepare, detect and build the 200 response for one input file."""

    start = time.perf_counter()
    inp, kind = read_path(path, settings, allow_npz=allow_npz)
    t_read = time.perf_counter()

    mode = mode or default_mode or ("detector" if kind == "image" else "full")
    prepared = prepare(inp, mode, user_meta, settings.max_array_elements)
    del inp
    t_prep = time.perf_counter()

    boxes, scores, n_windows, cut_at = detect_tiled(
        model.session, prepared.model_image, min_conf=settings.min_conf, max_windows=settings.max_windows
    )
    t_inf = time.perf_counter()
    height, width = prepared.model_image.shape
    if n_windows > 1:
        prepared.warnings.append(f"Long profile: detected in {n_windows} overlapping windows and merged.")
    if cut_at is not None:
        prepared.warnings.append(
            f"Profile cut: only the first {cut_at} of {width} px were analysed "
            f"(limit {settings.max_windows} windows)."
        )

    images = {"processed_png": png_base64(prepared.model_image)}
    if prepared.raw_image is not None:
        images = {"raw_png": png_base64(prepared.raw_image), **images}

    result = {
        "run_id": run_id,
        "api_version": API_VERSION,
        "mode": mode,
        "model": {"repo": model.repo, "revision": model.revision},
        "input": {
            "kind": kind,
            "filename": filename,
            "width_px": int(width),
            "height_px": int(height),
            "n_traces": prepared.n_traces,
            "n_samples": prepared.n_samples,
        },
        "metadata": prepared.metadata,
        "images": images,
        "axes": axes.build_axes(prepared.metadata, prepared.n_traces, prepared.n_samples),
        "detections": detections_payload(boxes, scores, prepared),
        "warnings": prepared.warnings,
    }
    end = time.perf_counter()
    result["timings_ms"] = {
        "read": round(1000 * (t_read - start), 1),
        "preprocess": round(1000 * (t_prep - t_read), 1),
        "inference": round(1000 * (t_inf - t_prep), 1),
        "total": round(1000 * (end - start), 1),
    }
    result["_size_class"] = size_class(prepared.n_traces * prepared.n_samples)
    return result
