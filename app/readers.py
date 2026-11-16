"""Input files: type by magic bytes, images to 8-bit grey, gprMax to GprScan.

Nothing here trusts a file name or a client content type. Every reader
checks sizes before it decodes, so a small compressed file cannot expand
into a large array.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageOps

from .errors import ApiError
from .gpr.loader import GprScan, GprSource, load_gprmax

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
ZIP_MAGIC = b"PK\x03\x04"  # .npz (bundled samples only, never accepted as an upload)
# HDF5 allows a user block, so the signature may sit at 0, 512, 1024, 2048, ...
HDF5_OFFSETS = (0, 512, 1024, 2048)
SNIFF_BYTES = 2048 + len(HDF5_MAGIC)

# Smallest record the chain can filter (sosfiltfilt pads ~27 samples each side).
MIN_SAMPLES = 64
MIN_PIXELS_SIDE = 8


def configure_pillow(max_pixels):
    """Make Pillow refuse decompression bombs above ``max_pixels``."""

    Image.MAX_IMAGE_PIXELS = int(max_pixels)
    warnings.simplefilter("error", Image.DecompressionBombWarning)


def sniff(head):
    """'png', 'jpeg', 'gprmax' or 'npz' from the first bytes; None if unknown."""

    if head.startswith(PNG_MAGIC):
        return "png"
    if head.startswith(JPEG_MAGIC):
        return "jpeg"
    for offset in HDF5_OFFSETS:
        if head[offset : offset + len(HDF5_MAGIC)] == HDF5_MAGIC:
            return "gprmax"
    if head.startswith(ZIP_MAGIC):
        return "npz"
    return None


WARN_COLOUR = "colour image converted to greyscale; colormaps may not map to amplitude"
COLOUR_MODES = {"RGB", "RGBA", "P", "PA", "CMYK", "YCbCr", "LAB", "HSV", "RGBX", "RGBa"}


# Large JPEGs are decoded at reduced size (JPEG DCT scaling, 1/2 to 1/8) when
# the model would downscale them anyway: the letterbox makes the long side
# 224 px (tiled profiles: each window's height), so a decode that keeps at least
# DRAFT_KEEP_PX on that side loses nothing the model can see. Only reductions
# of 2x or more are requested, so small JPEGs decode exactly as in GPRadar 06f.
DRAFT_KEEP_PX = 448
TILE_MAX_ASPECT = 2.0  # same rule as detect.tile_windows
EXIF_ORIENTATION = 0x0112


@dataclass
class ImageInput:
    grey: np.ndarray  # uint8, (height, width), what the model gets
    warnings: list = field(default_factory=list)
    # (height, width) of the image as uploaded (after EXIF rotation). Differs from
    # grey.shape when a large JPEG was decoded at reduced size.
    source_shape: tuple | None = None

    def __post_init__(self):
        if self.source_shape is None:
            self.source_shape = tuple(self.grey.shape)


@dataclass
class ScanInput:
    scan: GprScan


def _is_colour(image):
    """True if the image has real colour (not grey stored as RGB), on a thumbnail."""

    if image.mode not in COLOUR_MODES:
        return False
    factor = max(1, max(image.size) // 256)
    thumb = image.reduce(factor) if image.mode in ("RGB", "RGBA", "RGBX", "CMYK", "YCbCr") else image
    rgb = np.asarray(thumb.convert("RGB"), dtype=np.int16)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    return bool(np.mean(spread > 8) > 0.01)


def draft_size(width, height, swapped=False):
    """Requested (width, height) for ``Image.draft``, or None if no 2x reduction helps.

    ``width``/``height`` are the stored JPEG size; ``swapped`` means the EXIF
    orientation turns it by 90 degrees, so the displayed width is ``height``.
    """

    shown_w, shown_h = (height, width) if swapped else (width, height)
    side = shown_h if shown_w > TILE_MAX_ASPECT * shown_h else max(shown_w, shown_h)
    scale = DRAFT_KEEP_PX / side
    if scale > 0.5:
        return None
    return (max(1, int(np.ceil(width * scale))), max(1, int(np.ceil(height * scale))))


def _jpeg_is_colour(path):
    """Colour check on a 1/8-scale RGB decode (cheap), for JPEGs decoded as L."""

    with Image.open(path) as image:
        if image.mode not in COLOUR_MODES:
            return False
        image.draft("RGB", (max(1, image.size[0] // 8), max(1, image.size[1] // 8)))
        return _is_colour(image)


def read_image(path, file_type, max_pixels):
    """Decode a PNG/JPEG to 8-bit greyscale (EXIF orientation, then ``convert("L")`` as 06f).

    Large JPEGs use ``Image.draft("L", ...)`` before decoding (see DRAFT_KEEP_PX).
    """

    expected = {"png": "PNG", "jpeg": "JPEG"}[file_type]
    try:
        with Image.open(path) as image:
            if image.format != expected:
                raise ApiError("unsupported_type", "the file content is not a valid PNG or JPEG image")
            width, height = image.size
            if width * height > max_pixels:
                raise ApiError("too_large", f"image has {width * height} pixels; the limit is {max_pixels}")
            if min(width, height) < MIN_PIXELS_SIDE:
                raise ApiError("bad_request", f"image must be at least {MIN_PIXELS_SIDE} px on each side")
            swapped = image.getexif().get(EXIF_ORIENTATION) in (5, 6, 7, 8)
            source_shape = (width, height) if swapped else (height, width)
            target = draft_size(width, height, swapped) if file_type == "jpeg" else None
            warnings_out = []
            if target is not None:
                if _jpeg_is_colour(path):
                    warnings_out.append(WARN_COLOUR)
                image.draft("L", target)
            image.load()
            oriented = ImageOps.exif_transpose(image) or image
            if target is None and _is_colour(oriented):
                warnings_out.append(WARN_COLOUR)
            grey_image = oriented.convert("L")
            if oriented is not image:
                oriented.close()
            image.close()  # free the colour pixels before the array copy
            grey = np.asarray(grey_image, dtype=np.uint8)
            grey_image.close()
    except ApiError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ApiError("too_large", f"image exceeds the {max_pixels} pixel limit") from None
    except (OSError, ValueError, SyntaxError):
        raise ApiError("bad_request", "the image could not be decoded") from None
    if grey.shape != source_shape:
        factor = source_shape[1] / grey.shape[1]
        warnings_out.append(
            f"Large JPEG decoded at 1/{factor:.0f} size ({grey.shape[1]} x {grey.shape[0]} px) for memory; "
            "apex trace/sample refer to the original pixels."
        )
    return ImageInput(grey=grey, warnings=warnings_out, source_shape=source_shape)


def _check_dims(n_samples, n_traces, settings):
    if n_traces > settings.max_traces or n_samples > settings.max_samples:
        raise ApiError(
            "too_large",
            f"record is {n_samples} samples x {n_traces} traces; the limits are "
            f"{settings.max_samples} samples and {settings.max_traces} traces",
        )
    if n_samples * n_traces > settings.max_array_elements:
        raise ApiError(
            "too_large",
            f"record has {n_samples * n_traces} values; the limit is {settings.max_array_elements}",
        )


def _check_gprmax_shape(path, settings, component="Ez", receiver="rx1"):
    """Open the HDF5 file and check the record size before anything is loaded."""

    try:
        with h5py.File(path, "r") as output:
            dataset = output.get(f"rxs/{receiver}/{component}")
            if not isinstance(dataset, h5py.Dataset):
                raise ApiError("bad_request", f"not a gprMax output: no rxs/{receiver}/{component} dataset")
            shape = dataset.shape
    except ApiError:
        raise
    except (OSError, ValueError, KeyError):
        raise ApiError("bad_request", "the HDF5 file could not be read") from None
    if len(shape) not in (1, 2):
        raise ApiError("bad_request", "the gprMax record must be one trace or a time-by-trace array")
    _check_dims(int(shape[0]), int(shape[1]) if len(shape) == 2 else 1, settings)


def _check_scan(scan, settings):
    data = np.asarray(scan.data)
    if data.ndim != 2 or data.size == 0:
        raise ApiError("bad_request", "the record is empty")
    _check_dims(data.shape[0], data.shape[1], settings)
    if data.shape[0] < MIN_SAMPLES:
        raise ApiError("bad_request", f"the record needs at least {MIN_SAMPLES} samples per trace")
    if not np.all(np.isfinite(data)):
        raise ApiError("bad_request", "the record contains NaN or infinite values")


def read_gprmax(path, settings):
    """Load a merged gprMax B-scan (.out / .h5) with the vendored 02a loader."""

    _check_gprmax_shape(path, settings)
    try:
        scan = load_gprmax(Path(path))
    except (OSError, ValueError, KeyError):
        # The loader's messages contain the server-side path: do not forward them.
        raise ApiError("bad_request", "not a readable gprMax output (needs a dt attribute and rxs/rx1/Ez)") from None
    _check_scan(scan, settings)
    return ScanInput(scan=scan)


def read_scan_npz(path, settings):
    """Load a bundled .npz scene as the same GprScan a gprMax upload gives.

    The archive holds ``data`` (samples x traces), ``dt`` (s), ``trace_spacing``
    (m) and ``frequency`` (Hz), as written by GPRadar 07a. Anything else in it
    (scene JSON, a pre-rendered image) is ignored, so the sample runs through
    exactly the same processing as an uploaded file.
    """

    with np.load(path, allow_pickle=False) as archive:
        _check_dims(*archive["data"].shape, settings)
        data = np.asarray(archive["data"], dtype=np.float64)
        dt = float(archive["dt"])
        spacing = float(archive["trace_spacing"]) if "trace_spacing" in archive.files else None
        frequency = float(archive["frequency"]) if "frequency" in archive.files else None
    n_traces = data.shape[1]
    positions = np.zeros((n_traces, 3))
    if spacing:
        positions[:, 0] = spacing * np.arange(n_traces)
    source = GprSource(
        name="",
        kind=None,
        polarisation=None,
        waveform_id=None,
        waveform_type=None,
        amplitude=None,
        frequency=frequency,
        position=positions[0],
        positions=positions,
    )
    scan = GprScan(
        data=data,
        dt=dt,
        trace_spacing=spacing,
        frequency=frequency,
        source=source,
        component="Ez",
        positions=positions,
    )
    _check_scan(scan, settings)
    return ScanInput(scan=scan)


def read_path(path, settings, allow_npz=False):
    """Detect the type of the file at ``path`` and read it.

    Returns (input, kind) with kind 'image' or 'gprmax'.
    """

    with Path(path).open("rb") as handle:
        head = handle.read(SNIFF_BYTES)
    file_type = sniff(head)
    if file_type in ("png", "jpeg"):
        if Path(path).stat().st_size > settings.max_image_bytes:
            raise ApiError("too_large", f"images are limited to {settings.max_image_bytes} bytes")
        return read_image(path, file_type, settings.max_pixels), "image"
    if file_type == "gprmax":
        if Path(path).stat().st_size > settings.max_raw_bytes:
            raise ApiError("too_large", f"gprMax files are limited to {settings.max_raw_bytes} bytes")
        return read_gprmax(path, settings), "gprmax"
    if file_type == "npz" and allow_npz:
        return read_scan_npz(path, settings), "gprmax"
    raise ApiError("unsupported_type", "accepted files are PNG, JPEG and gprMax HDF5 output (.out / .h5)")


def load_manifest(samples_dir):
    """The sample list from manifest.json, keyed by id."""

    manifest = json.loads((Path(samples_dir) / "manifest.json").read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in manifest["samples"]}
