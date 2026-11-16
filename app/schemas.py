"""Pydantic models mirroring the API contract (section 1.3)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelInfo(Strict):
    repo: str
    file: str
    revision: str
    sha256: str


class Health(Strict):
    status: Literal["ok"]
    api_version: str
    model: ModelInfo


class Sample(Strict):
    id: str
    name: str
    kind: Literal["image", "gprmax"]
    default_mode: Literal["full", "detector"]
    citation: str
    description: str


class RunModel(Strict):
    repo: str
    revision: str


class RunInput(Strict):
    kind: Literal["image", "gprmax"]
    filename: str
    width_px: int
    height_px: int
    n_traces: int
    n_samples: int


class MetaField(Strict):
    value: float | None
    source: Literal["file", "user", "default"]


class Metadata(Strict):
    frequency_mhz: MetaField
    time_window_ns: MetaField
    trace_spacing_m: MetaField
    velocity_m_ns: MetaField


class Images(Strict):
    raw_png: str | None = None
    processed_png: str


class AxisX(Strict):
    unit: Literal["m", "trace"]
    min: float
    max: float


class AxisY(Strict):
    unit: Literal["ns", "sample"]
    min: float
    max: float


class AxisDepth(Strict):
    unit: Literal["m"]
    min: float
    max: float
    velocity_m_ns: float


class Axes(Strict):
    x: AxisX
    y: AxisY
    depth: AxisDepth | None


class Box(Strict):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class Apex(Strict):
    trace: float
    sample: float
    distance_m: float | None = None
    twt_ns: float | None = None
    depth_m: float | None = None


class Detection(Strict):
    id: int
    confidence: float = Field(ge=0, le=1)
    box_px: Box
    box_norm: Box
    apex: Apex


class Timings(Strict):
    read: float
    preprocess: float
    inference: float
    total: float


class RunResponse(Strict):
    run_id: str
    api_version: str
    mode: Literal["full", "detector"]
    model: RunModel
    input: RunInput
    metadata: Metadata
    images: Images
    axes: Axes
    detections: list[Detection]
    warnings: list[str]
    timings_ms: Timings


class ErrorBody(Strict):
    code: Literal[
        "bad_request",
        "not_found",
        "method_not_allowed",
        "too_large",
        "unsupported_type",
        "invalid_metadata",
        "rate_limited",
        "internal_error",
        "busy",
        "timeout",
    ]
    message: str


class ErrorResponse(Strict):
    error: ErrorBody
