"""Settings from environment variables (plan section 2.3), read once at app creation."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

API_VERSION = "1.1.1"
ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
MB = 1024 * 1024


def _env(name, default):
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def _flag(name):
    return _env(name, "0").lower() in ("1", "true", "yes", "on")


def _origins(text):
    return tuple(item.strip().rstrip("/") for item in text.split(",") if item.strip() and item.strip() != "*")


@dataclass(frozen=True)
class Settings:
    """All tunables. Sizes in bytes, times in seconds."""

    # Model (the Docker build sets these from its build args).
    model_path: Path = ROOT / "model" / "best.onnx"
    model_repo: str = "hectorvh/gpradar-hyperbola-yolo"
    model_file: str = "best.onnx"
    model_revision: str = "unknown"

    # ALLOWED_ORIGINS: exact origins, comma-separated; "*" is ignored.
    allowed_origins: tuple[str, ...] = ("http://localhost:5173",)

    max_image_bytes: int = 10 * MB  # MAX_IMAGE_MB
    max_raw_bytes: int = 25 * MB  # MAX_RAW_MB
    max_traces: int = 4000  # MAX_TRACES
    max_samples: int = 4096  # MAX_SAMPLES
    max_pixels: int = 12_000_000  # MAX_PIXELS (Pillow decompression-bomb guard)
    # MAX_ARRAY_ELEMENTS: samples x traces through the float chain. Not in the plan's
    # table: a 4096 x 4000 record would need well over 512 MB in the chain
    # (measured: 4M values peak ~450 MB, 2M values ~290 MB).
    max_array_elements: int = 2_000_000
    max_metadata_chars: int = 2_000

    run_timeout_s: float = 25.0  # RUN_TIMEOUT_S
    max_concurrent_runs: int = 1  # MAX_CONCURRENT_RUNS
    queue_wait_s: float = 2.0  # QUEUE_WAIT_S: wait for a free slot, then 503
    rate_limit_per_min: int = 10  # RATE_LIMIT_PER_MIN
    # CLIENT_IP_HEADER: a header that holds exactly one client address, set by
    # the edge proxy (Cloudflare on Render: CF-Connecting-IP). Empty disables it.
    client_ip_header: str = "cf-connecting-ip"
    # TRUSTED_PROXY_HOPS: fallback when that header is absent or invalid. Proxies
    # in front of the app that append to X-Forwarded-For; the client is the entry
    # this many places from the right.
    trusted_proxy_hops: int = 1
    # LOG_CLIENT_IP_SOURCES=1: log the client IP diagnostics line on every /run
    # (normally once per process). For checking a deployment; leave it off.
    log_client_ip_sources: bool = False
    min_conf: float = 0.05  # MIN_CONF
    max_windows: int = 12  # MAX_WINDOWS: tiling cap (Render 12, Cloud Run 40)
    onnx_threads: int = 1  # ONNX_THREADS (v1.2: 2)
    log_level: str = "info"  # LOG_LEVEL

    samples_dir: Path = SAMPLES_DIR

    @property
    def max_upload_bytes(self):
        return max(self.max_image_bytes, self.max_raw_bytes)

    def with_overrides(self, **kwargs):
        return replace(self, **kwargs)


def load_settings():
    """Build Settings from the environment."""

    base = Settings()
    return Settings(
        model_path=Path(_env("MODEL_PATH", str(base.model_path))),
        model_repo=_env("MODEL_REPO", base.model_repo),
        model_file=_env("MODEL_FILE", base.model_file),
        model_revision=_env("MODEL_REVISION", base.model_revision),
        allowed_origins=_origins(_env("ALLOWED_ORIGINS", ",".join(base.allowed_origins))),
        max_image_bytes=int(float(_env("MAX_IMAGE_MB", 10)) * MB),
        max_raw_bytes=int(float(_env("MAX_RAW_MB", 25)) * MB),
        max_traces=int(_env("MAX_TRACES", base.max_traces)),
        max_samples=int(_env("MAX_SAMPLES", base.max_samples)),
        max_pixels=int(_env("MAX_PIXELS", base.max_pixels)),
        max_array_elements=int(_env("MAX_ARRAY_ELEMENTS", base.max_array_elements)),
        run_timeout_s=float(_env("RUN_TIMEOUT_S", base.run_timeout_s)),
        max_concurrent_runs=int(_env("MAX_CONCURRENT_RUNS", base.max_concurrent_runs)),
        queue_wait_s=float(_env("QUEUE_WAIT_S", base.queue_wait_s)),
        rate_limit_per_min=int(_env("RATE_LIMIT_PER_MIN", base.rate_limit_per_min)),
        # Unlike the other settings, an empty CLIENT_IP_HEADER is meaningful (off).
        client_ip_header=os.environ.get("CLIENT_IP_HEADER", base.client_ip_header).strip().lower(),
        trusted_proxy_hops=int(_env("TRUSTED_PROXY_HOPS", base.trusted_proxy_hops)),
        log_client_ip_sources=_flag("LOG_CLIENT_IP_SOURCES"),
        min_conf=float(_env("MIN_CONF", base.min_conf)),
        max_windows=int(_env("MAX_WINDOWS", base.max_windows)),
        onnx_threads=int(_env("ONNX_THREADS", base.onnx_threads)),
        log_level=_env("LOG_LEVEL", base.log_level).lower(),
    )
