"""render.yaml must not raise a resource limit above its default in app/config.py.

Render uses the values in render.yaml, so a larger value there silently undoes
a limit chosen for the 512 MB instance (MAX_PIXELS was 25M there while the
default was lowered to 12M). The values are parsed with load_settings(), the
same code the service uses, and compared with Settings() field by field.
"""

from pathlib import Path

import pytest
import yaml

from app.config import ROOT, Settings, load_settings

RENDER_YAML = ROOT / "render.yaml"

# Environment variable -> Settings field. A larger value means more memory,
# CPU or trust, so render.yaml may only lower these.
LIMITS = {
    "MAX_IMAGE_MB": "max_image_bytes",
    "MAX_RAW_MB": "max_raw_bytes",
    "MAX_TRACES": "max_traces",
    "MAX_SAMPLES": "max_samples",
    "MAX_PIXELS": "max_pixels",
    "MAX_ARRAY_ELEMENTS": "max_array_elements",
    "RUN_TIMEOUT_S": "run_timeout_s",
    "MAX_CONCURRENT_RUNS": "max_concurrent_runs",
    "QUEUE_WAIT_S": "queue_wait_s",
    "RATE_LIMIT_PER_MIN": "rate_limit_per_min",
    "TRUSTED_PROXY_HOPS": "trusted_proxy_hops",
    "MAX_WINDOWS": "max_windows",
    "ONNX_THREADS": "onnx_threads",
}
# Numeric settings that are not resource limits (MIN_CONF is a score threshold).
NOT_LIMITS = {"MIN_CONF"}
# These two may exceed their defaults only while MAX_ARRAY_ELEMENTS (the real
# memory guard on samples x traces) is set and not above its default.
BOUNDED_BY_ARRAY_ELEMENTS = {"MAX_TRACES", "MAX_SAMPLES"}


def render_service():
    blueprint = yaml.safe_load(Path(RENDER_YAML).read_text(encoding="utf-8"))
    (service,) = [s for s in blueprint["services"] if s.get("name") == "gpradar-api"]
    return service


def render_env():
    return {item["key"]: str(item["value"]) for item in render_service()["envVars"] if "value" in item}


def test_service_is_free_docker_in_frankfurt():
    service = render_service()
    assert service["plan"] == "free"
    assert service["region"] == "frankfurt"
    assert service["runtime"] == "docker"
    assert service["healthCheckPath"] == "/health"


def test_client_ip_comes_from_cloudflare_header():
    # On Render the right-most X-Forwarded-For entry is a Cloudflare edge, so the
    # rate limit needs the header Cloudflare sets to the real client.
    assert render_env().get("CLIENT_IP_HEADER") == "cf-connecting-ip"


@pytest.mark.parametrize("key", ["HF_REVISION", "ALLOWED_ORIGINS"])
def test_secrets_are_set_in_the_dashboard_not_committed(key):
    (item,) = [e for e in render_service()["envVars"] if e["key"] == key]
    assert item.get("sync") is False
    assert "value" not in item


def is_number(text):
    try:
        float(text)
    except ValueError:
        return False
    return True


@pytest.fixture
def render_settings(monkeypatch):
    env = render_env()
    for key in list(LIMITS) + list(NOT_LIMITS):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return env, load_settings()


def test_every_numeric_render_value_is_classified():
    unknown = [k for k, v in render_env().items() if is_number(v) and k not in LIMITS and k not in NOT_LIMITS]
    assert not unknown, f"add these render.yaml keys to LIMITS or NOT_LIMITS: {unknown}"


def test_render_limits_do_not_exceed_config_defaults(render_settings):
    env, settings = render_settings
    default = Settings()
    array_guard_ok = (
        "MAX_ARRAY_ELEMENTS" in env and settings.max_array_elements <= default.max_array_elements
    )
    too_large = []
    for key, field_name in LIMITS.items():
        if key not in env:
            continue
        value, base = getattr(settings, field_name), getattr(default, field_name)
        if value <= base:
            continue
        if key in BOUNDED_BY_ARRAY_ELEMENTS and array_guard_ok:
            continue
        too_large.append(f"{key}={env[key]} (default {base})")
    assert not too_large, "render.yaml raises limits above app/config.py: " + ", ".join(too_large)


def test_render_sets_the_array_element_guard():
    assert "MAX_ARRAY_ELEMENTS" in render_env()
