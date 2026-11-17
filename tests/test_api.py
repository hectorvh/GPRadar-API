import glob
import io
import json
import logging
import os
import tempfile
from contextlib import contextmanager

import h5py
import numpy as np
import pytest
from PIL import Image

from app.readers import load_manifest
from app.config import SAMPLES_DIR
from app.schemas import ErrorResponse, Health, RunResponse, Sample
from tests.conftest import ALLOWED_ORIGIN
from tests.test_pipeline import DT, SPACING, synthetic_bscan

SAMPLE_IDS = list(load_manifest(SAMPLES_DIR))


def png_bytes(width=64, height=48):
    buffer = io.BytesIO()
    Image.fromarray(np.full((height, width), 128, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def gprmax_bytes(tmp_path):
    """A small merged-B-scan-like gprMax HDF5 file."""

    path = tmp_path / "scan.out"
    with h5py.File(path, "w") as out:
        out.attrs["dt"] = DT
        out.attrs["dx_dy_dz"] = [SPACING, SPACING, SPACING]
        out.attrs["rxsteps"] = [1, 0, 0]
        rx = out.create_group("rxs/rx1")
        rx.attrs["Position"] = [0.1, 0.0, 0.0]
        rx.create_dataset("Ez", data=synthetic_bscan())
    return path.read_bytes()


def assert_error(response, status, code):
    assert response.status_code == status, response.text
    body = ErrorResponse.model_validate(response.json())
    assert body.error.code == code
    assert "Traceback" not in response.text


def our_temp_files():
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "gpradar-*")))


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = Health.model_validate(response.json())
    assert body.status == "ok"
    assert len(body.model.sha256) == 64
    assert body.model.revision == "test"


def test_samples_list(client):
    response = client.get("/samples")
    assert response.status_code == 200
    items = [Sample.model_validate(item) for item in response.json()]
    assert [item.id for item in items] == SAMPLE_IDS
    assert all("file" not in item for item in response.json())


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_run_each_sample(client, sample_id):
    response = client.post("/run", data={"sample_id": sample_id})
    assert response.status_code == 200, response.text
    body = RunResponse.model_validate(response.json())
    assert body.input.filename
    assert body.images.processed_png
    assert all(d.confidence >= 0.05 for d in body.detections)
    for d in body.detections:
        assert 0 <= d.box_norm.x_min <= d.box_norm.x_max <= 1
        assert 0 <= d.box_norm.y_min <= d.box_norm.y_max <= 1
    if body.input.kind == "gprmax":
        assert body.mode == "full"
        assert body.images.raw_png
        assert body.axes.x.unit == "m" and body.axes.y.unit == "ns" and body.axes.depth is not None
        assert body.metadata.time_window_ns.source == "file"
        for d in body.detections:
            assert d.apex.twt_ns is not None and d.apex.depth_m is not None
    else:
        assert body.mode == "detector"
        assert body.axes.x.unit == "trace" and body.axes.depth is None


def test_run_sample_with_user_metadata(client):
    meta = json.dumps({"time_window_ns": 60, "trace_spacing_m": 0.02, "velocity_m_ns": 0.12})
    response = client.post("/run", data={"sample_id": SAMPLE_IDS[0], "metadata": meta})
    assert response.status_code == 200
    body = RunResponse.model_validate(response.json())
    assert body.metadata.time_window_ns.source == "user"
    assert body.axes.depth.max == pytest.approx(0.12 * 60 / 2)
    for d in body.detections:
        assert d.apex.depth_m == pytest.approx(0.12 * d.apex.twt_ns / 2, abs=1e-3)


def test_upload_png(client):
    before = our_temp_files()
    response = client.post("/run", files={"file": ("my scan.png", png_bytes(), "image/png")})
    assert response.status_code == 200, response.text
    body = RunResponse.model_validate(response.json())
    assert body.input.filename == "my scan.png"
    assert (body.input.width_px, body.input.height_px) == (64, 48)
    assert our_temp_files() <= before  # temp file deleted


def test_upload_gprmax(client, tmp_path):
    response = client.post("/run", files={"file": ("scan.out", gprmax_bytes(tmp_path), "application/octet-stream")})
    assert response.status_code == 200, response.text
    body = RunResponse.model_validate(response.json())
    assert body.input.kind == "gprmax" and body.mode == "full"
    assert body.metadata.trace_spacing_m.source == "file"
    assert body.metadata.trace_spacing_m.value == pytest.approx(SPACING)
    # A merged gprMax file has no frequency: the default applies.
    assert body.metadata.frequency_mhz.value == 400.0
    assert body.metadata.frequency_mhz.source == "default"


def test_user_metadata_overrides_the_file(client):
    sample = "gprmax_field_400mhz_seed182"
    response = client.post("/run", data={"sample_id": sample, "metadata": json.dumps({"trace_spacing_m": 0.1})})
    assert response.status_code == 200
    body = RunResponse.model_validate(response.json())
    assert body.metadata.trace_spacing_m.source == "user"
    assert body.metadata.trace_spacing_m.value == 0.1
    assert body.metadata.time_window_ns.source == "file"
    assert any("replaced the file" in w for w in body.warnings)


def test_colour_image_warns(client):
    buffer = io.BytesIO()
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[..., 0] = 200  # red
    Image.fromarray(rgb).save(buffer, format="PNG")
    response = client.post("/run", files={"file": ("c.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 200
    assert any("colour image converted to greyscale" in w for w in response.json()["warnings"])


def test_raw_detector_mode_warns(client):
    response = client.post("/run", data={"sample_id": "gprmax_field_400mhz_seed182", "mode": "detector"})
    assert response.status_code == 200
    assert "raw data without preprocessing; detections will be poor" in response.json()["warnings"]


def test_fake_png_is_415(client):
    before = our_temp_files()
    response = client.post("/run", files={"file": ("x.png", b"this is plain text, not a PNG", "image/png")})
    assert_error(response, 415, "unsupported_type")
    assert our_temp_files() <= before


def test_npz_upload_is_415(client):
    path = SAMPLES_DIR / "gprmax_field_400mhz_seed182.npz"
    response = client.post("/run", files={"file": ("scene.npz", path.read_bytes(), "application/zip")})
    assert_error(response, 415, "unsupported_type")


def test_oversize_is_413(make_client):
    # Over the request limit (4 kB + multipart allowance): rejected while streaming.
    with make_client(max_raw_bytes=4_000, max_image_bytes=4_000) as small:
        payload = b"\x89PNG\r\n\x1a\n" + b"\0" * 200_000
        response = small.post("/run", files={"file": ("big.png", payload, "image/png")})
        assert_error(response, 413, "too_large")


def test_oversize_image_under_request_limit_is_413(make_client):
    with make_client(max_image_bytes=1_000) as small:
        payload = b"\x89PNG\r\n\x1a\n" + b"\0" * 5_000
        response = small.post("/run", files={"file": ("big.png", payload, "image/png")})
        assert_error(response, 413, "too_large")


def test_file_and_sample_is_400(client):
    response = client.post(
        "/run",
        data={"sample_id": SAMPLE_IDS[0]},
        files={"file": ("a.png", png_bytes(), "image/png")},
    )
    assert_error(response, 400, "bad_request")


def test_neither_file_nor_sample_is_400(client):
    assert_error(client.post("/run", data={"mode": "full"}), 400, "bad_request")


def test_bad_mode_is_400(client):
    assert_error(client.post("/run", data={"sample_id": SAMPLE_IDS[0], "mode": "fast"}), 400, "bad_request")


@pytest.mark.parametrize(
    "metadata",
    [
        "not json",
        "[1, 2]",
        json.dumps({"velocity_m_ns": "fast"}),
        json.dumps({"velocity_m_ns": 5}),
        json.dumps({"colour": 1}),
    ],
)
def test_bad_metadata_is_422(client, metadata):
    response = client.post("/run", data={"sample_id": SAMPLE_IDS[0], "metadata": metadata})
    assert_error(response, 422, "invalid_metadata")


def test_eleventh_quick_request_is_429(make_client):
    with make_client() as fresh:
        headers = {"X-Forwarded-For": "203.0.113.7"}
        for _ in range(10):
            assert fresh.post("/run", data={}, headers=headers).status_code == 400
        response = fresh.post("/run", data={}, headers=headers)
        assert_error(response, 429, "rate_limited")
        assert "retry-after" in response.headers
        # Another client is not affected.
        other = fresh.post("/run", data={}, headers={"X-Forwarded-For": "203.0.113.8"})
        assert other.status_code == 400


def test_forged_leftmost_entry_does_not_change_the_rate_limit_key(make_client):
    # One trusted proxy appends the real client (203.0.113.9); the client forges the rest.
    with make_client() as fresh:
        for i in range(10):
            headers = {"X-Forwarded-For": f"10.0.0.{i}, 203.0.113.9"}
            assert fresh.post("/run", data={}, headers=headers).status_code == 400
        response = fresh.post("/run", data={}, headers={"X-Forwarded-For": "198.51.100.77, 203.0.113.9"})
        assert_error(response, 429, "rate_limited")


def test_two_trusted_hops(make_client):
    # Client -> CDN -> load balancer: the client is the second entry from the right.
    with make_client(trusted_proxy_hops=2) as fresh:
        for i in range(10):
            headers = {"X-Forwarded-For": f"10.0.0.{i}, 198.51.100.7, 172.16.0.{i}"}
            assert fresh.post("/run", data={}, headers=headers).status_code == 400
        response = fresh.post("/run", data={}, headers={"X-Forwarded-For": "198.51.100.7, 172.16.9.9"})
        assert_error(response, 429, "rate_limited")
        # Another client behind the same proxies has its own bucket.
        other = fresh.post("/run", data={}, headers={"X-Forwarded-For": "198.51.100.8, 172.16.0.1"})
        assert other.status_code == 400


def test_client_ip_rules():
    from app.ratelimit import client_ip

    headers = {"x-forwarded-for": "1.1.1.1, 2.2.2.2, 3.3.3.3"}
    assert client_ip(headers, "9.9.9.9", 1) == ("3.3.3.3", "x-forwarded-for")
    assert client_ip(headers, "9.9.9.9", 2) == ("2.2.2.2", "x-forwarded-for")
    assert client_ip(headers, "9.9.9.9", 4) == ("9.9.9.9", "peer")  # header too short
    assert client_ip({}, "9.9.9.9", 1) == ("9.9.9.9", "peer")  # header missing
    assert client_ip({"x-forwarded-for": "junk"}, "9.9.9.9", 1) == ("9.9.9.9", "peer")
    assert client_ip(headers, "9.9.9.9", 0) == ("9.9.9.9", "peer")


# --- Client IP behind Cloudflare (Render): CF-Connecting-IP first ----------------

CF = "cf-connecting-ip"


@pytest.mark.parametrize("value", ["198.51.100.4", " 198.51.100.4 ", "2001:db8::4"])
def test_client_ip_header_present_is_used(value):
    from app.ratelimit import client_ip

    headers = {CF: value, "x-forwarded-for": "203.0.113.1, 172.70.1.1"}
    ip, source = client_ip(headers, "9.9.9.9", 1, CF)
    assert ip == value.strip()
    assert source == "header:cf-connecting-ip"


@pytest.mark.parametrize(
    "value",
    ["198.51.100.4, 203.0.113.5", "198.51.100.4,", "junk", "", "999.1.1.1", "198.51.100.4:443"],
)
def test_client_ip_header_with_a_list_or_garbage_falls_back(value):
    from app.ratelimit import client_ip

    headers = {CF: value, "x-forwarded-for": "10.0.0.1, 172.70.1.1"}
    assert client_ip(headers, "9.9.9.9", 1, CF) == ("172.70.1.1", "x-forwarded-for")
    assert client_ip({CF: value}, "9.9.9.9", 1, CF) == ("9.9.9.9", "peer")


def test_client_ip_header_absent_keeps_the_forwarded_for_rules():
    from app.ratelimit import client_ip

    headers = {"x-forwarded-for": "1.1.1.1, 2.2.2.2, 3.3.3.3"}
    assert client_ip(headers, "9.9.9.9", 1, CF) == ("3.3.3.3", "x-forwarded-for")
    assert client_ip(headers, "9.9.9.9", 2, CF) == ("2.2.2.2", "x-forwarded-for")
    assert client_ip({}, "9.9.9.9", 1, CF) == ("9.9.9.9", "peer")


def test_client_ip_header_can_be_disabled():
    from app.ratelimit import client_ip

    headers = {CF: "198.51.100.4", "x-forwarded-for": "3.3.3.3"}
    assert client_ip(headers, "9.9.9.9", 1, "") == ("3.3.3.3", "x-forwarded-for")


@pytest.mark.parametrize("forged", ["6.6.6.6", "6.6.6.6, 7.7.7.7", "6.6.6.6, 198.51.100.4"])
def test_forged_forwarded_for_never_becomes_the_key_when_the_header_is_set(forged):
    from app.ratelimit import client_ip

    headers = {CF: "198.51.100.4", "x-forwarded-for": f"{forged}, 172.70.1.1"}
    assert client_ip(headers, "9.9.9.9", 1, CF) == ("198.51.100.4", "header:cf-connecting-ip")


def test_rate_limit_follows_cf_connecting_ip_not_forged_forwarded_for(make_client):
    # The live failure on Render: same client, a new forged X-Forwarded-For each
    # time (and a varying Cloudflare edge on the right). It must still hit 429.
    with make_client(rate_limit_per_min=10) as fresh:
        for i in range(10):
            headers = {CF: "198.51.100.20", "X-Forwarded-For": f"10.9.{i}.1, 172.70.{i}.1"}
            assert fresh.post("/run", data={}, headers=headers).status_code == 400
        response = fresh.post(
            "/run", data={}, headers={CF: "198.51.100.20", "X-Forwarded-For": "8.8.8.8, 172.70.99.1"}
        )
        assert_error(response, 429, "rate_limited")
        # Another real client behind the same Cloudflare edge has its own bucket.
        other = fresh.post(
            "/run", data={}, headers={CF: "198.51.100.21", "X-Forwarded-For": "8.8.8.8, 172.70.99.1"}
        )
        assert other.status_code == 400


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@contextmanager
def captured_gpradar_log():
    logger = logging.getLogger("gpradar")
    handler = _Capture()
    logger.addHandler(handler)
    try:
        yield handler.lines
    finally:
        logger.removeHandler(handler)


def _ip_lines(lines):
    return [line for line in lines if "client IP: source=" in line]


def test_client_ip_diagnostics_once_at_info_without_addresses(make_client):
    headers = {CF: "198.51.100.30", "True-Client-IP": "198.51.100.30", "X-Forwarded-For": "6.6.6.6, 172.70.1.1"}
    with captured_gpradar_log() as lines, make_client(log_level="info") as fresh:
        for _ in range(3):
            fresh.post("/run", data={}, headers=headers)
    assert any("client IP rule: header cf-connecting-ip" in line for line in lines)
    (line,) = _ip_lines(lines)
    assert "source=header:cf-connecting-ip" in line
    assert "x-forwarded-for entries=2" in line
    assert "cf-connecting-ip=present" in line and "true-client-ip=present" in line
    assert "class[cf-connecting-ip]=absent" not in line and "class[cf-connecting-ip]=list" not in line
    assert "class[x-forwarded-for -1]=public-ipv4" in line
    for address in ("198.51.100.30", "6.6.6.6", "172.70.1.1"):
        assert all(address not in text for text in lines)


def test_log_client_ip_sources_logs_every_run(make_client):
    with captured_gpradar_log() as lines, make_client(log_client_ip_sources=True) as fresh:
        for _ in range(3):
            fresh.post("/run", data={}, headers={"X-Forwarded-For": "6.6.6.6"})
    found = _ip_lines(lines)
    assert len(found) == 3
    assert all("source=x-forwarded-for" in line and "cf-connecting-ip=absent" in line for line in found)


def test_client_ip_settings_from_environment(monkeypatch):
    from app.config import load_settings

    monkeypatch.delenv("CLIENT_IP_HEADER", raising=False)
    monkeypatch.delenv("LOG_CLIENT_IP_SOURCES", raising=False)
    settings = load_settings()
    assert settings.client_ip_header == "cf-connecting-ip" and settings.log_client_ip_sources is False
    monkeypatch.setenv("CLIENT_IP_HEADER", "")
    monkeypatch.setenv("LOG_CLIENT_IP_SOURCES", "1")
    settings = load_settings()
    assert settings.client_ip_header == "" and settings.log_client_ip_sources is True
    monkeypatch.setenv("CLIENT_IP_HEADER", "True-Client-IP")
    assert load_settings().client_ip_header == "true-client-ip"


def jpeg_bytes(width, height):
    """A smooth colour JPEG (small file, any pixel count)."""

    x = np.linspace(0, 255, width, dtype=np.float32)
    y = np.linspace(0, 255, height, dtype=np.float32)
    grey = ((x[None, :] + y[:, None]) / 2).astype(np.uint8)
    rgb = np.stack([grey, grey[::-1], 255 - grey], axis=2)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def test_20_megapixel_jpeg_is_rejected(client):
    data = jpeg_bytes(5000, 4000)
    response = client.post("/run", files={"file": ("big.jpg", data, "image/jpeg")})
    assert_error(response, 413, "too_large")


def test_10_megapixel_jpeg_runs_at_reduced_size(client):
    data = jpeg_bytes(4000, 2500)
    response = client.post("/run", files={"file": ("ten.jpg", data, "image/jpeg")})
    assert response.status_code == 200, response.text
    body = RunResponse.model_validate(response.json())
    # Decoded at 1/8 (draft): the model image is 500 x 313; traces/samples stay original.
    assert (body.input.width_px, body.input.height_px) == (500, 313)
    assert (body.input.n_traces, body.input.n_samples) == (4000, 2500)
    assert any("decoded at 1/8" in w for w in body.warnings)
    assert any("colour image" in w for w in body.warnings)


def test_cors_allowed_origin(client):
    response = client.get("/health", headers={"Origin": ALLOWED_ORIGIN})
    assert response.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN
    preflight = client.options(
        "/run",
        headers={"Origin": ALLOWED_ORIGIN, "Access-Control-Request-Method": "POST"},
    )
    assert preflight.status_code == 200
    assert preflight.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN


def test_cors_other_origin(client):
    response = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in response.headers


def test_cors_header_on_error_responses(client):
    response = client.post("/run", data={}, headers={"Origin": ALLOWED_ORIGIN})
    assert response.status_code == 400
    assert response.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN


def test_unknown_route_is_json_404(client):
    assert_error(client.get("/nope"), 404, "not_found")
