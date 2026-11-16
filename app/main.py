"""FastAPI app: routes, CORS, error handlers, request limits.

Limits on POST /run, in the order they apply:
1. request body size (Content-Length, then counted while it streams);
2. per-IP token bucket (X-Forwarded-For entry TRUSTED_PROXY_HOPS from the
   right, see ratelimit.client_ip; uvicorn runs with --no-proxy-headers);
3. file type by magic bytes and per-type size, while the upload is copied;
4. one concurrency slot per run (503 busy if none frees up quickly);
5. a run timeout (504).
Errors are always {"error": {"code", "message"}} without stack traces. Each
/run log line carries only: run_id, kind, mode, size class, timings, status
code; never file names or contents. /docs and /openapi.json stay on on
purpose (portfolio project; the API holds no private data).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import pipeline
from .axes import parse_metadata
from .config import API_VERSION, load_settings
from .errors import ApiError
from .model import load_model
from .ratelimit import TokenBucket, client_ip, ip_class
from .readers import SNIFF_BYTES, configure_pillow, load_manifest, sniff
from .schemas import ErrorResponse, Health, RunResponse, Sample

log = logging.getLogger("gpradar")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    log.addHandler(handler)
    log.propagate = False

FORM_FIELDS = {"file", "sample_id", "mode", "metadata"}
MODES = ("full", "detector")
CHUNK = 256 * 1024
MULTIPART_OVERHEAD = 64 * 1024
ERROR_RESPONSES = {
    status: {"model": ErrorResponse} for status in (400, 413, 415, 422, 429, 500, 503, 504)
}


def _error_response(exc: ApiError):
    return JSONResponse(exc.body(), status_code=exc.status, headers=exc.headers)


class BodyLimitMiddleware:
    """Reject POST bodies over ``max_bytes`` without buffering them."""

    def __init__(self, app, max_bytes, paths=("/run",)):
        self.app = app
        self.max_bytes = int(max_bytes)
        self.paths = set(paths)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] not in self.paths or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        too_large = ApiError("too_large", f"the request is larger than {self.max_bytes} bytes")
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = 0
                if declared > self.max_bytes:
                    await _error_response(too_large)(scope, receive, send)
                    return
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise too_large
            return message

        await self.app(scope, limited_receive, send)


def _safe_filename(name):
    base = re.split(r"[\\/]", name or "")[-1]
    base = re.sub(r"[\x00-\x1f\x7f]", "", base).strip()
    return base[:120] or "upload"


def _unlink(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        log.warning("temp file cleanup failed")


async def _save_upload(upload: UploadFile, settings):
    """Copy the upload to a temp file, checking type and size on the way.

    Returns the temp path; the caller deletes it.
    """

    fd, path = tempfile.mkstemp(prefix="gpradar-", suffix=".bin")
    ok = False
    try:
        with os.fdopen(fd, "wb") as out:
            first = await upload.read(SNIFF_BYTES)
            file_type = sniff(first)
            if file_type not in ("png", "jpeg", "gprmax"):
                raise ApiError(
                    "unsupported_type", "accepted files are PNG, JPEG and gprMax HDF5 output (.out / .h5)"
                )
            limit = settings.max_image_bytes if file_type in ("png", "jpeg") else settings.max_raw_bytes
            size = len(first)
            out.write(first)
            while True:
                chunk = await upload.read(CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise ApiError("too_large", f"{file_type} files are limited to {limit} bytes")
                out.write(chunk)
        if size == 0:
            raise ApiError("bad_request", "the file is empty")
        ok = True
        return path
    finally:
        if not ok:
            _unlink(path)


def create_app(settings=None, model=None):
    """Build the app. ``model`` may be injected (tests); otherwise loaded at startup."""

    settings = settings or load_settings()
    log.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    configure_pillow(settings.max_pixels)
    manifest = load_manifest(settings.samples_dir)
    samples_root = Path(settings.samples_dir).resolve()

    @asynccontextmanager
    async def lifespan(app):
        app.state.model = model or load_model(settings)
        app.state.semaphore = asyncio.Semaphore(settings.max_concurrent_runs)
        app.state.executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrent_runs, thread_name_prefix="run"
        )
        log.info("model loaded: %s@%s sha256=%s", app.state.model.repo, app.state.model.revision, app.state.model.sha256[:12])
        log.debug("client IP: X-Forwarded-For entry %d from the right, else the socket peer",
                  settings.trusted_proxy_hops)
        try:
            yield
        finally:
            app.state.executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="GPRadar API",
        version=API_VERSION,
        description="GPR B-scan preprocessing and YOLO11n hyperbola detection. AGPL-3.0.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.limiter = TokenBucket(settings.rate_limit_per_min, settings.rate_limit_per_min / 60.0)
    app.state.client_ip_logged = False

    # Added first = inner. CORS is outermost so error responses carry CORS headers.
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_upload_bytes + MULTIPART_OVERHEAD)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        max_age=600,
    )

    @app.exception_handler(ApiError)
    async def api_error_handler(_request, exc: ApiError):
        return _error_response(exc)

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(_request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "bad_request")
        message = {404: "not found", 405: "method not allowed"}.get(exc.status_code, "bad request")
        return JSONResponse({"error": {"code": code, "message": message}}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request, _exc):
        return JSONResponse({"error": {"code": "bad_request", "message": "invalid request"}}, status_code=400)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_request, exc: Exception):
        log.error("unhandled error: %s", type(exc).__name__)
        return JSONResponse(
            {"error": {"code": "internal_error", "message": "internal error"}}, status_code=500
        )

    @app.get("/health", response_model=Health)
    async def health(request: Request):
        # Wake-up pings: nothing heavy here (the model was loaded at startup).
        return {"status": "ok", "api_version": API_VERSION, "model": request.app.state.model.info()}

    @app.get("/samples", response_model=list[Sample])
    async def samples():
        return [{key: value for key, value in entry.items() if key != "file"} for entry in manifest.values()]

    @app.post("/run", response_model=RunResponse, responses=ERROR_RESPONSES)
    async def run(request: Request):
        state = request.app.state
        run_id = uuid.uuid4().hex
        started = time.perf_counter()
        peer = request.client.host if request.client else None
        ip, ip_source = client_ip(request.headers, peer, settings.trusted_proxy_hops)
        if not state.client_ip_logged:
            # Once per process, debug only: lets you check the proxy setup without logging IPs.
            state.client_ip_logged = True
            n_forwarded = len([p for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()])
            log.debug(
                "first /run: client IP from %s (%d X-Forwarded-For entries, hops=%d), class=%s, peer class=%s",
                ip_source, n_forwarded, settings.trusted_proxy_hops, ip_class(ip), ip_class(peer or ""),
            )
        allowed, retry = state.limiter.take(ip)
        if not allowed:
            log.info("run_id=%s status=429", run_id)
            raise ApiError(
                "rate_limited",
                "too many runs from this address; try again shortly",
                headers={"Retry-After": str(max(1, int(retry + 0.999)))},
            )

        form = None
        tmp_path = None
        handed_off = False
        try:
            try:
                form = await request.form(max_files=1, max_fields=len(FORM_FIELDS), max_part_size=64 * 1024)
            except ApiError:
                raise
            except Exception:
                raise ApiError("bad_request", "expected a multipart/form-data body") from None
            unknown = set(form.keys()) - FORM_FIELDS
            if unknown:
                raise ApiError("bad_request", "unknown form field(s); allowed: file, sample_id, mode, metadata")

            upload = form.get("file")
            sample_id = form.get("sample_id")
            if isinstance(upload, str) and upload == "":
                upload = None
            if isinstance(sample_id, str) and sample_id.strip() == "":
                sample_id = None
            if upload is not None and not isinstance(upload, UploadFile):
                raise ApiError("bad_request", "file must be a file upload")
            if (upload is None) == (sample_id is None):
                raise ApiError("bad_request", "send exactly one of file or sample_id")

            mode = form.get("mode")
            mode = mode.strip().lower() if isinstance(mode, str) and mode.strip() else None
            if mode is not None and mode not in MODES:
                raise ApiError("bad_request", "mode must be 'full' or 'detector'")
            metadata_text = form.get("metadata")
            if metadata_text is not None and not isinstance(metadata_text, str):
                raise ApiError("invalid_metadata", "metadata must be a JSON string")
            user_meta = parse_metadata(metadata_text, settings.max_metadata_chars)

            if upload is not None:
                tmp_path = await _save_upload(upload, settings)
                job_path, filename, default_mode, allow_npz = tmp_path, _safe_filename(upload.filename), None, False
            else:
                entry = manifest.get(str(sample_id))
                if entry is None:
                    raise ApiError("bad_request", "unknown sample_id; see GET /samples")
                job_path = (samples_root / entry["file"]).resolve()
                if samples_root not in job_path.parents:
                    raise ApiError("bad_request", "unknown sample_id")
                filename, default_mode, allow_npz = entry["file"], entry["default_mode"], True

            def job():
                try:
                    return pipeline.run_job(
                        job_path, settings, state.model, mode, user_meta, filename, run_id,
                        default_mode=default_mode, allow_npz=allow_npz,
                    )
                finally:
                    if tmp_path is not None:
                        _unlink(tmp_path)

            try:
                await asyncio.wait_for(state.semaphore.acquire(), timeout=settings.queue_wait_s)
            except asyncio.TimeoutError:
                raise ApiError("busy", "the server is busy; try again in a few seconds") from None

            release_later = False
            try:
                future = asyncio.get_running_loop().run_in_executor(state.executor, job)
                handed_off = True
                try:
                    result = await asyncio.wait_for(asyncio.shield(future), timeout=settings.run_timeout_s)
                except asyncio.TimeoutError:
                    # The thread cannot be killed: keep its slot until it ends.
                    release_later = True
                    future.add_done_callback(lambda f: (state.semaphore.release(), f.exception()))
                    log.warning("run_id=%s timed out after %.0f s", run_id, settings.run_timeout_s)
                    raise ApiError("timeout", f"the run took longer than {settings.run_timeout_s:g} s") from None
            finally:
                if not release_later:
                    state.semaphore.release()
        except ApiError as exc:
            log.info("run_id=%s status=%d code=%s total_ms=%.0f", run_id, exc.status, exc.code,
                     1000 * (time.perf_counter() - started))
            raise
        except ValueError as exc:
            log.info("run_id=%s status=400 error=%s", run_id, type(exc).__name__)
            raise ApiError("bad_request", "the input could not be processed") from None
        except Exception as exc:
            log.error("run_id=%s status=500 error=%s", run_id, type(exc).__name__)
            raise ApiError("internal_error", "internal error") from None
        finally:
            if tmp_path is not None and not handed_off:
                _unlink(tmp_path)
            if form is not None:
                await form.close()

        size = result.pop("_size_class")
        t = result["timings_ms"]
        log.info(
            "run_id=%s status=200 kind=%s mode=%s size=%s read_ms=%.0f preprocess_ms=%.0f "
            "inference_ms=%.0f total_ms=%.0f",
            run_id, result["input"]["kind"], result["mode"], size,
            t["read"], t["preprocess"], t["inference"], t["total"],
        )
        return JSONResponse(RunResponse.model_validate(result).model_dump(exclude_unset=True))

    return app


app = create_app()
