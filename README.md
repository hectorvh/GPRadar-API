# GPRadar-API

A small web service that finds hyperbolas (buried pipes, voids, rocks) in
ground-penetrating radar (GPR) B-scans. You send it a B-scan image or a gprMax
simulation output. It runs the GPRadar processing chain and a YOLO11n detector
(ONNX, CPU only), then returns the boxes with their apex position, in traces and
samples and, when the scan's metadata allows, in metres, nanoseconds and approximate depth.

It is the backend of the GPRadar demo. The research code (processing, simulation,
training) lives in [hectorvh/GPRadar](https://github.com/hectorvh/GPRadar). The
service is sized for Render's free tier: 512 MB RAM and 0.1 CPU.

## Licence

**AGPL-3.0** (see [LICENSE](LICENSE)). The detector weights are AGPL-3.0
(Ultralytics YOLO11), and the processing code is vendored from GPRadar, which is
AGPL-3.0 too. If you run this service for other people over a network, the AGPL
requires you to offer them its source code: the web UI links to this repository in its footer.

## Model

| | |
|---|---|
| Repository | [hectorvh/gpradar-hyperbola-yolo](https://huggingface.co/hectorvh/gpradar-hyperbola-yolo) |
| File | `best.onnx` (YOLO11n, 224 x 224 input, output `(1, 5, 1029)`, no NMS in the graph) |
| Revision | pinned with the Docker build argument `HF_REVISION` (a full commit hash; `/health` reports it) |
| SHA-256 | `965ab4f73869c234a75a9ce1a0437ac243b0b2520aa77e5f6b364a354112b42d` (the file `tests/fixtures/golden.json` was made with) |

The Docker build downloads the model once at that revision and stores it in the
image, so the running service never contacts Hugging Face. `GET /health` reports
the repository, revision and SHA-256 of the model it loaded.

## Data

- Real B-scan images: Mojahid et al. (2025), *Intelligent recognition of subsurface
  utilities and voids using ground-penetrating radar and deep learning techniques:
  Data of GPR profiles images*, Mendeley Data, DOI
  [10.17632/ww7fd9t325.1](https://doi.org/10.17632/ww7fd9t325.1). Licence CC BY 4.0.
  The two demo images in `app/samples/` are from its train split.
- Simulated scene: made with gprMax (Warren, Giannopoulos and Giannakis 2016,
  *gprMax: Open source software to simulate electromagnetic wave propagation for
  Ground Penetrating Radar*, Computer Physics Communications 209, 163-170).

## API

| Method | Path | |
|---|---|---|
| GET | `/health` | `{status, api_version, model: {repo, file, revision, sha256}}` |
| GET | `/samples` | the demo samples: `[{id, name, kind, default_mode, citation, description}]` |
| POST | `/run` | multipart form: `file` **or** `sample_id`, `mode` (`full` / `detector`), `metadata` (JSON string) |

`file` can be PNG, JPEG or a gprMax HDF5 output (`.out` / `.h5`). The server checks
the file type from its first bytes and ignores the file name.
`metadata` fields, all optional: `frequency_mhz` (10-3000), `time_window_ns`
(1-5000), `trace_spacing_m` (0.001-5), `velocity_m_ns` (0.01-0.3); out of range
gives 422. Each field resolves user → file → default and the response says
which (`source`). Defaults: 400 MHz, 0.10 m/ns; the time window and trace
spacing have no default, so the axes of a plain image stay in traces and samples.

The response contains every detection with confidence ≥ 0.05, so the UI can filter
by threshold without another request. The `processed_png` image is exactly what
the model saw, and `box_px` is in its pixels. The apex is the top centre of a box:
`twt_ns = y_min / height x time_window_ns`, `distance_m = x_centre_trace x trace_spacing_m`,
`depth_m = velocity x twt_ns / 2` (approximate). Errors are
`{"error": {"code", "message"}}` with these codes: 400 `bad_request`, 413 `too_large`,
415 `unsupported_type`, 422 `invalid_metadata`, 429 `rate_limited`, 503 `busy`,
504 `timeout`. The interactive schema is at `/docs`.

**Modes.**
- gprMax, `full` (default): time zero, dewow, mean-trace removal, band-pass,
  AGC (windows of three periods of the centre frequency) on the whole record,
  robust 8-bit scaling, then the resize rule below. The raw record is returned
  too (`raw_png`, same resize).
- gprMax, `detector`: robust 8-bit scaling, the same resize, and a warning.
- image, `detector` (default): 8-bit grey (EXIF orientation applied; colour
  images are converted with a warning) straight to the model. JPEGs that the
  model would shrink anyway are decoded at reduced size with `Image.draft("L")`
  (1/2 to 1/8, keeping at least 448 px on the side the model scales to 224; a
  warning says so). Smaller JPEGs decode exactly as in GPRadar 06f.
- image, `full`: allowed with a warning. Pixels become `(v - 128) / 128`,
  `dt = time_window_ns / height` (default 50 ns), then dewow, background
  removal, band-pass and AGC (no time zero).

**Resize rule for raw records (gprMax uploads and the bundled `.npz`).** The
8-bit record image is resized to exactly **224 x 224** with Pillow bilinear
(`Image.fromarray(u8, "L").resize((224, 224), Image.Resampling.BILINEAR)`),
whatever the number of traces and samples, in this order: process the whole
record → `robust_to_uint8` → resize. This is GPRadar
`src/07a_export_demo_samples.py` `processed_image`, which made the golden
input; `tests/test_pipeline.py` checks that the bundled scene reproduces the
stored `processed_u8_224` exactly. So a long record is squeezed rather than tiled, as
the training crops were (05a also resized every crop to 224 x 224). `box_px`
is in that 224 x 224 image; apex `trace`/`sample` are mapped back to the record.

**Tiling (images only).** If width / height ≤ 2 the whole image goes to the model. Longer
profiles are cut into square windows (side = height) with 50% overlap; boxes cut
by a window edge are joined with the matching box from the next window, then NMS
(IoU 0.5). At most 12 windows (`MAX_WINDOWS`); a longer profile is cut and the
response says so. Returned images wider than 2000 px are downscaled for display
only: `box_px` stays in model-image pixels, and `box_norm` (0-1) is for drawing.

**Limits** (environment variables, plan 2.3):

| Variable | Default | Purpose |
|---|---|---|
| `ALLOWED_ORIGINS` | `http://localhost:5173` | CORS allow-list, comma-separated (no `*`) |
| `MAX_IMAGE_MB` | 10 | image upload limit |
| `MAX_RAW_MB` | 25 | gprMax upload limit |
| `MAX_TRACES` / `MAX_SAMPLES` | 4000 / 4096 | array size guard |
| `MAX_ARRAY_ELEMENTS` | 2,000,000 | samples x traces through the float chain (memory guard, see below) |
| `MAX_PIXELS` | 12,000,000 | image pixel limit (Pillow `MAX_IMAGE_PIXELS` and our own check) |
| `RUN_TIMEOUT_S` | 25 | hard limit per run (504) |
| `MAX_CONCURRENT_RUNS` | 1 | semaphore; extra requests wait `QUEUE_WAIT_S` (2), then 503 |
| `RATE_LIMIT_PER_MIN` | 10 | `/run` per client IP |
| `TRUSTED_PROXY_HOPS` | 1 | proxies in front of the app; the client IP is the `X-Forwarded-For` entry this many places from the right |
| `MIN_CONF` | 0.05 | lowest confidence returned |
| `MAX_WINDOWS` | 12 | tiling cap |
| `ONNX_THREADS` | 1 | onnxruntime intra/inter-op threads |
| `LOG_LEVEL` | info | log level |

Size limits are enforced while the upload streams: `Content-Length` is checked
first, and then the bytes are counted, so a missing or false header cannot get
past the limit. The file type comes from magic bytes, not the extension.
`MAX_ARRAY_ELEMENTS` exists because the float64 chain makes several copies of
the record: a 4096 x 4000 record would not fit in 512 MB (measured: 2M values
peak at ~290 MB, 4M values ~450 MB). The logs record run_id, kind, mode, size class,
timings and status code. They never record file names or file contents.
`/docs` is on deliberately (portfolio project).

**Client IP.** Each proxy appends to `X-Forwarded-For`, so only the right-hand
entries are reliable; the client can forge anything to their left. The app takes the entry
`TRUSTED_PROXY_HOPS` places from the right, and it falls back to the socket peer
(`request.client.host`) when the header is missing, too short or not an IP.
uvicorn runs with `--no-proxy-headers` on purpose: with `--proxy-headers
--forwarded-allow-ips='*'` it would replace the client with the left-most
(forgeable) entry. To check the setting on Render, set `LOG_LEVEL=debug`: the
first `/run` logs where the client IP came from, the number of header entries
and the address class (for example `public-ipv4`), but never the IP itself. If the class is
`private`, there is one more proxy than configured: raise `TRUSTED_PROXY_HOPS`.

Measured peak memory (one process, including the test client's copy of the
upload): largest allowed JPEG (12 MP colour) 185 MiB; 12 MP PNG (no draft)
283 MiB; gprMax record at the 2M-value cap ~290 MiB.

## Run locally

Python 3.12:

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# the model: point MODEL_PATH at a local best.onnx (or put it in model/best.onnx)
export MODEL_PATH=/path/to/best.onnx

ALLOWED_ORIGINS=http://localhost:5173 uvicorn app.main:app --port 8000
python scripts/smoke.py http://localhost:8000     # /health and /run on each sample
```

Tests:

```bash
python -m pytest
```

If `model/best.onnx` (or the file at `MODEL_PATH`) is missing, the API tests use a
tiny dummy ONNX model and the golden test is skipped. The golden test checks
that the API reproduces the research code's boxes (`tests/fixtures/golden.json`):
the same number of boxes at confidence ≥ 0.25, each with IoU ≥ 0.9. GitHub
Actions runs `pytest` on every push. To run the golden test in CI too, set the
repository variable `HF_REVISION`; CI then downloads the model at that commit.

## Run with Docker

```bash
docker build --build-arg HF_REVISION=<HF_COMMIT> -t gpradar-api .
docker run --rm -p 8000:8000 -m 512m --cpus 0.1   -e ALLOWED_ORIGINS=http://localhost:5173 gpradar-api
python scripts/smoke.py http://localhost:8000
docker stats --no-stream
```

The build downloads `best.onnx` with `huggingface_hub` at `HF_REVISION`. It
stops with an error if the argument is missing or still the placeholder.
Optional: `--build-arg MODEL_SHA256=<hex>` checks the file's hash.
On Render, `render.yaml` defines the service. Its values override the defaults
in the limits table, so `tests/test_render_config.py` fails if any limit there
is larger than its default in `app/config.py`.
Render passes environment variables to the Docker build as build arguments,
so you set `HF_REVISION` and `ALLOWED_ORIGINS` in the Render dashboard.

## Layout

```
app/
  main.py        routes, CORS, error handlers, request limits
  config.py      settings from environment variables
  model.py       ONNX session (loaded once), model info + SHA-256
  detect.py      letterbox, NMS, detect (vendored from GPRadar 06f) + tiling
  readers.py     file type by magic bytes; image -> uint8; gprMax / sample .npz -> GprScan
  pipeline.py    input -> model image; the /run job
  axes.py        metadata (file / user / default) and physical axes, apex
  schemas.py     Pydantic models of the responses
  ratelimit.py   per-IP token bucket
  gpr/           vendored from GPRadar (each file names its source and commit)
  samples/       manifest.json + demo files
tests/           pytest; fixtures/golden.json
scripts/         smoke.py (calls /health and /run on each sample)
```

The vendored code is copied from GPRadar at commit `1b052df`, and each file says
which source file it came from. To update it, change GPRadar first and copy the
code again. Do not edit the copies here.
