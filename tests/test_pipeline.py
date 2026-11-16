import numpy as np

from app.gpr.chain import process_scene_array
from app.gpr.loader import GprScan, GprSource
from app.config import SAMPLES_DIR, Settings
from app.pipeline import WARN_IMAGE_FULL, chain_to_uint8, prepare
from app.readers import ImageInput, ScanInput, read_scan_npz

DT = 0.1e-9  # 0.1 ns
FREQ = 400e6
N_SAMPLES, N_TRACES = 512, 64
SPACING = 0.05


def ricker(t, f):
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def synthetic_bscan(seed=0):
    """Direct wave + one diffraction hyperbola + noise, gprMax-like (samples x traces)."""

    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES) * DT
    x = np.arange(N_TRACES) * SPACING
    v = 0.1e9  # m/s
    t0, x0, delay = 25e-9, x[N_TRACES // 2], 3e-9
    data = np.zeros((N_SAMPLES, N_TRACES))
    for j, xj in enumerate(x):
        data[:, j] += 5.0 * ricker(t - delay, FREQ)
        apex = np.sqrt(t0**2 + (2 * (xj - x0) / v) ** 2)
        data[:, j] += 0.5 * ricker(t - delay - apex, FREQ)
    return data + 0.01 * rng.standard_normal(data.shape)


def scan_input(data):
    positions = np.zeros((data.shape[1], 3))
    positions[:, 0] = SPACING * np.arange(data.shape[1])
    source = GprSource("", None, None, None, None, None, FREQ, positions[0], positions)
    return ScanInput(GprScan(data, DT, SPACING, FREQ, source, "Ez", positions))


def test_chain_runs_without_nan_and_keeps_the_shape():
    data = synthetic_bscan()
    processed, _picks, windows = process_scene_array(data, DT, FREQ)
    assert np.all(np.isfinite(processed))
    assert windows["dewow_ns"] == 7.5  # three periods at 400 MHz
    grey, _ = chain_to_uint8(data, DT, FREQ)
    assert grey.dtype == np.uint8
    assert grey.shape == data.shape
    assert grey.min() < 100 < 160 < grey.max()  # the hyperbola gives contrast


def test_gprmax_full_mode_builds_the_model_image():
    prepared = prepare(scan_input(synthetic_bscan()), "full", {}, max_elements=10**7)
    assert prepared.model_image.dtype == np.uint8
    assert prepared.model_image.shape == (224, 224)
    assert prepared.raw_image.shape == (224, 224)
    assert (prepared.n_samples, prepared.n_traces) == (N_SAMPLES, N_TRACES)
    assert prepared.metadata["time_window_ns"] == {"value": N_SAMPLES * DT * 1e9, "source": "file"}
    assert prepared.metadata["frequency_mhz"]["source"] == "file"
    assert prepared.metadata["velocity_m_ns"] == {"value": 0.10, "source": "default"}


def test_image_full_mode_runs_and_warns():
    rng = np.random.default_rng(1)
    grey = rng.integers(90, 170, size=(120, 160), dtype=np.uint8)
    prepared = prepare(ImageInput(grey), "full", {}, max_elements=10**7)
    assert WARN_IMAGE_FULL in prepared.warnings
    assert prepared.model_image.dtype == np.uint8
    assert prepared.model_image.shape == grey.shape
    assert prepared.raw_image is grey


def test_image_detector_mode_is_the_image_itself():
    grey = np.full((50, 60), 128, dtype=np.uint8)
    prepared = prepare(ImageInput(grey), "detector", {"time_window_ns": 100.0}, max_elements=10**7)
    assert prepared.model_image is not None and np.array_equal(prepared.model_image, grey)
    assert prepared.raw_image is None
    assert WARN_IMAGE_FULL not in prepared.warnings
    assert prepared.metadata["time_window_ns"] == {"value": 100.0, "source": "user"}


def test_raw_record_is_resized_like_gpradar_07a():
    """The bundled scene reproduces 07a's stored ``processed_u8_224`` (golden input)."""

    path = SAMPLES_DIR / "gprmax_field_400mhz_seed182.npz"
    with np.load(path, allow_pickle=False) as archive:
        expected = archive["processed_u8_224"]
    prepared = prepare(read_scan_npz(path, Settings()), "full", {}, max_elements=10**7)
    assert prepared.model_image.shape == expected.shape == (224, 224)
    diff = np.abs(prepared.model_image.astype(int) - expected.astype(int))
    # 07a processed the float64 record; the archive stores it as float32.
    assert diff.max() <= 1 and diff.mean() < 0.01


def test_long_record_is_squeezed_to_224_square_not_tiled():
    data = np.tile(synthetic_bscan(), (1, 16))  # 1024 traces
    prepared = prepare(scan_input(data), "full", {}, max_elements=10**7)
    assert prepared.model_image.shape == (224, 224)
    assert prepared.n_traces == 1024

