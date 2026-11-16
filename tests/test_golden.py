"""Golden test: the API reproduces the research code's detections (needs the real model).

tests/fixtures/golden.json was written by GPRadar src/07a_export_demo_samples.py
with the 06f ONNX path. For every sample the API must give the same number of
boxes at conf >= 0.25, each matching a golden box with IoU >= 0.9.
"""

import json
from pathlib import Path

import numpy as np
import pytest

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "golden.json").read_text(encoding="utf-8"))
MIN_IOU = 0.9


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@pytest.fixture
def golden_client(make_client, real_model):
    if not real_model:
        pytest.skip("real best.onnx not available (set MODEL_PATH or put it in model/best.onnx)")
    with make_client() as client:
        sha = client.get("/health").json()["model"]["sha256"]
        assert sha == GOLDEN["model_sha256"], "golden.json was made with a different best.onnx"
        yield client


@pytest.mark.parametrize("sample", GOLDEN["samples"], ids=[s["id"] for s in GOLDEN["samples"]])
def test_golden_boxes(golden_client, sample):
    response = golden_client.post("/run", data={"sample_id": sample["id"]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert [body["input"]["width_px"], body["input"]["height_px"]] == sample["input_size_wh"]
    got = [d for d in body["detections"] if d["confidence"] >= GOLDEN["conf"]]
    assert len(got) == len(sample["boxes"])
    free = list(range(len(got)))
    for want in sample["boxes"]:
        ious = [iou(want["xyxy"], [got[j]["box_px"][k] for k in ("x_min", "y_min", "x_max", "y_max")]) for j in free]
        best = int(np.argmax(ious))
        assert ious[best] >= MIN_IOU
        assert got[free[best]]["confidence"] == pytest.approx(want["score"], abs=0.01)
        free.pop(best)


def test_mendeley_sample_detects_the_known_hyperbola(golden_client):
    """At least one detection >= 0.25 near the hand-labelled box (YOLO label file)."""

    sample_id = "mendeley_utility_0028"
    label = (Path(__file__).resolve().parents[1] / "app" / "samples" / f"{sample_id}.txt").read_text().split()
    cx, cy, w, h = (float(v) for v in label[1:5])
    body = golden_client.post("/run", data={"sample_id": sample_id}).json()
    width, height = body["input"]["width_px"], body["input"]["height_px"]
    truth = [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height]
    hits = [
        d for d in body["detections"]
        if d["confidence"] >= 0.25
        and iou(truth, [d["box_px"][k] for k in ("x_min", "y_min", "x_max", "y_max")]) >= 0.5
    ]
    assert hits
