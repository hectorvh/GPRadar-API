"""Shared fixtures: the model (real if available, else a tiny dummy ONNX) and app clients."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import ROOT, Settings
from app.main import create_app

ALLOWED_ORIGIN = "https://allowed.example"
REAL_MODEL = Path(os.environ.get("MODEL_PATH", ROOT / "model" / "best.onnx"))


def write_dummy_onnx(path):
    """A YOLO-shaped ONNX: input (1,3,224,224) -> output (1,5,1029), one fixed box.

    The output does not depend on the image (it is multiplied by 0), which is
    enough to test the API plumbing without the real weights.
    """

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    pred = np.zeros((1, 5, 1029), dtype=np.float32)
    pred[0, :, 0] = [112.0, 80.0, 120.0, 100.0, 0.9]  # cx, cy, w, h, score
    graph = helper.make_graph(
        [
            helper.make_node("ReduceSum", ["images"], ["total"], keepdims=0),
            helper.make_node("Mul", ["total", "zero"], ["nothing"]),
            helper.make_node("Add", ["pred", "nothing"], ["output0"]),
        ],
        "dummy_yolo",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 224, 224])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 5, 1029])],
        initializer=[
            numpy_helper.from_array(pred, "pred"),
            numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    if REAL_MODEL.is_file():
        return REAL_MODEL
    return write_dummy_onnx(tmp_path_factory.mktemp("model") / "dummy.onnx")


@pytest.fixture(scope="session")
def real_model():
    return REAL_MODEL.is_file()


@pytest.fixture
def make_client(model_path):
    """Factory: a TestClient on a fresh app (fresh rate limiter) with overrides."""

    @contextmanager
    def factory(**overrides):
        settings = Settings(
            model_path=model_path,
            model_revision="test",
            allowed_origins=(ALLOWED_ORIGIN,),
        ).with_overrides(**overrides)
        with TestClient(create_app(settings)) as client:
            yield client

    return factory


@pytest.fixture
def client(make_client):
    with make_client() as test_client:
        yield test_client
