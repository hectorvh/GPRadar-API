"""ONNX session, loaded once at startup, and the model's identity."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import onnxruntime as ort


@dataclass(frozen=True)
class LoadedModel:
    session: ort.InferenceSession
    repo: str
    file: str
    revision: str
    sha256: str

    def info(self):
        return {"repo": self.repo, "file": self.file, "revision": self.revision, "sha256": self.sha256}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def open_session(path, threads=1):
    """CPU session tuned for a 0.1-CPU, 512 MB host: one thread, no memory arena."""

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_cpu_mem_arena = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def load_model(settings):
    path = Path(settings.model_path)
    if not path.is_file():
        raise RuntimeError(
            f"model file not found at {path}; set MODEL_PATH or build the Docker image"
        )
    return LoadedModel(
        session=open_session(path, settings.onnx_threads),
        repo=settings.model_repo,
        file=settings.model_file,
        revision=settings.model_revision,
        sha256=sha256_file(path),
    )
