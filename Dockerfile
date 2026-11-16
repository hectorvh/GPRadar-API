# GPRadar-API (plan 2.8). Sized for Render free: 512 MB RAM, 0.1 CPU.
#
#   docker build --build-arg HF_REVISION=<40-char HF commit> -t gpradar-api .
#   docker run --rm -p 8000:8000 -m 512m --cpus 0.1 -e ALLOWED_ORIGINS=http://localhost:5173 gpradar-api
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 HF_HUB_DISABLE_TELEMETRY=1 \
    MALLOC_ARENA_MAX=2
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the model into the image, pinned to one HF commit (edit when you ship a new model)
ARG HF_MODEL_REPO=hectorvh/gpradar-hyperbola-yolo
ARG HF_MODEL_FILE=best.onnx
ARG HF_REVISION=REPLACE_WITH_HF_COMMIT_HASH
# Optional: fail the build if the downloaded file has another SHA-256.
ARG MODEL_SHA256=
RUN case "$HF_REVISION" in REPLACE_*|"") echo "set --build-arg HF_REVISION=<commit hash>" >&2; exit 1;; esac \
 && python -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('${HF_MODEL_REPO}', '${HF_MODEL_FILE}', revision='${HF_REVISION}'), '/app/model.onnx')" \
 && rm -rf /root/.cache/huggingface \
 && if [ -n "$MODEL_SHA256" ]; then echo "$MODEL_SHA256  /app/model.onnx" | sha256sum -c -; fi
ENV MODEL_PATH=/app/model.onnx MODEL_REPO=${HF_MODEL_REPO} MODEL_FILE=${HF_MODEL_FILE} MODEL_REVISION=${HF_REVISION}

COPY app ./app
COPY LICENSE README.md ./
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
# --no-proxy-headers: uvicorn must not rewrite the client address. With
# --proxy-headers --forwarded-allow-ips='*' it would set request.client to the
# LEFT-most X-Forwarded-For entry, which the client can forge. The app reads the
# header itself (TRUSTED_PROXY_HOPS from the right; see app/ratelimit.py).
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --no-proxy-headers"]
