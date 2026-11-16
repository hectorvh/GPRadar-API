"""Call /health and /run (each sample) on a running server and print the results.

    python scripts/smoke.py http://localhost:8000

Prints the first 600 characters of each response, the status and the
client-side wall time. Standard library only.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
import uuid


def call(method, url, body=None, headers=None):
    request = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            status, text = response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        status, text = exc.code, exc.read().decode("utf-8")
    return status, text, 1000 * (time.perf_counter() - start)


def multipart(fields):
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    base = (argv[0] if argv else "http://localhost:8000").rstrip("/")
    status, text, ms = call("GET", f"{base}/health")
    print(f"GET /health -> {status} in {ms:.0f} ms\n{text[:600]}\n")
    status, text, ms = call("GET", f"{base}/samples")
    samples = json.loads(text)
    for sample in samples:
        body, headers = multipart({"sample_id": sample["id"]})
        status, text, ms = call("POST", f"{base}/run", body, headers)
        timings = json.loads(text).get("timings_ms") if status == 200 else None
        print(f"POST /run sample_id={sample['id']} -> {status} in {ms:.0f} ms (server timings_ms {timings})")
        print(text[:600] + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
