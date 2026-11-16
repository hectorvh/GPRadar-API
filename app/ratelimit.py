"""In-memory token bucket per client IP (one process, so no shared store)."""

from __future__ import annotations

import ipaddress
import threading
import time
from collections import OrderedDict


def client_ip(headers, peer_host, trusted_proxy_hops=1):
    """Client IP and where it came from: (ip, source).

    Each proxy appends the address it received the request from to
    X-Forwarded-For, so only the right-hand end of the header is trustworthy:
    anything to the left may have been sent by the client. With N trusted
    proxies in front of the app, the client is the entry N places from the
    right. If the header is missing, too short or that entry is not an IP
    address, the socket peer (``request.client.host``) is used.

    uvicorn runs with --no-proxy-headers, so ``peer_host`` is the real TCP
    peer and this function is the only place that reads the header.
    """

    hops = int(trusted_proxy_hops)
    parts = [item.strip() for item in headers.get("x-forwarded-for", "").split(",") if item.strip()]
    if hops > 0 and len(parts) >= hops:
        try:
            return str(ipaddress.ip_address(parts[-hops])), "x-forwarded-for"
        except ValueError:
            pass
    return (peer_host or "unknown"), "peer"


def ip_class(ip):
    """Coarse class of an address for logs: public, private, loopback, ... (never the IP)."""

    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return "not-an-ip"
    for name in ("loopback", "private", "link_local", "reserved", "multicast"):
        if getattr(address, f"is_{name}"):
            return name.replace("_", "-")
    return f"public-ipv{address.version}"


class TokenBucket:
    """``capacity`` tokens per key, refilled at ``rate_per_s``; LRU-capped key table."""

    def __init__(self, capacity, rate_per_s, max_keys=10_000, clock=time.monotonic):
        self.capacity = float(capacity)
        self.rate = float(rate_per_s)
        self.max_keys = max_keys
        self.clock = clock
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()

    def take(self, key):
        """Spend one token. Returns (allowed, seconds until the next token)."""

        now = self.clock()
        with self._lock:
            tokens, last = self._buckets.pop(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            allowed = tokens >= 1.0
            if allowed:
                tokens -= 1.0
            self._buckets[key] = (tokens, now)
            while len(self._buckets) > self.max_keys:
                self._buckets.popitem(last=False)
        retry = 0.0 if allowed else (1.0 - tokens) / self.rate if self.rate > 0 else 60.0
        return allowed, retry
