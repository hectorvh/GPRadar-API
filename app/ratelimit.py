"""In-memory token bucket per client IP (one process, so no shared store)."""

from __future__ import annotations

import ipaddress
import threading
import time
from collections import OrderedDict


EDGE_HEADERS = ("cf-connecting-ip", "true-client-ip")


def _single_ip(value):
    """The address if ``value`` is exactly one valid IP, else None (lists are invalid)."""

    if not value or "," in value:
        return None
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError:
        return None


def _forwarded_entries(headers):
    return [item.strip() for item in headers.get("x-forwarded-for", "").split(",") if item.strip()]


def client_ip(headers, peer_host, trusted_proxy_hops=1, client_ip_header="cf-connecting-ip"):
    """Client IP and where it came from: (ip, source).

    1. ``client_ip_header`` (CF-Connecting-IP on Render): Cloudflare sits in
       front of Render and overwrites this header with the address that
       connected to it, so a client cannot forge it. It must hold exactly one
       valid IP; a list or anything else is ignored. Source "header:<name>".
    2. Fallback, X-Forwarded-For: each proxy appends the address it received
       the request from, so only the right-hand end is trustworthy. With N
       trusted proxies the client is the entry N places from the right. (On
       Render that entry is a Cloudflare edge, which is why step 1 exists.)
    3. The socket peer (``request.client.host``).

    uvicorn runs with --no-proxy-headers, so ``peer_host`` is the real TCP
    peer and this function is the only place that reads these headers.
    """

    name = (client_ip_header or "").strip().lower()
    if name:
        address = _single_ip(headers.get(name))
        if address is not None:
            return address, f"header:{name}"
    hops = int(trusted_proxy_hops)
    parts = _forwarded_entries(headers)
    if hops > 0 and len(parts) >= hops:
        try:
            return str(ipaddress.ip_address(parts[-hops])), "x-forwarded-for"
        except ValueError:
            pass
    return (peer_host or "unknown"), "peer"


def describe_client_ip(headers, peer_host, trusted_proxy_hops, client_ip_header, source):
    """One log line about where the client IP came from. Never contains an address.

    Lists the chosen source, the number of X-Forwarded-For entries, whether the
    edge headers are present, and the ``ip_class`` of every candidate.
    """

    def header_class(value):
        if value is None:
            return "absent"
        if "," in value:
            return "list"
        return ip_class(value.strip())

    names = list(EDGE_HEADERS)
    configured = (client_ip_header or "").strip().lower()
    if configured and configured not in names:
        names.insert(0, configured)
    parts = _forwarded_entries(headers)
    hops = int(trusted_proxy_hops)
    hop_entry = parts[-hops] if hops > 0 and len(parts) >= hops else None
    fields = [
        f"source={source}",
        f"header={configured or 'off'}",
        f"x-forwarded-for entries={len(parts)}",
        *(f"{name}={'present' if headers.get(name) is not None else 'absent'}" for name in names),
        *(f"class[{name}]={header_class(headers.get(name))}" for name in names),
        f"class[x-forwarded-for -{hops}]={'absent' if hop_entry is None else ip_class(hop_entry)}",
        f"class[x-forwarded-for left-most]={ip_class(parts[0]) if parts else 'absent'}",
        f"class[peer]={ip_class(peer_host or '')}",
    ]
    return "client IP: " + ", ".join(fields)


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
