"""Request checks shared by the notetaker server (services/api.py) and the recorder agent
(services/agent.py). There is no sign-in yet, so these are what keep the pilot private:

  network   the client is this computer or a Tailscale device (100.64.0.0/10,
            fd7a:115c:a1e0::/48; more with NOTETAKER_ALLOW="cidr,...")
  Host      the request is addressed to this machine by a name it really has: localhost, an
            allowed IP, a Tailscale MagicDNS name (*.ts.net) or this PC's hostname. Blocks
            DNS rebinding, where a web page re-points its own domain at 127.0.0.1.
  Origin    a state-changing request (POST/PUT/PATCH/DELETE) from a page on another site is
            refused (Sec-Fetch-Site / Origin), so a malicious page can't stop or delete a
            recording, or upload, through your browser.
  size      uploads must say how big they are, and at most MAX_UPLOAD bytes, checked before
            the body is read (the framework would otherwise spool all of it to disk first).

Tests use FastAPI's TestClient (client "testclient", host "testserver"); those names are
accepted only when NOTETAKER_TESTING=1 (set by tests/conftest.py).
"""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlsplit

from starlette.requests import Request
from starlette.responses import JSONResponse

ALLOW = [ipaddress.ip_network(n) for n in
         ["127.0.0.0/8", "::1/128", "100.64.0.0/10", "fd7a:115c:a1e0::/48",
          *filter(None, os.environ.get("NOTETAKER_ALLOW", "").split(","))]]
MAX_UPLOAD = 2 * 1024**3 + 16 * 1024**2          # 2 GB file + form overhead
UPLOAD_PATHS = ("/api/meetings/upload", "/api/recordings/")
STATE_CHANGING = {"POST", "PUT", "PATCH", "DELETE"}


def testing() -> bool:
    return os.environ.get("NOTETAKER_TESTING") == "1"


def _ip_ok(host: str) -> bool:
    try:
        return any(ipaddress.ip_address(host.strip("[]")) in n for n in ALLOW)
    except ValueError:
        return False


def client_allowed(host: str) -> bool:
    """The connecting peer: this computer or a Tailscale device."""
    return (testing() and host == "testclient") or _ip_ok(host)


def _names() -> set[str]:
    names = {"localhost", socket.gethostname().lower()}
    names |= {n.strip().lower() for n in os.environ.get("NOTETAKER_HOSTS", "").split(",") if n.strip()}
    return names


def host_allowed(host_header: str) -> bool:
    """The name the request was addressed to (Host header, port removed)."""
    host = host_header.strip().lower()
    if host.startswith("["):                      # [::1]:8000
        host = host[1:].split("]", 1)[0]
    elif host.count(":") == 1:
        host = host.rsplit(":", 1)[0]
    if not host:
        return False
    if testing() and host in ("testserver", "testclient"):
        return True
    return host in _names() or host.endswith(".ts.net") or _ip_ok(host)


def cross_site(request: Request) -> bool:
    if request.headers.get("sec-fetch-site", "") == "cross-site":
        return True
    origin = request.headers.get("origin")
    if not origin:
        return False                              # not a browser cross-origin request
    if origin == "null":
        return True
    return not host_allowed(urlsplit(origin).netloc)


def check(request: Request) -> JSONResponse | None:
    """None if the request may proceed, else the response refusing it."""
    if not client_allowed(request.client.host if request.client else ""):
        return JSONResponse({"detail": "not allowed from this network"}, status_code=403)
    if not host_allowed(request.headers.get("host", "")):
        return JSONResponse({"detail": "unknown host name"}, status_code=403)
    if request.method in STATE_CHANGING and cross_site(request):
        return JSONResponse({"detail": "cross-site request refused"}, status_code=403)
    if request.method == "POST" and request.url.path.startswith(UPLOAD_PATHS):
        size = request.headers.get("content-length")
        if size is None or not size.isdigit():
            return JSONResponse({"detail": "uploads must send Content-Length"}, status_code=411)
        if int(size) > MAX_UPLOAD:
            return JSONResponse({"detail": "upload too large (2 GB max)"}, status_code=413)
    return None
