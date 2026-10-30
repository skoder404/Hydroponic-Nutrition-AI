"""Unit tests for reverse proxy TLS configuration, CORS, and HTTPS redirection (T9-005).

Acceptance Criteria:
- [x] Frontend and API reachable over HTTPS locally
- [x] HTTP redirects to HTTPS
- [x] Works with the CORS allow-list
"""

from __future__ import annotations

from pathlib import Path
import re
import yaml

from fastapi.testclient import TestClient
import pytest

from backend.app.main import create_app

ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client instance."""
    app = create_app(configure_logging=False)
    return TestClient(app)


def test_caddyfile_configuration_syntax_and_directives():
    """Verify deploy/Caddyfile contains required reverse proxy, TLS, HSTS, and routing rules."""
    caddyfile_path = ROOT / "deploy" / "Caddyfile"
    assert caddyfile_path.is_file(), "deploy/Caddyfile must exist"

    content = caddyfile_path.read_text(encoding="utf-8")

    # 1. TLS configuration: defaults to internal self-signed certificate for local development
    assert "tls {$TLS_MODE:internal}" in content, (
        "Caddyfile must configure TLS with internal fallback"
    )

    # 2. HSTS and Security Headers
    assert "Strict-Transport-Security" in content, "Must configure HSTS header"
    assert "X-Content-Type-Options" in content, "Must configure nosniff header"
    assert "X-Frame-Options" in content, "Must configure DENY frame header"

    # 3. Request body limit
    assert "max_size 20MB" in content, "Must cap upload size to 20MB"

    # 4. Reverse proxy routing to backend
    assert "handle /api/*" in content, "Must route /api/* to backend"
    assert "handle /health*" in content, "Must route /health* to backend"
    assert "handle /metrics*" in content, "Must route /metrics* to backend"
    assert "header_up X-Forwarded-Proto https" in content, (
        "Must pass X-Forwarded-Proto https"
    )

    # 5. Reverse proxy routing to frontend
    assert "reverse_proxy {$FRONTEND_UPSTREAM:frontend:3000}" in content, (
        "Must route fallback web UI traffic to Next.js frontend"
    )


def test_nginx_conf_configuration_syntax_and_directives():
    """Verify deploy/nginx.conf contains HTTP redirect, TLS termination, and upstream proxying."""
    nginx_path = ROOT / "deploy" / "nginx.conf"
    assert nginx_path.is_file(), "deploy/nginx.conf must exist"

    content = nginx_path.read_text(encoding="utf-8")

    # 1. Plain HTTP redirect block (Port 80)
    assert "listen 80;" in content, "NGINX must listen on port 80 for HTTP"
    assert "return 301 https://$host$request_uri;" in content, (
        "NGINX must issue 301 redirect from HTTP to HTTPS"
    )

    # 2. HTTPS server block (Port 443)
    assert "listen 443 ssl;" in content, "NGINX must listen on port 443 with SSL"
    assert "Strict-Transport-Security" in content, "NGINX must enforce HSTS header"

    # 3. Upstream proxying
    assert "location /api/" in content, "NGINX must proxy /api/ to backend"
    assert "location / {" in content, "NGINX must proxy / to frontend"
    assert "proxy_set_header X-Forwarded-Proto https;" in content, (
        "NGINX must set X-Forwarded-Proto https header"
    )


def test_docker_compose_tls_override():
    """Verify deploy/docker-compose.tls.yml parses as valid compose file with proxy service."""
    tls_compose_path = ROOT / "deploy" / "docker-compose.tls.yml"
    assert tls_compose_path.is_file(), "deploy/docker-compose.tls.yml must exist"

    data = yaml.safe_load(tls_compose_path.read_text(encoding="utf-8"))
    assert "services" in data, "Must define services mapping"
    assert "proxy" in data["services"], "Must configure 'proxy' service"

    proxy = data["services"]["proxy"]
    assert "ports" in proxy, "Proxy must expose ports"
    ports_str = " ".join(str(p) for p in proxy["ports"])
    assert "80:80" in ports_str, "Proxy must expose port 80 for HTTP redirect"
    assert "443:443" in ports_str, "Proxy must expose port 443 for HTTPS"

    # Must depend on backend and frontend
    depends_on = proxy.get("depends_on", {})
    assert "backend" in depends_on, "Proxy must depend on backend"
    assert "frontend" in depends_on, "Proxy must depend on frontend"


def test_cors_allow_list_with_https_origins(client: TestClient):
    """AC 3: Verify backend CORS allows HTTPS origins and handles preflight requests."""
    # Test HTTPS localhost origin
    https_origin = "https://localhost:3000"
    resp = client.get("/api/v1/health", headers={"Origin": https_origin})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == https_origin
    assert resp.headers.get("access-control-allow-credentials") == "true"

    # Test HTTPS loopback IP origin
    https_ip_origin = "https://127.0.0.1:3000"
    resp_ip = client.get("/api/v1/health", headers={"Origin": https_ip_origin})
    assert resp_ip.status_code == 200
    assert resp_ip.headers.get("access-control-allow-origin") == https_ip_origin

    # Test HTTPS campus LAN IP origin
    https_lan_origin = "https://192.168.1.100:3000"
    resp_lan = client.get("/api/v1/health", headers={"Origin": https_lan_origin})
    assert resp_lan.status_code == 200
    assert resp_lan.headers.get("access-control-allow-origin") == https_lan_origin

    # Test CORS preflight (OPTIONS) request from HTTPS origin
    preflight = client.options(
        "/api/v1/analyze",
        headers={
            "Origin": https_origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type, X-Request-ID",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers.get("access-control-allow-origin") == https_origin
    assert "POST" in preflight.headers.get("access-control-allow-methods", "")

    # Test disallowed external origin
    untrusted_origin = "https://evil-untrusted-domain.com"
    resp_untrusted = client.get("/api/v1/health", headers={"Origin": untrusted_origin})
    assert resp_untrusted.status_code == 200
    assert resp_untrusted.headers.get("access-control-allow-origin") != untrusted_origin


def test_https_redirection_and_hsts_behavior():
    """AC 2: Verify HTTP-to-HTTPS redirect logic and HSTS headers matching https.md."""
    # Verify NGINX redirect rule
    nginx_content = (ROOT / "deploy" / "nginx.conf").read_text(encoding="utf-8")
    assert re.search(r"return\s+301\s+https://", nginx_content) is not None, (
        "NGINX must issue 301 HTTPS redirection"
    )

    # Verify Caddy Auto-HTTPS behavior
    caddy_content = (ROOT / "deploy" / "Caddyfile").read_text(encoding="utf-8")
    assert "Strict-Transport-Security" in caddy_content, (
        "Caddyfile must include HSTS directive"
    )
    # Auto-HTTPS is enabled by default in Caddy unless 'auto_https off' is set
    assert "auto_https off" not in caddy_content, (
        "Auto-HTTPS redirection must remain active"
    )
