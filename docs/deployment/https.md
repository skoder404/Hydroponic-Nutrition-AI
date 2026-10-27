# HTTPS / TLS Deployment Guide

> **Security Rule:** Plain HTTP (`http://`) is strictly permitted **only** for local machine development on `localhost` or `127.0.0.1`. Any campus network, staging, or production deployment **must** terminate TLS via a reverse proxy with HTTP Strict Transport Security (HSTS) enabled ([`TECH_RULES.md`](../TECH_RULES.md#security-rules)).

---

## 1. Localhost-Only HTTP Exception

During local software development, running services over plain HTTP on loopback interfaces (`127.0.0.1`, `localhost`) is permitted to avoid self-signed certificate friction during developer testing:

- **Frontend:** `http://localhost:3000` (Next.js development server)
- **Backend:** `http://127.0.0.1:8000` (FastAPI / Uvicorn)

### Strict Non-Localhost Mandate
Any deployment accessible outside the developer's local machine—including campus local area networks (LAN), university server racks, staging VMs, or public cloud hosts—**must** use HTTPS. Unencrypted HTTP exposes:
1. Plant camera streams and sensor telemetry to eavesdropping or on-path tampering.
2. Recommendation outputs and diagnostic history to interception.
3. Server infrastructure to man-in-the-middle (MITM) injection attacks.

---

## 2. Architecture Overview

In production or campus deployments, application servers do **not** handle raw TLS directly. Instead, a dedicated reverse proxy (such as **Caddy** or **NGINX**) terminates TLS at the boundary, enforces HSTS, proxies requests to internal services, and manages SSL/TLS certificate rotation.

```text
 Client (Browser / Mobile / Tablet)
               │
               ▼  HTTPS (Port 443) + TLS 1.3 + HSTS
   ┌──────────────────────────────────────────────┐
   │         Reverse Proxy (Caddy / NGINX)        │
   │  - Terminates TLS (Let's Encrypt / ZeroSSL)  │
   │  - Enforces HSTS (max-age=31536000)          │
   │  - Buffers & caps uploads (e.g., 20 MB)      │
   │  - Passes X-Forwarded-For & X-Forwarded-Proto│
   └───────────────┬──────────────────────────────┘
                   │
         Internal Network / Loopback (HTTP)
         ┌─────────┴─────────┐
         ▼                   ▼
  Next.js Frontend    FastAPI Backend
  (127.0.0.1:3000)    (127.0.0.1:8000)
```

---

## 3. Option A: Caddy Configuration (Recommended)

[Caddy](https://caddyserver.com/) provides automated HTTPS with automatic Let's Encrypt certificate issuance, renewal, modern TLS cipher suites, and HTTP/2 + HTTP/3 support out of the box.

Create a `Caddyfile` at your deployment root:

```caddy
# Production / Campus Domain
hydroponics.campus.edu {
    # Security Headers & Strict-Transport-Security (HSTS)
    header {
        # Enforce HTTPS for 1 year including all subdomains
        Strict-Transport-Security "max-age=31536000; includeSubDomains; preload"
        # Prevent MIME type sniffing
        X-Content-Type-Options "nosniff"
        # Clickjacking defense
        X-Frame-Options "DENY"
        # Referrer policy
        Referrer-Policy "strict-origin-when-cross-origin"
    }

    # Upload size limit matching MAX_UPLOAD_MB=15 (allow up to 20MB envelope)
    request_body {
        max_size 20MB
    }

    # Route API requests to FastAPI backend
    handle /api/* {
        reverse_proxy 127.0.0.1:8000 {
            header_up Host {host}
            header_up X-Real-IP {remote_host}
            header_up X-Forwarded-For {remote_host}
            header_up X-Forwarded-Proto https
        }
    }

    # Route OpenAPI docs to FastAPI backend (if exposed)
    handle /docs* {
        reverse_proxy 127.0.0.1:8000
    }
    handle /openapi.json {
        reverse_proxy 127.0.0.1:8000
    }

    # Route all other web UI traffic to Next.js frontend
    handle {
        reverse_proxy 127.0.0.1:3000 {
            header_up Host {host}
            header_up X-Real-IP {remote_host}
            header_up X-Forwarded-For {remote_host}
            header_up X-Forwarded-Proto https
        }
    }

    # Access and error logging
    log {
        output file /var/log/caddy/hydroponics_access.log
        format json
    }
}
```

---

## 4. Option B: NGINX Configuration

For environments standardizing on [NGINX](https://nginx.org/), use the following virtual host configuration (e.g. `/etc/nginx/sites-available/hydroponics`):

```nginx
# Redirect all plain HTTP traffic to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name hydroponics.campus.edu;

    # Certbot ACME webroot challenge for TLS renewal
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    location / {
        return 301 https://$host$request_uri;
    }
}

# Main HTTPS Server Block
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name hydroponics.campus.edu;

    # TLS Certificate Paths (issued via Certbot / Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/hydroponics.campus.edu/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/hydroponics.campus.edu/privkey.pem;

    # Hardened TLS Protocols & Ciphers (TLS 1.2 and TLS 1.3 only)
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384';
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;
    ssl_session_tickets off;

    # HTTP Strict Transport Security (HSTS) - 1 Year Duration
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Upload size allowance (aligns with backend MAX_UPLOAD_MB=15)
    client_max_body_size 20M;

    # FastAPI Backend Reverse Proxy
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;

        # WebSocket support (for progress bars or live updates)
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Request timeouts
        proxy_connect_timeout 10s;
        proxy_read_timeout 60s;
        proxy_send_timeout 60s;
    }

    # API Documentation & OpenAPI schema
    location ~ ^/(docs|redoc|openapi.json) {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
    }

    # Next.js Frontend Reverse Proxy
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;

        # Next.js Fast Refresh / WebSocket support
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Logging
    access_log /var/log/nginx/hydroponics_access.log combined;
    error_log /var/log/nginx/hydroponics_error.log warn;
}
```

---

## 5. Verification Checklist

Before opening campus or external traffic to the server, verify the deployment against this checklist:

1. [ ] **HTTP Redirect:** `curl -I http://hydroponics.campus.edu` returns `301` / `308` redirecting to `https://`.
2. [ ] **HSTS Header Present:** `curl -sI https://hydroponics.campus.edu` contains `strict-transport-security: max-age=31536000; includeSubDomains; preload`.
3. [ ] **Valid Certificate:** Browser shows a green padlock or `curl -v https://hydroponics.campus.edu` verifies without `--insecure`.
4. [ ] **Backend Health Check:** `curl -s https://hydroponics.campus.edu/api/v1/health` returns HTTP 200 with `status: ok`.
5. [ ] **Upload Size Allowance:** 15 MB leaf photo upload succeeds; 25 MB photo returns `413 Payload Too Large` from proxy or backend.
