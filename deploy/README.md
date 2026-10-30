# Reverse Proxy & TLS Deployment (T9-005)

This directory provides reverse proxy configurations fronting both the Next.js Frontend and FastAPI Backend with TLS termination and HTTP-to-HTTPS redirection, matching [`docs/deployment/https.md`](../docs/deployment/https.md).

---

## 1. Quick Start with Docker Compose

To run the entire hydroponics stack behind Caddy with automated local self-signed HTTPS:

```bash
docker compose -f docker-compose.yml -f deploy/docker-compose.tls.yml up -d
```

- **Frontend:** `https://localhost` (proxied to `frontend:3000`)
- **API & Health:** `https://localhost/api/v1/health` (proxied to `backend:8000`)
- **HTTP Redirection:** `http://localhost` automatically issues a `308 Permanent Redirect` to `https://localhost`.

---

## 2. Standalone Caddy (Local Testing)

If running frontend and backend directly on your host machine:

1. Install [Caddy](https://caddyserver.com/).
2. Run Caddy pointing to the local upstream services:
   ```bash
   BACKEND_UPSTREAM=127.0.0.1:8000 FRONTEND_UPSTREAM=127.0.0.1:3000 caddy run --config deploy/Caddyfile
   ```
3. Caddy automatically provisions a local self-signed certificate via its `tls internal` engine and configures loopback trust.

---

## 3. Production Deployment with Domain

In production or campus LAN:

```bash
DOMAIN=hydroponics.campus.edu TLS_MODE="" docker compose -f docker-compose.yml -f deploy/docker-compose.tls.yml up -d
```

Caddy automatically obtains and renews trusted certificates via Let's Encrypt / ZeroSSL via ACME.

---

## 4. CORS Integration

The reverse proxy terminates TLS and serves both web UI and API under the same origin (`https://localhost` or `https://hydroponics.campus.edu`).
- Same-origin API calls avoid CORS restrictions entirely.
- Direct cross-origin HTTPS requests (such as local dev frontends on `https://localhost:3000`) are handled by FastAPI's CORS regex:
  `allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+)(:\d+)?"`
- Preflight `OPTIONS` requests receive appropriate `Access-Control-Allow-Origin`, `Access-Control-Allow-Methods`, and `Access-Control-Allow-Headers` responses.
