# Security Policy

mediakit is a self-hosted media-processing toolkit intended to run on a trusted,
single-GPU machine. It has no built-in user accounts or multi-tenancy.

## Supported versions

This is a pre-1.0 project; only the latest `main` receives fixes.

| Version | Supported |
|---------|-----------|
| `main`  | ✅        |
| older   | ❌        |

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Instead, use GitHub's
private vulnerability reporting ("Report a vulnerability" under the repository's
**Security** tab). Include a description, reproduction steps, and the affected
version/commit. You can expect an initial response within a few days.

## Deployment notes

- **Authentication** is a single bearer token (`API_TOKEN`). If `API_TOKEN` is empty,
  auth is **disabled** — never expose such an instance to an untrusted network.
- The service executes GPU workloads via a local **ComfyUI** instance and shells out
  to **ffmpeg**; treat both, and the machine they run on, as part of the trust boundary.
- Uploads are size-capped (`STORAGE_MAX_UPLOAD_MB`, default 20 MB) but are otherwise
  written to local storage — run behind your own reverse proxy / network controls.
