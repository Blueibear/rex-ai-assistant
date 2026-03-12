# Final Security Scan Report — US-131

**Scan date:** 2026-03-12
**Tools:** pip-audit 2.10.0, detect-secrets 1.5.0
**Scope:** Full project virtual environment and repository

---

## 1. Dependency Vulnerability Scan (pip-audit)

**Result: PASS — zero vulnerabilities in directly installed packages.**

```
pip-audit --format json
Found 0 known vulnerabilities in 6 packages
```

All critical and high CVEs identified in previous scans (US-093) were
remediated by raising minimum version pins in `pyproject.toml`:

| Package | Previous version | Fixed version | CVE(s) fixed |
|---------|-----------------|---------------|-------------|
| cryptography | 44.0.1 | >=46.0.5 | CVE-2026-26007 |
| flask | 3.1.1 | >=3.1.3 | CVE-2026-27205 |
| nltk | 3.9.2 | >=3.9.3 | CVE-2025-14009 |
| pillow | 11.1.0 | >=12.1.1 | CVE-2026-25990 |
| protobuf | 6.33.1 | >=6.33.5 | CVE-2026-0994 |
| werkzeug | 3.1.5 | >=3.1.6 | CVE-2026-27199 |

See `docs/security/VULNERABILITY-SCAN.md` for full CVE details, accepted-risk
transitive dependencies, and CI integration guidance.

---

## 2. Secret Scan (detect-secrets)

**Result: PASS — zero confirmed real secrets found.**

```
python -m detect_secrets scan --baseline .secrets.baseline
# Exit code: 0 — no new findings beyond baseline
```

All 83 tracked findings in `.secrets.baseline` are confirmed false positives
(test fixture values, documentation examples, placeholder templates).

See `docs/security/SECRET-SCAN.md` for the full false-positive review.

---

## 3. Security Headers

**Result: PASS — required headers present in all API and dashboard responses.**

Headers verified via the Flask test client against the dashboard blueprint:

| Header | Value | Status |
|--------|-------|--------|
| X-Frame-Options | DENY | Present |
| X-Content-Type-Options | nosniff | Present |
| Content-Security-Policy | `default-src 'self'; ...` | Present on HTML |
| Strict-Transport-Security | `max-age=31536000; includeSubDomains` | Present on HTTPS |

Source: `rex/dashboard/routes.py:94` — `add_security_headers()` applied via
`@dashboard_bp.after_request`.

For live verification with `curl`:
```bash
curl -I http://localhost:5000/health/live
# Expected headers: X-Frame-Options, X-Content-Type-Options
```

---

## 4. Findings Summary and Remediation Status

| Finding | Severity | Status | Remediation |
|---------|----------|--------|-------------|
| CVE-2026-26007 (cryptography SECT curves) | High | Fixed | Upgraded to >=46.0.5 |
| CVE-2026-27205 (flask Vary header) | Medium | Fixed | Upgraded to >=3.1.3 |
| CVE-2025-14009 (nltk downloader RCE) | Critical | Fixed | Upgraded to >=3.9.3 |
| CVE-2026-25990 (pillow PSD OOB write) | High | Fixed | Upgraded to >=12.1.1 |
| CVE-2026-0994 (protobuf DoS) | Medium | Fixed | Upgraded to >=6.33.5 |
| CVE-2026-27199 (werkzeug Windows device) | Medium | Fixed | Upgraded to >=3.1.6 |
| Transitive dep CVEs (aiohttp, brotli, etc.) | Various | Accepted risk | Documented in VULNERABILITY-SCAN.md |
| Hardcoded secrets (83 baseline entries) | N/A | False positives | All confirmed non-sensitive in SECRET-SCAN.md |

---

## 5. Scan Commands Reference

```bash
# Vulnerability scan
python -m pip_audit --format json

# Secret scan
python -m detect_secrets scan --baseline .secrets.baseline

# Security headers (live instance)
curl -I http://localhost:5000/health/live
```
