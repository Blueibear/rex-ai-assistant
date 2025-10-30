# Security Defaults & Best Practices

**Last Updated:** 2025-10-29
**Status:** Production Recommendations

---

## Overview

This document outlines security defaults, recommended practices, and hardening options for deploying Rex AI Assistant in production environments.

---

## HTTP API Security (`rex-speak-api`, `flask_proxy`)

### Current Security Measures ✅

1. **CORS Restrictions**
   - Default allowlist: `http://localhost:3000,http://localhost:5000`
   - Configured via: `REX_ALLOWED_ORIGINS`
   - Wildcard (`*`) no longer used

2. **Rate Limiting**
   - Configured via `flask-limiter`
   - Default: `30/minute` per endpoint
   - Configured via: `REX_RATE_LIMIT`

3. **Security Headers**
   - Flask 3.0.0+ includes improved security defaults
   - werkzeug>=3.0.0 includes CVE fixes

### Recommended Production Settings

#### 1. API Key Authentication
**Status:** Optional (should be required in production)

**Implementation:**
```python
# Set environment variable
REX_SPEAK_API_KEY=your-secret-key-here

# In rex_speak_api.py (recommended addition):
@app.before_request
def check_api_key():
    if os.getenv("REX_REQUIRE_API_KEY", "false").lower() == "true":
        provided_key = request.headers.get("X-API-Key")
        expected_key = os.getenv("REX_SPEAK_API_KEY")
        if not expected_key or provided_key != expected_key:
            abort(401, "Invalid or missing API key")
```

**Configuration:**
```bash
# Enable API key requirement
export REX_REQUIRE_API_KEY=true
export REX_SPEAK_API_KEY=$(openssl rand -hex 32)
```

#### 2. Strict Rate Limiting
**Current:** `30/minute`
**Production:** Adjust based on expected usage

```bash
# Conservative rate limit for production
export REX_RATE_LIMIT="10/minute"

# Or per-endpoint granular limits
export REX_RATE_LIMIT_SPEAK="5/minute"
export REX_RATE_LIMIT_TRANSCRIBE="10/minute"
```

#### 3. CORS Allowlist
**Current:** Localhost only
**Production:** Specify exact origins

```bash
# Production CORS (no wildcards)
export REX_ALLOWED_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"

# For internal services only
export REX_ALLOWED_ORIGINS="https://internal.company.net"
```

#### 4. HTTPS Only
**Recommendation:** Deploy behind reverse proxy (nginx, Caddy) with TLS

```nginx
# nginx example
server {
    listen 443 ssl http2;
    server_name rex-api.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # Strong SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### 5. Request Timeouts
**Current:** Flask defaults
**Production:** Enforce strict timeouts

```bash
# Limit request processing time
export REX_REQUEST_TIMEOUT=30  # seconds

# Limit payload size
export REX_MAX_CONTENT_LENGTH=10485760  # 10MB
```

**Implementation (recommended addition):**
```python
# In rex_speak_api.py
from werkzeug.exceptions import RequestTimeout

app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("REX_MAX_CONTENT_LENGTH", 10 * 1024 * 1024))

@app.before_request
def timeout_handler():
    request.environ.setdefault("werkzeug.socket_timeout",
                               int(os.getenv("REX_REQUEST_TIMEOUT", 30)))
```

---

## Secrets Management

### Environment Variables (Current) ✅

**Best Practice:** Use `.env` file locally, environment injection in production

```bash
# .env file (DO NOT COMMIT)
OPENAI_API_KEY=sk-...
REX_SPEAK_API_KEY=...
BRAVE_API_KEY=...
```

**Gitignore:** ✅ `.env` excluded

### Production Secrets Management

#### Option 1: Docker Secrets
```bash
# Create secrets
echo "sk-..." | docker secret create openai_api_key -

# Use in docker-compose.yml
services:
  rex:
    secrets:
      - openai_api_key
    environment:
      - OPENAI_API_KEY_FILE=/run/secrets/openai_api_key
```

#### Option 2: Kubernetes Secrets
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: rex-secrets
type: Opaque
stringData:
  openai-api-key: "sk-..."
  rex-api-key: "..."
---
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: rex
        envFrom:
        - secretRef:
            name: rex-secrets
```

#### Option 3: HashiCorp Vault
```bash
# Store secret
vault kv put secret/rex openai_api_key="sk-..."

# Retrieve in application
vault kv get -field=openai_api_key secret/rex
```

---

## Network Security

### Firewall Rules
**Recommendation:** Restrict API access to known networks

```bash
# iptables example (Linux)
iptables -A INPUT -p tcp --dport 5000 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 5000 -j DROP

# ufw example (Ubuntu)
ufw allow from 10.0.0.0/8 to any port 5000
ufw deny 5000
```

### Service Isolation
**Recommendation:** Run in isolated network namespace

```yaml
# docker-compose.yml
services:
  rex:
    networks:
      - internal
    # Don't expose directly to host

  nginx:
    networks:
      - internal
      - public
    ports:
      - "443:443"

networks:
  internal:
    internal: true
  public:
```

---

## Input Validation

### Current Protections ✅

1. **Plugin Safety** (`rex/plugins/__init__.py`)
   - Timeout enforcement (30s default)
   - Output size limits (1MB)
   - TTS text sanitization (control character removal)
   - Rate limiting (10 req/min per plugin)

2. **Path Traversal Prevention** (`config.py`)
   - LLM model path validation
   - Rejects `..` and absolute paths

3. **Audio Input Limits**
   - Capture duration capped (5s default)
   - Sample rate validated

### Additional Recommendations

#### 1. Strict Input Schemas
```python
# Add to API endpoints
from pydantic import BaseModel, Field, validator

class SpeakRequest(BaseModel):
    text: str = Field(..., max_length=10000)
    language: str = Field(default="en", regex="^[a-z]{2}$")

    @validator("text")
    def validate_text(cls, v):
        if len(v.strip()) == 0:
            raise ValueError("Text cannot be empty")
        return v.strip()
```

#### 2. WAV/Audio Validation
```python
# Validate audio file properties
import soundfile as sf

def validate_audio(file_path, max_duration=30, max_size=10*1024*1024):
    # Check file size
    if file_path.stat().st_size > max_size:
        raise ValueError(f"Audio file too large (max {max_size/1024/1024}MB)")

    # Check duration
    info = sf.info(file_path)
    if info.duration > max_duration:
        raise ValueError(f"Audio too long (max {max_duration}s)")

    # Check format
    if info.format not in ("WAV", "FLAC"):
        raise ValueError("Only WAV and FLAC formats supported")
```

---

## Dependency Security

### Current Measures ✅

1. **Pinned Minimum Versions**
   - Security-critical packages have explicit minimum versions
   - See `pyproject.toml` and `requirements.txt`

2. **Known Vulnerabilities Fixed**
   - 16 CVEs addressed (12 fixed, 2 documented, 2 transitive)
   - See `SECURITY_ADVISORY.md` for details

3. **Regular Audits**
   - Run `pip-audit` before releases
   - Address HIGH and MODERATE severity issues

### Ongoing Maintenance

#### Automated Scanning
```bash
# Install pip-audit
pip install pip-audit

# Run audit
pip-audit

# Check for outdated packages
pip list --outdated

# Update to latest secure versions
pip install --upgrade cryptography setuptools pip
```

#### Dependabot Integration
✅ **Enabled** on GitHub
- Monitors for security updates
- Creates automated PRs for dependency upgrades

---

## Logging & Monitoring

### Current Logging ✅
- Container-friendly (stdout by default)
- File logging opt-in via `REX_FILE_LOGGING_ENABLED`
- Sensitive data not logged (API keys, user audio)

### Production Recommendations

#### 1. Structured Logging
```python
# Add to logging_utils.py
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if hasattr(record, "context_id"):
            log_data["context_id"] = record.context_id
        return json.dumps(log_data)
```

#### 2. Audit Logging
```python
# Log security-relevant events
AUDIT_EVENTS = {
    "api_key_invalid": "Invalid API key attempt from {ip}",
    "rate_limit_exceeded": "Rate limit exceeded for {endpoint} from {ip}",
    "config_loaded": "Configuration loaded from {source}",
    "user_switch": "Active user changed from {old_user} to {new_user}",
}

def audit_log(event, **kwargs):
    logger.warning(f"AUDIT: {AUDIT_EVENTS[event].format(**kwargs)}", extra={"audit": True})
```

#### 3. Monitoring Endpoints
```python
# Health check endpoint (no auth required)
@app.route("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}

# Metrics endpoint (auth required)
@app.route("/metrics")
@require_api_key
def metrics():
    return {
        "requests_total": request_counter.value,
        "errors_total": error_counter.value,
        "active_connections": len(active_connections),
    }
```

---

## Deployment Checklist

### Before Production

- [ ] Change default API keys and secrets
- [ ] Enable API key authentication (`REX_REQUIRE_API_KEY=true`)
- [ ] Configure strict CORS allowlist (no wildcards)
- [ ] Set conservative rate limits
- [ ] Deploy behind HTTPS reverse proxy
- [ ] Configure firewall rules
- [ ] Enable audit logging
- [ ] Set up monitoring/alerting
- [ ] Test with `pip-audit` and address vulnerabilities
- [ ] Review and restrict file system permissions
- [ ] Configure log rotation (if using file logging)
- [ ] Document incident response procedures
- [ ] Test backup/restore procedures

### Regular Maintenance

- [ ] Weekly: Review logs for suspicious activity
- [ ] Monthly: Update dependencies (`pip-audit`, check Dependabot PRs)
- [ ] Quarterly: Security audit and penetration testing
- [ ] Annually: Review and update security documentation

---

## Incident Response

### Security Issue Reporting
Report security vulnerabilities privately to: **[security email or GitHub Security Advisories]**

### Response Procedure
1. **Identify**: Log analysis, error monitoring
2. **Contain**: Rate limit, block IPs, disable compromised endpoints
3. **Eradicate**: Update dependencies, patch vulnerabilities
4. **Recover**: Restore from known-good state, rotate secrets
5. **Document**: Post-mortem, update procedures

---

## References

- [SECURITY_ADVISORY.md](./SECURITY_ADVISORY.md) - Vulnerability tracking
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CIS Docker Benchmarks](https://www.cisecurity.org/benchmark/docker)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

---

**Note:** This document represents recommended practices. Actual security requirements vary by deployment environment and risk profile. Consult with security professionals for high-risk deployments.
