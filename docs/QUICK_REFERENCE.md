# 🚀 Rex AI Stabilization - Quick Reference Card

## 📥 DEPLOYMENT (5 Steps, ~20 min)

### 1️⃣ Backup (2 min)
```bash
cd C:\Users\james\rex-ai-test\rex-ai-assistant
git add -A && git commit -m "pre-stabilization backup"
git tag backup-$(date +%Y%m%d)
```

### 2️⃣ Copy Files (3 min)
```
outputs/rex_assistant_errors.py  →  rex/assistant_errors.py
outputs/rex_config.py             →  rex/config.py
outputs/rex_speak_api_fixed.py   →  rex_speak_api.py
outputs/requirements.txt          →  requirements.txt
```

### 3️⃣ Update Dependencies (10 min)
```powershell
.\.venv\Scripts\activate
pip uninstall -y torch torchvision torchaudio
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 4️⃣ Configure .env (2 min)
```bash
REX_SPEAK_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
REX_ACTIVE_USER=james
REX_WAKEWORD=rex
REX_LLM_MODEL=distilgpt2
```

### 5️⃣ Validate (3 min)
```powershell
python outputs/validate_deployment.py
python scripts/doctor.py
```

---

## 🔍 TESTING ONE-LINERS

### Import Tests
```python
python -c "from rex.config import settings; print('✓ Config OK')"
python -c "from rex.assistant_errors import AssistantError; print('✓ Errors OK')"
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
```

### Component Tests
```bash
python wakeword_listener.py        # Test wake word
python rex_assistant.py             # Test CLI assistant
python rex_speak_api.py            # Test TTS API (Ctrl+C to stop)
```

### API Test (PowerShell)
```powershell
$key = $env:REX_SPEAK_API_KEY
$body = @{text="Hello world"; user="james"} | ConvertTo-Json
Invoke-WebRequest -Uri http://localhost:5000/speak -Method POST -Headers @{"X-API-Key"=$key;"Content-Type"="application/json"} -Body $body -OutFile test.wav
```

---

## 🔧 TROUBLESHOOTING

### Import Errors
```bash
# Problem: ModuleNotFoundError: No module named 'rex'
# Fix: Ensure rex/__init__.py exists
type rex\__init__.py  # Should show content
```

### Config Not Loading
```bash
# Problem: ConfigurationError or empty settings
# Fix: Check .env format
cat .env  # On Linux/Mac
type .env  # On Windows
```

### PyTorch CUDA Issues
```bash
# Problem: CUDA not detected
# Fix: Reinstall with CUDA wheels
pip cache purge
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.cuda.is_available())"
```

### API Key Fails
```bash
# Problem: 401 Unauthorized
# Fix: Regenerate key and update .env
python -c "import secrets; print(secrets.token_urlsafe(32))"
# Copy output to .env: REX_SPEAK_API_KEY=<output>
```

---

## 🎯 KEY CHANGES AT A GLANCE

| Issue | Before | After |
|-------|--------|-------|
| **Imports** | `from config import` | `from rex.config import` |
| **API Keys** | `key == API_KEY` | `hmac.compare_digest(key, API_KEY)` |
| **PyTorch** | `torch==2.8.0` ❌ | `torch==2.5.1` ✅ |
| **Config** | 2 systems ❌ | 1 unified ✅ |
| **Paths** | No validation ❌ | Sanitized ✅ |

---

## 📂 FILE LOCATIONS

```
C:\Users\james\rex-ai-test\rex-ai-assistant\
├── rex/
│   ├── assistant_errors.py    ← REPLACE
│   └── config.py               ← REPLACE
├── rex_speak_api.py           ← REPLACE
├── requirements.txt            ← REPLACE
├── .env                        ← CREATE/UPDATE
└── .gitignore                  ← VERIFY (.env listed)
```

---

## 🔒 SECURITY CHECKLIST

- [ ] `.env` in `.gitignore`
- [ ] `REX_SPEAK_API_KEY` generated (32+ chars)
- [ ] No API keys in Python files
- [ ] `hmac.compare_digest()` used for validation
- [ ] Path traversal prevention active

---

## ⚡ QUICK COMMANDS

### Start Services
```bash
# TTS API
python rex_speak_api.py

# CLI Chat
python rex_assistant.py

# Voice Loop
python rex_loop.py
```

### Generate Secure Key
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Check Status
```bash
python outputs/validate_deployment.py
python scripts/doctor.py
pytest tests/ -v --tb=short
```

### View Config
```bash
python -c "from rex.config import settings; import json; print(json.dumps(settings.dict(), indent=2))"
```

---

## 📞 SUPPORT

### Logs
```
logs/rex.log        # Info logs
logs/error.log      # Error logs
```

### Diagnostics
```bash
python scripts/doctor.py                    # System check
python outputs/validate_deployment.py       # Post-deploy validation
```

### Documentation
- `README_STABILIZATION.md` - Full technical details
- `DEPLOYMENT_CHECKLIST.md` - Step-by-step guide
- `FINAL_DELIVERY_SUMMARY.txt` - Executive summary

---

## 🚨 ROLLBACK

### If deployment fails:
```bash
git reset --hard backup-20251016  # Use your tag
git clean -fd
```

---

## ✅ SUCCESS INDICATORS

- [ ] `python rex_assistant.py` starts without errors
- [ ] `python scripts/doctor.py` shows no critical errors
- [ ] `pytest` runs (even if some tests fail)
- [ ] Flask API responds to `/health` endpoint
- [ ] No `ModuleNotFoundError` exceptions

---

## 🎓 LEARNING POINTS

1. **Always use `hmac.compare_digest()` for secrets**
2. **Single config system = clarity**
3. **Version pinning prevents surprises**
4. **Zero-dependency exceptions break import loops**
5. **Sanitize all file paths**

---

**Version:** 1.0
**Updated:** 2025-10-16
**Print:** Keep handy during deployment!
