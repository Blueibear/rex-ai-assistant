# 🚀 Rex AI Assistant - Deployment Checklist

## Pre-Deployment Validation

### ✅ Phase 1: File Preparation (5 min)

- [ ] **Backup current project**
  ```bash
  cd C:\Users\james\rex-ai-test\rex-ai-assistant
  git status
  git add -A
  git commit -m "chore: backup before stabilization v2"
  git tag pre-stabilization-backup
  ```

- [ ] **Download stabilized files**
  - Copy from Claude outputs to local system
  - Files to copy:
    - `rex_assistant_errors.py` → `rex/assistant_errors.py`
    - `rex_config.py` → `rex/config.py`
    - `requirements.txt` → `requirements.txt`
    - `rex_speak_api_fixed.py` → `rex_speak_api.py`

---

### ✅ Phase 2: Dependency Update (10 min)

- [ ] **Activate virtual environment**
  ```powershell
  cd C:\Users\james\rex-ai-test\rex-ai-assistant
  .\.venv\Scripts\activate
  ```

- [ ] **Uninstall conflicting PyTorch**
  ```powershell
  pip uninstall -y torch torchvision torchaudio
  ```

- [ ] **Install correct PyTorch (CUDA 11.8)**
  ```powershell
  pip install torch==2.5.1 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
  ```

- [ ] **Verify CUDA**
  ```powershell
  python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
  ```

- [ ] **Install requirements**
  ```powershell
  pip install --upgrade pip
  pip install -r requirements.txt
  pip install -r requirements-ml.txt
  pip install -r requirements-dev.txt
  ```

---

### ✅ Phase 3: Configuration (5 min)

- [ ] **Create/update .env file**
  ```bash
  # In project root
  notepad .env
  ```

  Minimum required variables:
  ```bash
  # Core
  REX_WAKEWORD=rex
  REX_ACTIVE_USER=james
  
  # LLM
  REX_LLM_PROVIDER=transformers
  REX_LLM_MODEL=distilgpt2
  REX_LLM_MAX_TOKENS=120
  
  # TTS API Security (REQUIRED)
  REX_SPEAK_API_KEY=<generate-secure-random-key>
  
  # Optional: OpenAI
  OPENAI_API_KEY=
  OPENAI_BASE_URL=https://api.openai.com/v1
  
  # Optional: Search
  SERPAPI_KEY=
  BRAVE_API_KEY=
  ```

- [ ] **Generate secure API key**
  ```powershell
  python -c "import secrets; print(secrets.token_urlsafe(32))"
  ```
  Copy output to `REX_SPEAK_API_KEY`

- [ ] **Verify .gitignore**
  ```bash
  # Ensure these lines exist:
  .env
  *.log
  __pycache__/
  *.pyc
  .venv/
  ```

---

### ✅ Phase 4: Import Updates (10 min)

Run global find/replace in your IDE:

- [ ] **Update import statements**
  
  | Find | Replace | Scope |
  |------|---------|-------|
  | `from assistant_errors import` | `from rex.assistant_errors import` | All files |
  | `from config import` | `from rex.config import` | All files |
  | `import config` | `from rex import config` | All files |

- [ ] **Verify no old imports remain**
  ```powershell
  # PowerShell search
  Get-ChildItem -Path . -Filter "*.py" -Recurse | Select-String -Pattern "from assistant_errors import" | Where-Object {$_.Line -notmatch "rex.assistant_errors"}
  ```

---

### ✅ Phase 5: Validation (10 min)

- [ ] **Test imports**
  ```powershell
  python -c "from rex.config import settings; print('Config OK')"
  python -c "from rex.assistant_errors import AssistantError; print('Errors OK')"
  python -c "from llm_client import LanguageModel; print('LLM OK')"
  ```

- [ ] **Run doctor script**
  ```powershell
  python scripts/doctor.py
  ```
  Expected: All checks pass or show warnings (not errors)

- [ ] **Run pytest**
  ```powershell
  pytest tests/ -v --tb=short
  ```
  Target: >80% passing (some may fail due to environment)

- [ ] **Test configuration loading**
  ```powershell
  python -c "from rex.config import settings; import json; print(json.dumps(settings.dict(), indent=2))"
  ```
  Verify: No errors, shows all config values

---

### ✅ Phase 6: Component Testing (15 min)

- [ ] **Test wake word listener**
  ```powershell
  python wakeword_listener.py
  ```
  Speak "rex" or your wake word
  Expected: Detection logged

- [ ] **Test TTS API**
  ```powershell
  # Terminal 1:
  python rex_speak_api.py
  
  # Terminal 2 (PowerShell):
  $headers = @{
      "Content-Type" = "application/json"
      "X-API-Key" = "your-key-from-env"
  }
  $body = @{
      text = "Hello, this is a test."
      user = "james"
  } | ConvertTo-Json
  
  Invoke-WebRequest -Uri http://localhost:5000/speak -Method POST -Headers $headers -Body $body -OutFile test.wav
  ```
  Expected: `test.wav` created and plays audio

- [ ] **Test CLI assistant**
  ```powershell
  python rex_assistant.py
  ```
  Type a message, check for response
  Expected: No import errors, generates reply

- [ ] **Test voice loop (optional - requires microphone)**
  ```powershell
  python rex_loop.py
  ```
  Say wake word, then speak
  Expected: Full pipeline works

---

### ✅ Phase 7: Security Audit (5 min)

- [ ] **Check for exposed credentials**
  ```powershell
  # Search for API keys in code
  Get-ChildItem -Path . -Filter "*.py" -Recurse | Select-String -Pattern "sk-[a-zA-Z0-9]+" | Select-Object -First 5
  ```
  Expected: No matches (keys should be in .env only)

- [ ] **Verify .env not in git**
  ```powershell
  git status
  ```
  Expected: `.env` listed under "Untracked files" or ignored

- [ ] **Test API key validation**
  ```powershell
  # Wrong key should fail
  curl -X POST http://localhost:5000/speak -H "Content-Type: application/json" -H "X-API-Key: wrong-key" -d "{\"text\":\"test\"}"
  ```
  Expected: 401 Unauthorized

---

### ✅ Phase 8: Documentation (5 min)

- [ ] **Read stabilization docs**
  - [ ] `README_STABILIZATION.md`
  - [ ] `DEPLOYMENT_CHECKLIST.md` (this file)

- [ ] **Update project README**
  Add note about stabilization:
  ```markdown
  ## Recent Updates
  
  **October 2025 - v2.0 Stabilization**
  - Fixed circular imports
  - Unified configuration system
  - Enhanced security (HMAC API key validation)
  - Corrected PyTorch version (2.5.1)
  - See `README_STABILIZATION.md` for details
  ```

---

## 🎯 Success Criteria

Mark each as complete:

- [ ] ✅ All imports resolve without `ModuleNotFoundError`
- [ ] ✅ `pytest` runs (even if some tests fail)
- [ ] ✅ `python scripts/doctor.py` shows no critical errors
- [ ] ✅ Flask API starts without crashes
- [ ] ✅ No credentials in version control
- [ ] ✅ `.env` file exists and is ignored by git
- [ ] ✅ At least one end-to-end component works (CLI or API)

---

## 🚨 Rollback Plan

If deployment fails:

### Option 1: Git Revert
```powershell
git reset --hard pre-stabilization-backup
git clean -fd
```

### Option 2: Restore Backup
```powershell
# If you backed up manually
xcopy /E /I /Y C:\Users\james\rex-ai-test\rex-ai-assistant-backup\* C:\Users\james\rex-ai-test\rex-ai-assistant\
```

---

## 📊 Deployment Status

| Phase | Status | Time | Notes |
|-------|--------|------|-------|
| 1. File Prep | ⏳ Pending | 5 min | |
| 2. Dependencies | ⏳ Pending | 10 min | |
| 3. Configuration | ⏳ Pending | 5 min | |
| 4. Imports | ⏳ Pending | 10 min | |
| 5. Validation | ⏳ Pending | 10 min | |
| 6. Component Test | ⏳ Pending | 15 min | |
| 7. Security | ⏳ Pending | 5 min | |
| 8. Documentation | ⏳ Pending | 5 min | |
| **Total** | **⏳ Pending** | **~65 min** | |

Legend: ⏳ Pending | 🔄 In Progress | ✅ Complete | ❌ Failed

---

## 📞 Troubleshooting

### Common Issues

**Issue:** Import errors after file copy
**Fix:** Ensure `rex/__init__.py` exists and exports key modules

**Issue:** PyTorch CUDA not detected
**Fix:** 
```powershell
pip uninstall torch torchvision torchaudio
pip cache purge
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Issue:** Config not loading
**Fix:** Check `.env` format (no quotes around values unless intentional)

**Issue:** TTS API 500 errors
**Fix:** Check logs in `logs/error.log`, ensure TTS model downloads

---

## ✅ Sign-Off

Deployment completed by: _________________
Date: _________________
Issues encountered: _________________
Rollback required: [ ] Yes [ ] No

---

**Next Steps After Deployment:**
1. Monitor logs for 24 hours
2. Test with real voice interactions
3. Benchmark performance vs previous version
4. Update team documentation
5. Plan Phase 3 enhancements (type hints, async audit)

---

**Document Version:** 1.0
**Last Updated:** 2025-10-16
**Status:** Ready for execution
