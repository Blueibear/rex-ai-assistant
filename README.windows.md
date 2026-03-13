# 🧠 Rex AI Assistant (Windows Quickstart)

This guide is a **Windows-specific** quickstart for setting up Rex AI Assistant
with NVIDIA RTX GPU (CUDA). For full project details, see [README.md](README.md).

---

## 🚀 Quick Setup on Windows

### 1. Prerequisites

- Python **3.11+**
- Git
- [FFmpeg](https://ffmpeg.org/download.html)
- NVIDIA drivers installed (with CUDA support)
- Microphone + speakers

### 2. Clone and create a virtual environment

```powershell
git clone https://github.com/Blueibear/askrex-assistant.git
cd rex-ai-assistant

python -m venv .venv
.venv\Scripts\activate
```

### 3. Install CUDA-enabled PyTorch (CUDA 12.4)

Run this **before** installing the rest of the requirements (RTX 3060 / CUDA 12.4):

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

You should see a version like `2.x.x+cu124`, `cuda.is_available() = True`, and your RTX GPU name.

### 4. Install project dependencies

```powershell
pip install --upgrade pip
pip install -r requirements-gpu-cu124.txt
```

### 5. Run the assistant

```powershell
python rex_assistant.py
```

Say your wake word (default: **rex**) and talk to the assistant.

### 6. Run the voice loop

```powershell
python rex_loop.py
```

This runs the full **wake word → STT → LLM → TTS** pipeline.

### 7. Run tests

```powershell
pytest
```

---

## ⚙️ Config (Windows)

Set environment variables in PowerShell before running:

```powershell
$env:REX_WAKEWORD="rex"
$env:REX_ACTIVE_USER="james"
$env:WHISPER_MODEL="base"
$env:OPENAI_API_KEY="your_api_key_here"
```

---

## 🔉 Notes

- Memory profiles live under `Memory/<user>/`
- Voices can be customised by adding a WAV file in your profile JSON
- Web search requires `SERPAPI_KEY` or defaults to DuckDuckGo scraping
- The local Flask API (`rex_speak_api.py`) can provide TTS over HTTP

---

## ✅ Summary

- Use **CUDA-enabled PyTorch (cu124 wheels)** for RTX 3060 GPUs  
- Install requirements after PyTorch (requirements-gpu-cu124.txt)  
- Start with `rex_assistant.py` or `rex_loop.py`  
- Configure via environment variables
