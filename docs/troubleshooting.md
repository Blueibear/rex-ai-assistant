# Rex AI Assistant — Troubleshooting

## Missing API Keys

**Error:** `REX_SPEAK_API_KEY: not set`

**Solution:** Set the API key in your `.env` file:
```env
REX_SPEAK_API_KEY=your-secret-key-here
```

## FFmpeg Not Found

**Error:** `ffmpeg executable not found`

**Solution:**
- **macOS:** `brew install ffmpeg`
- **Ubuntu/Debian:** `sudo apt install ffmpeg`
- **Windows:** Download from https://ffmpeg.org/download.html and add to PATH

## PyTorch Installation Issues

**Error:** `torch is not installed`

**Solution:** Use the appropriate requirements file:
```bash
# CPU-only
pip uninstall -y torch torchvision torchaudio
pip install -r requirements-cpu.txt

# GPU with CUDA 12.4 (Windows 11)
pip uninstall -y torch torchvision torchaudio
pip install -r requirements-gpu-cu124.txt

# GPU with CUDA 11.8
pip uninstall -y torch torchvision torchaudio
pip install -r requirements-gpu.txt
```

## Microphone Permissions (macOS)

**Error:** `Audio device not accessible`

**Solution:**
1. Open **System Settings** → **Privacy & Security** → **Microphone**
2. Enable microphone access for **Terminal** or your Python interpreter

## WASAPI Issues (Windows)

**Error:** `sounddevice` or `portaudio` errors on Windows

**Solution:**
1. Install Visual C++ Redistributables: https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Install `pyaudio` from wheels: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

## Wake Word Not Detected

**Issue:** Rex doesn't respond to wake word

**Solution:**
1. Check microphone is working: `python audio_config.py --list`
2. Lower threshold: `REX_WAKEWORD_THRESHOLD=0.3` in `.env`
3. Test wake word detection: `python wakeword_listener.py`
4. Record custom wake word: `python record_wakeword.py`

## Rate Limit Errors (TTS API)

**Error:** `429 Too Many Requests`

**Solution:** Increase rate limits in `.env`:
```env
REX_SPEAK_RATE_LIMIT=60
REX_SPEAK_RATE_WINDOW=60
```

For production deployments with multiple workers, use Redis:
```env
REX_SPEAK_STORAGE_URI=redis://localhost:6379/0
```

## CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
1. Use smaller Whisper model: `REX_WHISPER_MODEL=tiny` or `base`
2. Reduce max tokens: `REX_LLM_MAX_TOKENS=50`
3. Switch to CPU: `REX_DEVICE=cpu` and `REX_WHISPER_DEVICE=cpu`
