# Scripts

This directory contains one-off developer utilities and manual demos that do
not belong at the repository root.

- `check_gpu_status.py`: prints PyTorch CUDA availability and basic GPU diagnostics
- `check_imports.py`: validates syntax and presence of core module files
- `check_patch_status.py`: inspects the installed TTS package for the transformers compatibility patch marker
- `check_tts_imports.py`: prints the TTS XTTS stream generator import section for debugging
- `find_gpt2_model.py`: locates `GPT2InferenceModel` inside an installed TTS package
- `generate_wake_sound.py`: regenerates the wake acknowledgment WAV asset on demand
- `list_audio.py`: lists available PortAudio host APIs and audio devices
- `list_voices.py`: lists available Windows TTS voices
- `manual_search_demo.py`: runs an interactive demo of the web search plugin
- `manual_whisper_demo.py`: manually transcribes an audio file with Whisper
- `play_test.py`: plays the wake acknowledgment WAV with `simpleaudio`
- `record_wakeword.py`: records and saves a custom wake-word model with `openwakeword`
- `test_imports.py`: runs a broad manual import smoke check for Rex modules
- `test_mic_open.py`: probes candidate microphone devices with `sounddevice`
- `test_transformers_patch.py`: manually verifies the transformers BeamSearchScorer compatibility shim
