"""List available Windows TTS voices and test them."""
import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')

print("Available Windows TTS Voices:\n")
for i, voice in enumerate(voices):
    print(f"{i}: {voice.name}")
    print(f"   ID: {voice.id}")
    print(f"   Languages: {voice.languages}")
    print()

print("\nTo use a voice, add this to your .env file:")
print("REX_WINDOWS_TTS_VOICE_INDEX=<number>")
print("\nFor example: REX_WINDOWS_TTS_VOICE_INDEX=1")
