try:
    import simpleaudio as sa
except ImportError:
    print("ERROR: simpleaudio is not installed.")
    print("On Windows, simpleaudio has build issues and is not required.")
    print("Install on Linux/Mac with: pip install simpleaudio")
    exit(1)

print("Loading test sound...")
wave_obj = sa.WaveObject.from_wave_file("assets/wake_acknowledgment.wav")
play_obj = wave_obj.play()
print("Playing...")

play_obj.wait_done()
print("Done.")

