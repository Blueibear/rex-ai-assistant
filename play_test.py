import simpleaudio as sa

print("Loading test sound...")
wave_obj = sa.WaveObject.from_wave_file("assets/wake_acknowledgment.wav")
play_obj = wave_obj.play()
print("Playing...")

play_obj.wait_done()
print("Done.")
