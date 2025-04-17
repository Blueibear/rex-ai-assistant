import whisper

model = whisper.load_model("base")  # Options: tiny, base, small, medium, large

result = model.transcribe("test_audio.mp3")
print(result["text"])
