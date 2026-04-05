import sounddevice as sd

print("default:", sd.default.device)
print("hostapis:")
for i, h in enumerate(sd.query_hostapis()):
    print(
        f"  {i}: {h['name']} | default_in={h.get('default_input_device')} default_out={h.get('default_output_device')}"
    )

print("devices:")
for i, d in enumerate(sd.query_devices()):
    print(
        f"{i}: {d['name']} | in={d['max_input_channels']} out={d['max_output_channels']} | sr={d['default_samplerate']}"
    )
