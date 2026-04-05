import time

import sounddevice as sd

CANDIDATES = [12, 22, 1, 6, 0, 5]  # most likely good ones first


def try_device(idx):
    info = sd.query_devices(idx)
    sr = int(info["default_samplerate"])
    # Prefer 16000 if device supports it (common for wakeword)
    prefer = 16000
    srs = [prefer, sr] if prefer != sr else [sr]

    for rate in srs:
        try:
            with sd.InputStream(device=idx, channels=1, samplerate=rate, dtype="int16"):
                time.sleep(0.25)
            return True, rate, info["name"]
        except Exception as e:
            last = str(e)
    return False, None, info["name"] + " | last_error=" + last


print("Default devices:", sd.default.device)
print("Testing candidates:", CANDIDATES)

for idx in CANDIDATES:
    ok, rate, name = try_device(idx)
    if ok:
        print(f"OK  device={idx} rate={rate} name={name}")
    else:
        print(f"FAIL device={idx} name={name}")
