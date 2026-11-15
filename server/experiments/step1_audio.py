import sounddevice as sd
import numpy as np
import wave

SAMPLE_RATE = 16000
SECONDS = 5
OUTPUT_PATH = "recording.wav"

recording = sd.rec(SECONDS * SAMPLE_RATE, samplerate = SAMPLE_RATE, channels = 1, dtype = 'int16')
print("Recording...")
sd.wait()
print("done")
print(f"last 10 samples:\n{recording[-10:]}")
print(f"shape:  {recording.shape}")
print(f"dtype:  {recording.dtype}")
print(f"min:    {recording.min()}")
print(f"max:    {recording.max()}")
print(f"bytes:  {recording.nbytes}")

rms = np.sqrt((recording.astype(np.float64)**2).mean())
print(f"rms:    {rms:.1f}")

with wave.open(OUTPUT_PATH, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)      # bytes per sample, not bits
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(recording.tobytes())   # the NumPy method that gives raw bytes

print(f"saved:  {OUTPUT_PATH}")

print("Playing Back...")
sd.play(recording, samplerate = SAMPLE_RATE)
sd.wait()
print("Done")