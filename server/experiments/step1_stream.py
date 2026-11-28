import sounddevice as sd
import wave
import numpy as np

SAMPLE_RATE = 16000
SECONDS = 5
BLOCK = 320
OUTPUT_PATH = "stream.wav"

chunks = []

def callback(indata, frames, time_info, status):
    if status:
        print(status)
    chunks.append(bytes(indata))

with sd.RawInputStream(samplerate = SAMPLE_RATE, channels = 1, dtype = "int16", blocksize = BLOCK, callback = callback):
    print("Recording...")
    sd.sleep(SECONDS * 1000)

pcm = b"".join(chunks)
print(f"chunks: {len(chunks)}  bytes per chunk: {len(chunks[0])}  total: {len(pcm)}")


with wave.open(OUTPUT_PATH, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)      # bytes per sample, not bits
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(pcm)

print(f"saved:  {OUTPUT_PATH}")