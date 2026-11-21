import wave
import numpy as np
from faster_whisper import WhisperModel
import time

WAV_PATH = "recording.wav"
EXPECTED_RATE = 16000
EXPECTED_CHANNELS = 1
EXPECTED_WIDTH = 2
MODEL_NAME = "small.en"

with wave.open(WAV_PATH, "rb") as wf:
    rate = wf.getframerate()
    channels = wf.getnchannels()
    width = wf.getsampwidth()
    n_frames = wf.getnframes()
    assert rate == EXPECTED_RATE, f"expected {EXPECTED_RATE} Hz, got {rate}"
    assert channels == EXPECTED_CHANNELS, f"expected {EXPECTED_CHANNELS} channel, got {channels}"
    assert width == EXPECTED_WIDTH, f"expected sample width of {EXPECTED_WIDTH}, got {width}"
    pcm = wf.readframes(n_frames)

samples = np.frombuffer(pcm, dtype = np.int16)
audio = (samples.astype(np.float32))/32768

print(f"duration: {n_frames / rate:.2f} s")
print(f"shape: {audio.shape}  dtype: {audio.dtype}")
print(f"min: {audio.min():.3f}  max: {audio.max():.3f}")

rundevice = "mps"
modelstart = time.perf_counter()
model = WhisperModel(MODEL_NAME, device = rundevice, compute_type="int8")
modelend = time.perf_counter()
modeltime = modelend - modelstart

transstart = time.perf_counter()
segments, info = model.transcribe(audio, language = "en", beam_size = 1, vad_filter = True, temperature = 0.0)

for seg in segments:
    print(f"{seg.text!r}  temp={seg.temperature}  logprob={seg.avg_logprob:.2f}")
    
transend = time.perf_counter()
transtime = transend - transstart

#if rtf is <1, then faster than real time, =1 means real time, >1 is slower than real time.
rtf = transtime / info.duration

print(f"model: {MODEL_NAME}, device: {rundevice}")
print(f"duration: {info.duration:.2f}  after VAD: {info.duration_after_vad:.2f}")
print(f"Model Creation time : {modeltime}, Transcription time: {transtime}, real-time factor: {rtf}")