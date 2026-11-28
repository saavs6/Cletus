import subprocess
import sounddevice as sd
import wave
import numpy as np
import time
from piper import PiperVoice
from pathlib import Path

WAV_PATH = "piper_test.wav"
EXPECTED_RATE = 22050
EXPECTED_CHANNELS = 1
EXPECTED_WIDTH = 2
SERVER_DIR = Path(__file__).resolve().parent.parent
ONNX_PATH = SERVER_DIR / "voices" / "en_US-lessac-medium.onnx"

def load_wav(wav_path):
    with wave.open(wav_path, "rb") as wf:
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
    return audio

text = "hello world"
start = time.perf_counter()
'''
subprocess.run(
    ["say", "-o", WAV_PATH, "--data-format=LEI16@16000", text], check = True
)  
'''
voice = PiperVoice.load(ONNX_PATH)
load_time = time.perf_counter() - start

start = time.perf_counter()
with wave.open("piper_test.wav", "wb") as wf:
    voice.synthesize_wav(text, wf)
syn_time = time.perf_counter() - start

reply_sound = load_wav(WAV_PATH)

sd.play(reply_sound, samplerate = EXPECTED_RATE)
sd.wait()
duration = len(reply_sound) / EXPECTED_RATE

rtf = syn_time / duration
print(rtf)

