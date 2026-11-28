import sounddevice as sd
import wave
import numpy as np
import time
from piper import PiperVoice
from pathlib import Path
import re
from scipy.signal import resample_poly

WAV_PATH = "piper_test.wav"
EXPECTED_RATE = 22050
WIRE_RATE = 16000
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

def clean_for_speech (text):
    clean_layer0 = re.sub(r"^assistant\n+", "", text)
    clean_layer1 = re.sub(r"\*[^*]*\*", r"", clean_layer0) #remove stage directions eg: *jumps*
    clean_layer2 = re.sub(r"\*", r"", clean_layer1) #remove all asteriks
    clean_layer3 = re.sub(r"[^\w\s.,!?'’\"-]", "", clean_layer2) #filter for only letters, numbers, and puncuation
    clean_text = re.sub(r"\s+", r" ", clean_layer3).strip() #remove all extra whitespace
    
    return clean_text

def resample(sound, rate_in, rate_out):
    new_sound = resample_poly(sound, rate_out, rate_in)
    return new_sound

def to_int16(audio):
    new_audio = np.clip(audio * 32767, -32767, 32767).astype(np.int16)
    return new_audio


text = "I am a fish!" #edit this line to test

text = clean_for_speech(text)
start = time.perf_counter()
voice = PiperVoice.load(ONNX_PATH)
load_time = time.perf_counter() - start

start = time.perf_counter()
with wave.open("piper_test.wav", "wb") as wf:
    voice.synthesize_wav(text, wf)
syn_time = time.perf_counter() - start

reply_sound = load_wav(WAV_PATH)
start = time.perf_counter()
reply_sound = resample(reply_sound, EXPECTED_RATE, WIRE_RATE)
resample_time = time.perf_counter() - start

sd.play(reply_sound, samplerate = WIRE_RATE)
sd.wait()
duration = len(reply_sound) / WIRE_RATE

rtf = syn_time / duration
print(rtf)
print(resample_time)

wire_sound = to_int16(reply_sound)

OUTPUT_PATH = "wiretest.wav"

with wave.open(OUTPUT_PATH, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)      # bytes per sample, not bits
    wf.setframerate(WIRE_RATE)
    wf.writeframes(wire_sound.tobytes())   # the NumPy method that gives raw bytes

print(f"saved:  {OUTPUT_PATH}")
