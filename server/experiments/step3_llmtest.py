import ollama
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent   # experiments/ -> server/
PERSONA_PATH = SERVER_DIR / "cletus" / "persona1_v2.txt"

MODEL = "llama3.1:8B"

user_text = "What actually are you?"
REMINDER = "(Reply as Cletus: plain spoken words, no asterisks, under forty words.)"

messages = [
    {"role": "system", "content": PERSONA_PATH.read_text()},
    {"role": "user", "content": f"{user_text}\n\n{REMINDER}"},
]

response = ollama.chat(model= MODEL, messages=messages)
load_s = response.load_duration / 1e9
prefill_s = response.prompt_eval_duration / 1e9
decode_s = response.eval_duration / 1e9

prefill_tps = response.prompt_eval_count / prefill_s
decode_tps = response.eval_count / decode_s 

print(response)
print()
print(f"load: took {load_s:.2f} s")
print(f"prefill:  {response.prompt_eval_count} tokens in {prefill_s:.2f} s = {prefill_tps:.1f} tok/s")
print(f"decode:  {response.eval_count} tokens in {decode_s:.2f} s = {decode_tps:.1f} tok/s")
print(response.message.content)      # the reply text