from pathlib import Path
import ollama
import time

history = []
MAX_TURNS = 6
MODEL = "llama3.1:8B"
SERVER_DIR = Path(__file__).resolve().parent.parent   # experiments/ -> server/
PERSONA_PATH = SERVER_DIR / "cletus" / "persona1_v2.txt"

REMINDER = "(Reply as Cletus: plain spoken words, no asterisks, under forty words.)"


while True:
    user_text = input("you: ")
    if user_text.strip().lower() in ("quit", "exit"):
        break
    
    messages = [{"role": "system", "content": PERSONA_PATH.read_text()}] + history + [{"role": "user", "content": f"{user_text}\n\n{REMINDER}"}]      # persona msg + history + new user msg with reminder
    
    start = time.perf_counter()
    ttft = None
    pieces = []

    stream = ollama.chat(model=MODEL, messages=messages, stream=True,
                        options={"num_ctx": 8192}, keep_alive="1h")

    print("cletus: ", end="", flush=True)
    for chunk in stream:
        piece = chunk.message.content
        if piece and ttft is None:
            ttft = time.perf_counter() - start   # time since start
        print(piece, end="", flush=True)
        pieces.append(piece)
        if chunk.done:
            final = chunk                     # has the timing stats

    total = time.perf_counter() - start
    reply = "".join(pieces)                               # join pieces into one string
    print(f"\n  [ttft {ttft*1000:.0f} ms | total {total:.2f} s | {final.eval_count} tokens]")

    history.append({"role": "user", "content": user_text})               # bare user message
    history.append({"role": "assistant", "content": reply})               # assistant reply
    history = history[-MAX_TURNS * 2:]                     # keep only the last MAX_TURNS exchanges