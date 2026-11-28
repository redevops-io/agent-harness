"""Criterion 4 measurements against the judge server. usage: infra_check.py baseline|after OUT.json
decode: 3 x (prompt ~30, 900 completion tokens, thinking off so the count is exact); long prompts: synthetic text sized with the server's own tokenizer."""
import json, sys, time, urllib.request
BASE = "http://192.168.40.105:30807"; MODEL = "qwen3.8-27b"
def post(path, body, timeout=3600):
    r = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": "Bearer x"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
def ntok(text): return post("/tokenize", {"model": MODEL, "prompt": text})["count"]
def chat(prompt, max_tokens):
    t0 = time.time()
    r = post("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
                                       "chat_template_kwargs": {"enable_thinking": False}})
    return r["usage"], r["choices"][0]["finish_reason"], round(time.time() - t0, 2)
def sized(n):
    unit = "The pump casing was inspected and the vibration trend recorded for the sister unit. "
    k = ntok(unit); text = unit * (n // k)
    while ntok(text) > n: text = text[:-len(unit)]
    return text + "\nIn one word, what was recorded?"
res = []
for _ in range(3):
    u, f, s = chat("Write a long numbered list of distinct facts about pumps, one per line, continuing until told to stop.", 900)
    res.append({"kind": "decode", "prompt": u["prompt_tokens"], "completion": u["completion_tokens"], "seconds": s, "tok_per_s": round(u["completion_tokens"] / s, 1), "finish": f})
if sys.argv[1] == "after":
    for n in (97000, 155000):
        try:
            u, f, s = chat(sized(n), 17); res.append({"kind": f"long_{n}", "prompt": u["prompt_tokens"], "completion": u["completion_tokens"], "seconds": s, "finish": f, "ok": True})
        except Exception as e:
            res.append({"kind": f"long_{n}", "ok": False, "error": repr(e)[:300]})
json.dump(res, open(sys.argv[2], "w"), indent=1); print(json.dumps(res))
