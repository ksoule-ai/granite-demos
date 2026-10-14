"""Chat and typed decisions from one Gemma 4 endpoint (scripts/create_gemma_decider_endpoint.py).

    pip install httpx python-dotenv "transformers>=5" huggingface_hub jinja2
    pip install --no-deps "decider-ai>=1.8.0"
    python scripts/gemma_decider_client.py

Reads HF_TOKEN and GEMMA_DECIDER_ENDPOINT_URL (the endpoint's URL, with or
without /v1) from .env or the environment.

Mapika/decider-chat-gemma4-31b has no adapter: a decision is the stock model's
next-token distribution over the option letters at the "Answer: (" slot of
decider's chat-layout prompt, softmaxed at T(n) = max(0.05, a + b ln n) for a
question with n options (decider_config.json). This module does what
decider.serve_vllm does, over HTTP instead of in-process:

  * The prompt rows are built by decider-ai's own code (prompt_fast.build_rows
    with the model's chat template, thinking off) and sent as token ids, so
    they match decider token for token. Only decider's pure-Python modules are
    imported, which is why --no-deps is enough.
  * Each row is a one-token completion limited to the question's letters
    (allowed_token_ids), at temperature 0. The endpoint runs with
    --logprobs-mode processed_logprobs, so the letters' log-probabilities come
    back renormalised over the letters: a log-softmax, which differs from the
    logits by one constant, so softmax(logprob / T) = softmax(logit / T).
    Temperature 0 matters: the model's generation_config sets top_k=64 and
    top_p=0.95, which would cut unlikely letters from the processed
    logprobs; greedy requests skip both.
  * Answers are assembled by decider's systemone code, so they have the shape
    of POST /v1/systemone: {"q1": {"type": "noul", "noul": 0.93}, ...}.
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor

import httpx
from decider import systemone as S1
from decider import temperature as TT
from decider.prompt import chat_for, label_table, load_decider_config
from decider.prompt_fast import build_rows
from dotenv import load_dotenv
from transformers import AutoTokenizer

REPO = "Mapika/decider-chat-gemma4-31b"
# The endpoint's --max-model-len is 8192; this leaves room for the template,
# the questions and the options after the state.
MAX_STATE_TOKENS = 6144

load_dotenv()
ENDPOINT_URL = os.environ.get("GEMMA_DECIDER_ENDPOINT_URL", "").rstrip("/")
if ENDPOINT_URL and not ENDPOINT_URL.endswith("/v1"):
    ENDPOINT_URL += "/v1"

_config = load_decider_config(REPO)
tok = AutoTokenizer.from_pretrained(REPO)
CHAT = chat_for(tok, _config)  # the chat layout's template head and tail, thinking off
LETTER_IDS = label_table(tok)[1]
(TEMPERATURE, TEMPERATURE_MAP), _ = TT.from_config(_config)
ISOLATED = bool(_config.get("isolated_levels", False))

http = httpx.Client(headers={"Authorization": f"Bearer {os.environ.get('HF_TOKEN', '')}"}, timeout=60.0)


def _post(path: str, payload: dict) -> dict:
    response = http.post(f"{ENDPOINT_URL}{path}", json=payload)
    response.raise_for_status()
    return response.json()


def chat(messages: list[dict], **kwargs) -> str:
    """An ordinary chat completion from the same endpoint (thinking off unless kwargs say otherwise)."""
    payload = {"model": REPO, "messages": messages, "chat_template_kwargs": {"enable_thinking": False}, **kwargs}
    return _post("/chat/completions", payload)["choices"][0]["message"]["content"]


def _prepare(state, questions: dict):
    """decider.serve.prepare for independent questions: -> (rendered questions, row index, rows)."""
    rqs = {k: S1.render_question(v) for k, v in questions.items()}
    flat, index = S1.plan_rows(rqs, ISOLATED)
    items, _ = build_rows(
        tok,
        S1.render_state(state),
        [[(row["question"], list(row["options"]))] for row in flat],
        max_ctx_tokens=MAX_STATE_TOKENS,
        chat=CHAT,
    )
    for item, kind in zip(items, S1.row_types(rqs, index)):
        item["types"] = [kind]
    return rqs, index, items


def _letter_logprobs(ids: list[int], n: int) -> list[float]:
    """The log-softmax over the first n letters at the end of `ids` (the answer slot)."""
    letters = LETTER_IDS[:n]
    body = _post(
        "/completions",
        {
            "model": REPO,
            "prompt": ids,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": n,
            "allowed_token_ids": letters,
            "return_tokens_as_token_ids": True,
        },
    )
    top = body["choices"][0]["logprobs"]["top_logprobs"][0]
    by_id = {int(token.split(":")[1]): value for token, value in top.items()}
    return [by_id.get(t, -1e9) for t in letters]


def _softmax(logprobs: list[float], temperature: float) -> list[float]:
    top = max(logprobs)
    weights = [math.exp((x - top) / temperature) for x in logprobs]
    total = sum(weights)
    return [w / total for w in weights]


def systemone(state, questions: dict) -> dict:
    """Typed decisions in the shape of POST /v1/systemone's "answers".

    state: a string or JSON value. questions: {id: {"type": "noul" | "choice" | "score",
    "instructions": ..., "criteria": ...}}, as the System One API takes them.
    """
    rqs, index, items = _prepare(state, questions)
    temperatures = TT.for_items(TEMPERATURE, TEMPERATURE_MAP, items)

    def score(i: int) -> list[float]:
        item = items[i]
        t = temperatures[i][0] if isinstance(temperatures, list) else temperatures
        return _softmax(_letter_logprobs(item["ids"], item["nopts"][0]), t)

    if not items:
        return {}
    # Every row starts with the same state. Run the first one alone so the rest
    # find that prefix in vLLM's cache, then run the rest together.
    probs = [score(0)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        probs += list(pool.map(score, range(1, len(items))))
    return S1.assemble(rqs, index, probs)


if __name__ == "__main__":
    if not ENDPOINT_URL:
        raise SystemExit("GEMMA_DECIDER_ENDPOINT_URL not set — put it in .env or export it.")
    ticket = "Customer: I was charged twice for my March invoice and need one of the charges refunded today."
    print("chat:", chat([{"role": "user", "content": f"Summarise in one sentence:\n{ticket}"}], max_tokens=80))
    answers = systemone(
        ticket,
        {
            "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
            "team": {"type": "choice", "instructions": "Which team should handle this?",
                     "criteria": {"billing": None, "technical": None, "sales": None}},
            "urgency": {"type": "score", "instructions": "How urgent is this?",
                        "criteria": ["not urgent", "somewhat urgent", "urgent"]},
        },
    )
    for key, answer in answers.items():
        print(f"{key}:", answer)
