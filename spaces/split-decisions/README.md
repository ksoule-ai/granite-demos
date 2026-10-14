---
title: Decisions Switching Costs
emoji: 🔀
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
models:
  - ibm-granite/granite-switch-4.1-3b-preview
short_description: Five models race yes/no decisions and one-word answers
---

# Decisions Switching Costs

Some questions want a fast yes/no call. Others want a word. Decisions
Switching Costs gives five models a list that mixes the two, one question at a time, and times
each model end to end.

It grew out of the Obstacle Course in
[Noul Race](https://github.com/ksoule-ai/granite-demos/tree/main/spaces/noul-race).
Nothing is scored here: the page compares speed.

## The models

| Model | ⚡ Yes/no question | 💬 Freeform question |
|---|---|---|
| **Granite Switch** ([`ibm-granite/granite-switch-4.1-3b-preview`](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview)) | Uncertainty adapter: its certainty in a prefilled "Yes" | Base model, same endpoint |
| **JEV-9B** ([`autotrust/JEV-9B`](https://huggingface.co/autotrust/JEV-9B)), AutoTrust's open reproduction of Jev | Decision adapter: the probability of "true" | Base model, same endpoint, thinking off |
| **Gemma Decider** ([`Mapika/decider-chat-gemma4-31b`](https://huggingface.co/Mapika/decider-chat-gemma4-31b)), the unmodified Gemma 4 31B with a decider config | decider's readout: the model's own probability for the answer letter | Same model, chat completion, thinking off |
| **GPT Luna**, on OpenRouter | [`openai/gpt-6-luna-decisions`](https://openrouter.ai/openai/gpt-6-luna-decisions) | [`openai/gpt-6-luna`](https://openrouter.ai/openai/gpt-6-luna), reasoning off, priority tier |
| **Jev + Qwen**, on OpenRouter | [`typesafe/jev-1.13`](https://openrouter.ai/typesafe/jev-1.13), TypeSafe AI's System One model | [`qwen/qwen3.8-27b`](https://openrouter.ai/qwen/qwen3.8-27b), reasoning off |

A yes/no answer is a **noul**: the probability that the answer is yes. A
freeform question is a **brainstorming** question: it asks the model to come up
with something from the article, such as a fitting word, label, name or theme,
and the model is told to answer in one word (up to 20 tokens). No model
reasons before it writes.

**Jev returns decisions only**, so it can't write an answer. It is paired with
Qwen3.8 27B, which takes the freeform questions. Qwen thinks by default, so
the request turns its reasoning off.

**GPT Luna is two model ids on OpenRouter.** Decisions go to
`openai/gpt-6-luna-decisions`, through the same System One request that Jev
takes. Text goes to `openai/gpt-6-luna` as a chat completion, with reasoning
off.

**Priority tier.** Luna's chat requests set `service_tier` to `priority`,
OpenAI's fastest processing tier, which OpenRouter passes through. It costs
about twice the default rate. Nothing else in the race has one:

- OpenAI's Decisions API rejects `service_tier` as an unknown parameter, and
  OpenRouter ignores it on decision requests. So Luna Decisions and Jev run
  on their only endpoint.
- Qwen's providers have no tiers. It already goes to the fastest one.

**Gemma Decider has no adapter.** A decision is the base model's next-token
distribution over the option letters at the "Answer: (" slot of
[decider](https://pypi.org/project/decider-ai/)'s prompt, softmaxed at the
temperature its config gives. The Space builds the prompt with decider's own
code, as token ids, and sends a one-token completion limited to the option
letters, as
[`scripts/gemma_decider_client.py`](https://github.com/ksoule-ai/granite-demos/blob/main/scripts/gemma_decider_client.py)
does. decider-ai declares torch among its dependencies but its prompt modules
don't use it, so the app installs it without dependencies at start-up; only
`transformers` is needed, for the tokenizer.

**Fastest provider.** Every OpenRouter chat request sets `provider.sort` to
`throughput`, so it goes to the fastest host OpenRouter has for that model.
That covers Qwen, Luna's text model, and gpt-oss-120b, which writes the
questions.

## The page

The look follows IBM's Carbon design, as in IBM's
[Granite Vision Document Intelligence](https://huggingface.co/spaces/ibm-granite/granite-vision-document-intelligence)
demo: IBM Plex type, a cool-grey frame around white panels, and black and
white pill buttons. Like that demo, it's light mode only.

- **About panel.** Opens over the page from the ⓘ button in the top bar: what
  the demo is, the two question types and the five models. It doesn't open on
  its own.
- **Description.** Under the top bar: agents need to switch between quick
  decisions and natural-language tasks, each switch can cost time, and Granite
  Switch lets one model flip between deciding and chatting while reusing its
  prefix cache.
- **Top bar.** **New article** sets up another round. **Run the models**
  starts the run, and is switched on once the questions are in.
- **Article (Context/State) panel.** A random English Wikipedia article, with a link to it.
  The text is under **Read the article**, closed until you open it.
- **Endpoints panel.** Whether the Granite Switch, JEV-9B and Gemma endpoints
  are awake, with a coloured dot for each.
- **Total Latency (lower is better) panel.** A bar per model that lengthens as its run goes on, with
  "Time spent" as the axis label.
  - Each question is a segment, in the order the model took it, in the colour
    of its icon: yellow for ⚡ decision time on yes/no questions, grey for 💬
    chat time on freeform ones.
  - Beside each bar are the model's total time and its **total cache reuse**:
    the share of its prompt tokens read from cache over all its requests so
    far. Only requests whose API reports a cache count are included; OpenRouter's
    decision requests (Jev's and Luna Decisions') don't, and hovering shows how
    many requests are counted. A model whose API reports none reads "cache not
    reported".
  - **The bars swap places as models finish**, so first place is always on
    top: a finished model moves up under the ones that finished before it,
    models still running follow, and a model that couldn't run is last.
  - Finishers get 🥇, 🥈 or 🥉, and the winner's bar is outlined in green.
  - All bars share one scale, which steps up (5 s, 10 s, 15 s, …) as the
    slowest model passes each mark.
- **Answers panel.** A compact table: a row per question, a column per model,
  in a fixed order. Each column fills downwards as answers arrive, and the
  question a model is working on is highlighted. A noul is shown as the number
  with Yes (above 0.5) or No after it, as in `0.97 (Yes)`. A freeform answer
  is shown as written.
- **How each model was called.** Closed until you open it. One line per model:
  the time, the kind and number of requests, and the same total cache reuse
  in tokens.

**The questions.** gpt-oss-120b writes 10 yes/no questions and 5 brainstorming
ones about the article. They appear as they're written, then are put in a
random order. Each has an icon: ⚡ for yes/no, 💬 for freeform. The
brainstorming questions have no single right answer and ask for one word, such
as "In one word, how would you describe the mood in the city after the fire?"

**Restarting.** Both buttons stay live during a run, and the run button reads
**Restart**. Either one stops the run in progress first: each model finishes
the question it's on and sends no more, and a model still waiting for its
endpoint to wake stops waiting (the endpoint keeps waking for the next run).
Then the fresh run, or the new round, starts straight away. Closing the page
stops a run the same way.

The five models run side by side. Each one takes the questions in order, and
doesn't send a question until the answer to the one before it has come back.
So every question is its own request.

**A model that can't run** has a greyed bar that reads N/A, no medal and a
hatched column, with the reason under **How each model was called**: a missing API
key, an endpoint that won't wake, or every call failing. The other models
carry on.

## How Granite produces a noul

Granite never answers the yes/no question itself. The prompt ends with
"Reply with exactly one word, 'Yes' or 'No'.", the answer is prefilled as
"Yes", and Granite Switch's embedded `uncertainty` adapter scores it. The noul
is **c(yes)**: the adapter's certainty that "Yes" is correct.

The adapter always replies `{"score": "N"}`, where the digit N is the only
part that carries information. The request prefills the reply up to the digit
and asks vLLM to continue it, so the model generates one token. Its top-10
logprobs give the certainty: keep the digit candidates, renormalize, and take
the expected value of their mapped certainties. The digits map to ten bins
(0.05, 0.15, … 0.95), so the noul always falls between 0.05 and 0.95.

The adapter is an aLoRA: it switches on at its invocation token and reads the
base model's KV cache for everything before it. The article comes first in
every prompt, so every question after the first reuses the cached article,
whether it goes to the adapter or the base model.

The [Noul Race README](https://github.com/ksoule-ai/granite-demos/tree/main/spaces/noul-race#how-granite-produces-a-noul)
has the measurements behind this method.

## Endpoints and warm start

Granite Switch runs on vLLM on **a single NVIDIA L4 GPU (24 GB)**, on a
Hugging Face Inference Endpoint. JEV-9B runs on stock vLLM on its own
endpoint, created by
[`scripts/create_jev_endpoint.py`](https://github.com/ksoule-ai/granite-demos/blob/main/scripts/create_jev_endpoint.py).
Gemma Decider runs on stock vLLM on **a single NVIDIA A100 GPU (80 GB)**,
created by
[`scripts/create_gemma_decider_endpoint.py`](https://github.com/ksoule-ai/granite-demos/blob/main/scripts/create_gemma_decider_endpoint.py).

All three endpoints scale to zero after 15 idle minutes, and a cold start takes
about 3 to 5 minutes. Opening the page starts waking them and sends each a
throwaway request once it's up. The Endpoints panel shows *checking /
asleep, waking / warming / ready*, refreshed every 3 seconds. A
model's clock starts once its endpoint is awake, and the wait is reported
separately.

## Setup

1. Create a Gradio Space (free CPU hardware is enough) and push this folder to
   it, or run `python scripts/create_split_decisions_space.py` from the repo
   root.
2. **Settings → Variables and secrets**, add these secrets:
   - `HF_ENDPOINT_URL`: the Granite Switch endpoint URL, ending in `/v1`.
   - `HF_TOKEN`: a token allowed to call that endpoint.
   - `OPENROUTER_API_KEY`: used for GPT Luna, for Jev and Qwen, and for
     writing the questions. Without it the page has no questions to run.
   - `JEV9B_ENDPOINT_URL`: the JEV-9B endpoint URL. It's called with
     `HF_TOKEN`. Without it, JEV-9B shows as unavailable.
   - `GEMMA_DECIDER_ENDPOINT_URL`: the Gemma decider endpoint URL. It's
     called with `HF_TOKEN`. Without it, Gemma Decider shows as unavailable.
   - Optional: `MODEL_ID`, `GRANITE_HARDWARE`, `JEV9B_HARDWARE`,
     `GEMMA_DECIDER_HARDWARE`, `JEV_MODEL`,
     `JEV_PARTNER_MODEL`, `LUNA_MODEL`, `LUNA_DECISIONS_MODEL`,
     `QUESTION_MODEL`, `MAX_CONTEXT_CHARS`. A model id with no vendor prefix
     gets its default's: `jev-1.13` becomes `typesafe/jev-1.13`.

No OpenAI key is needed.

## Caveats

- The models don't run on the same hardware or over the same network path, so
  the times compare services, not models alone.
- Wikipedia articles are public, so the hosted models may answer some
  questions from what they already know.
- The Granite noul comes from an adapter built to judge whether an answer is
  correct, not a model trained to output calibrated decisions.
- The Gemma decider endpoint is a 31B model on an A100. Opening the page wakes
  it, and it stays up for 15 minutes after the last use.
- Only the first 20,000 characters of a long article are used.
