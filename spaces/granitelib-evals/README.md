---
title: Granitelib vs Decision Models
emoji: 📊
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
models:
  - ibm-granite/granitelib-rag-r1.0
  - ibm-granite/granitelib-core-r1.0
  - ibm-granite/granitelib-guardian-r1.0
short_description: Granite 4.1 adapters vs decision models across granitelib
---

# Granitelib vs Decision Models

An interactive view of every granitelib evaluation that compares Granite 4.1 LoRA and aLoRA adapters with decision models (jev-1.13, microsoft-decision-1, gpt-6-luna-decisions, clef-flash and openjev/openjev).

The scores are transcribed from the model cards as of 14 October 2026, when the decision-model results were added:

| Repo | Card | Benchmarks |
|---|---|---|
| [granitelib-guardian-r1.0](https://huggingface.co/ibm-granite/granitelib-guardian-r1.0) | factuality-detection, from open [PR #45](https://huggingface.co/ibm-granite/granitelib-guardian-r1.0/discussions/45) | ELI5 and Biographies factuality: AUC, F1, precision, recall, accuracy |
| [granitelib-guardian-r1.0](https://huggingface.co/ibm-granite/granitelib-guardian-r1.0) | guardian-core | OOD Safety (plus 10 per-dataset scores), RAG Hallucination Detection, Function Calling Hallucination Detection |
| [granitelib-core-r1.0](https://huggingface.co/ibm-granite/granitelib-core-r1.0) | requirement-check | IFEval, InfoBench-GPT4, InfoBench-Human |
| [granitelib-rag-r1.0](https://huggingface.co/ibm-granite/granitelib-rag-r1.0) | answerability | MT-RAG answerability (4 metrics) |
| [granitelib-rag-r1.0](https://huggingface.co/ibm-granite/granitelib-rag-r1.0) | hallucination_detection | RAGTruth QA precision, recall, F1 |
| [granitelib-rag-r1.0](https://huggingface.co/ibm-granite/granitelib-rag-r1.0) | query_clarification | Underspecified / clear-query accuracy |

## Using it

- **Scoreboard** puts every headline benchmark on one 0–100 scale and shows the Granite adapter's margin over the best decision model.
- **Evaluations** has one view per card, led by a radar of its metrics, with the card's caveats, a chart of every metric with margins and the full source table. Guardian Core adds a 10-axis radar of its per-dataset OOD Safety scores.
- **Sources** lists each commit the numbers came from and which decision models ran on which card.

The legend at the top right of the scoreboard and of each card holds the controls: click the Granite chip to pick the adapter (default: 30B aLoRA), or click a decision model to compare against it.

## Caveats

The cards note that decision-model prompts were not tuned per model, and on the hallucination detection and query clarification cards the wording was chosen for openjev/openjev only. Hallucination detection publishes LoRA adapters only, so aLoRA selections fall back to the LoRA of the same size. Each card's own caveats are repeated on its Evaluations view.

## Updating

The page is static: `index.html` draws everything from `data.js`, which holds every score, citation and caveat. To pick up new results, run `scripts/check_granitelib_evals.py`, edit `data.js`, run `scripts/validate_granitelib_evals.py`, then deploy. [UPDATING.md](UPDATING.md) has the steps and a field reference.
