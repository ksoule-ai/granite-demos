# Updating the Granitelib Adapters vs Decision Models page

The page is static. Every score, citation and caveat lives in [`data.js`](data.js), and
[`index.html`](index.html) draws everything from it. Refreshing the page with new results
means four steps: find what changed on the model cards, copy it into `data.js`, validate,
and deploy.

All commands run from the repo root.

## 1. Find new results

```bash
pip install "huggingface_hub>=0.24" python-dotenv
python scripts/check_granitelib_evals.py
```

For each repo in `data.js`'s `REPOS`, the script lists commits newer than the ones
`data.js` cites. It saves each diff and prints the added table rows that mention a
decision model; prompted base-model rows are left out, because the page never shows them.
It also reports:

- open pull requests whose diffs add decision-model rows
- pull requests that `data.js` cites and that have since been merged or closed

Diffs go to a temporary directory unless you pass `--out DIR`. Pass `--strict` to exit
with status 1 when anything new turns up, for example in a cron job.

The script prints only the changed rows. Open the card itself, or the saved diff, to see
each table's column headers and any notes about how the decision models were run.

## 2. Edit `data.js`

`data.js` is `window.GRANITELIB_DATA = { ... };`, and everything after the `=` must stay
valid JSON: double quotes, no trailing commas, no comments.

**Record the source.** Every card needs one.

- **A new commit** goes first in `COMMITS["<repo>"]`, the list is newest first. Give it the
  full 40-character `sha`, `date` ("15 Oct 2026"), `time` ("17:30", UTC), the commit
  `title`, the `files` it touched, and `took`, a sentence on what it changed.
- **An unmerged pull request** goes in `PRS["<repo>"]` with `num`, `url`, `opened`,
  `title`, `files` and `took`. The card then gets a `pr` block with `num`, `url`, `by`,
  `opened`, `title` and `file`, where `file` links to the card's README at the PR revision
  (`.../blob/refs%2Fpr%2F<num>/<card>/README.md`).

**Update a card.** Find its entry in `EVALS` by `id` and change the score rows. If the card
added a metric, add it to `metrics` and add a value to every row, in the same order.

**Add a card.** Copy an existing entry in `EVALS` that has the same shape:

| Shape of the card | Entry to copy |
|---|---|
| Several separate benchmarks, one score each | `requirement-check` |
| One benchmark, several metrics | `answerability` |
| Two test sets with the same metrics | `factuality-detection`, which has `radarGroups` |

Fields:

| Field | What it holds |
|---|---|
| `id` | Short slug; also the page anchor (`#eval-<id>`) |
| `title`, `task` | Card name and a one-sentence description |
| `repo`, `card`, `anchor` | Repo name, the card's folder, and the README section anchor |
| `scale` | `1` if the card reports 0–1 scores, `100` if it reports 0–100 |
| `pct` | `true` if the card prints values with a `%` sign |
| `benchmarks` | `true` when the metrics are separate benchmarks (labels charts "Benchmark") |
| `metrics` | One `{ "name", "metric", "detail" }` per column. `name` is the row and radar label. Leave `metric` as `""`; rows don't show subheaders. `detail` is the fuller description shown only in tooltips (metric, split, sample count). `board` optionally renames the row on the Scoreboard alone, e.g. Answerability's "Weighted F1" shows as "MT-RAG" there. `lines` wraps a long radar label; `short` is the radar label inside a `radarGroups` card |
| `headline` | Indexes into `metrics` that go on the Scoreboard, usually the card's main score |
| `radarGroups` | Optional. Splits the metrics into several radars, each needing 3 or more |
| `dm` | Decision-model rows, keyed by `DM` id: `jev`, `msd`, `gpt`, `clef`, `openjev` |
| `granite` | Adapter rows, keyed `3b-lora`, `3b-alora`, `8b-lora`, `8b-alora`, `30b-lora`, `30b-alora` |
| `caveats` | Sentences shown in the card's note box: prompt caveats, missing adapters and so on |
| `sourceNote` | One line under the source table about where the numbers came from |

Rules:

- **Copy scores exactly** as the card prints them, as strings, for example `"0.81"` or
  `"92.2"`. The page keeps the card's precision.
- **Never add prompted or base-model rows.** The page doesn't show prompted Granite, and the
  validator rejects a `prompted` block.
- **Skip granite-4.0-micro and other non-4.1 rows.**
- **Missing adapters.** If a card is missing an adapter, such as having no aLoRA, leave its
  key out. The page falls back to the same size's other adapter and flags it.
- **New decision models.** Add the model to `DM` with an unused `marker`
  (`triangle`, `square`, `diamond`, `tri-down`, `x`), a `dash` pattern and a `url`.
- **Leaving a model off the scoreboard.** Set `"scoreboard": false` on its `DM` entry. It
  drops off the Scoreboard tab (markers, margins and legend chip) but stays on the
  Evaluations tab. openjev/openjev is set this way for now.
- **New repos.** Add the repo to `REPOS`, with `label` (shown beside each card's name) and
  `short` (a one-word name for it).

**When a cited PR merges.** Move its entry from `PRS` to the merge commit in `COMMITS`,
delete the card's `pr` block and remove its "open pull request" caveat. Check the
numbers against the merged card, and update the Sources-tab sentence in `index.html` that
mentions the PR.

The tab counts (benchmarks, cards, commits) come from the data automatically.

## 3. Validate and preview

```bash
python scripts/validate_granitelib_evals.py
```

The validator checks:

- every row has one value per metric, and values parse and fall within the card's scale
- headline and radar indexes are in range
- decision-model and adapter keys are known
- every card cites a commit or a PR

It exits 1 if anything is wrong.

To preview, open `spaces/granitelib-evals/index.html` in a browser; it loads `data.js` from
the same folder.

## 4. Deploy

```bash
python scripts/create_granitelib_evals_space.py           # private
python scripts/create_granitelib_evals_space.py --public  # public
```

This uploads the folder to the Space. It needs `HF_TOKEN` with write scope in `.env`.
