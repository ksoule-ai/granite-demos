"""Check spaces/granitelib-evals/data.js before deploying the Granitelib Adapters vs Decision Models Space.

Run from the repo root:

    python scripts/validate_granitelib_evals.py

It reads the JSON assigned to window.GRANITELIB_DATA and checks that every card is
well formed: each score row has one value per metric, values parse and fall in the
card's scale, headline and radar indexes point at real metrics, decision models and
adapters are known ones, and every card cites a commit or pull request. It prints
problems and exits 1 if there are any, so it can gate a deploy.

Standard library only.
"""

import json
import re
import sys
from pathlib import Path

DATA = Path(__file__).parent.parent / "spaces" / "granitelib-evals" / "data.js"
ADAPTER_KEYS = {"3b-lora", "3b-alora", "8b-lora", "8b-alora", "30b-lora", "30b-alora"}
MARKERS = {"triangle", "square", "diamond", "tri-down", "x", "circle"}
SHA = re.compile(r"^[0-9a-f]{40}$")


def load(path=DATA):
    text = path.read_text()
    marker = "window.GRANITELIB_DATA ="
    if marker not in text:
        sys.exit(f"{path}: no '{marker}' assignment found.")
    body = text.split(marker, 1)[1].strip()
    return json.loads(body[:-1] if body.endswith(";") else body)


def check(d):
    errors, warnings = [], []
    err, warn = errors.append, warnings.append

    for key in ("DM", "REPOS", "COMMITS", "PRS", "EVALS", "OOD_SETS", "OOD"):
        if key not in d:
            err(f"missing top-level key {key}")
    if errors:
        return errors, warnings

    dm_ids = [m.get("id") for m in d["DM"]]
    if len(set(dm_ids)) != len(dm_ids):
        err(f"DM: duplicate ids in {dm_ids}")
    for m in d["DM"]:
        if m.get("marker") not in MARKERS:
            err(f"DM {m.get('id')}: marker must be one of {sorted(MARKERS)}")
        if not m.get("name"):
            err(f"DM {m.get('id')}: missing name")
        if "scoreboard" in m and not isinstance(m["scoreboard"], bool):
            err(f"DM {m.get('id')}: 'scoreboard' must be true or false")
    markers = [m.get("marker") for m in d["DM"]]
    if len(set(markers)) != len(markers):
        warn("DM: two decision models share a marker shape, so they'll be hard to tell apart")

    for repo, commits in d["COMMITS"].items():
        if repo not in d["REPOS"]:
            err(f"COMMITS: repo {repo} is not in REPOS")
        for c in commits:
            if not SHA.match(c.get("sha", "")):
                err(f"COMMITS {repo}: '{c.get('sha')}' is not a full 40-character sha")
            for field in ("date", "time", "title", "files", "took"):
                if not c.get(field):
                    err(f"COMMITS {repo} {c.get('sha', '?')[:7]}: missing {field}")
    pr_nums = {}
    for repo, prs in d["PRS"].items():
        if repo not in d["REPOS"]:
            err(f"PRS: repo {repo} is not in REPOS")
        for pr in prs:
            pr_nums.setdefault(repo, set()).add(pr.get("num"))
            for field in ("num", "url", "opened", "title", "files", "took"):
                if not pr.get(field):
                    err(f"PRS {repo} #{pr.get('num')}: missing {field}")

    ids = [e.get("id") for e in d["EVALS"]]
    if len(set(ids)) != len(ids):
        err(f"EVALS: duplicate ids in {ids}")
    for ev in d["EVALS"]:
        name = f"EVALS {ev.get('id')}"
        for field in ("id", "title", "repo", "card", "anchor", "task", "scale", "metrics", "headline", "dm", "granite", "caveats"):
            if field not in ev:
                err(f"{name}: missing {field}")
        if any(f not in ev for f in ("metrics", "headline", "dm", "granite")):
            continue
        if ev["repo"] not in d["REPOS"]:
            err(f"{name}: repo {ev['repo']} is not in REPOS")
        if "prompted" in ev:
            err(f"{name}: has a 'prompted' block. Prompted (base) Granite is never shown; remove it")
        n = len(ev["metrics"])
        hi = 1 if ev["scale"] == 1 else 100
        if ev["scale"] not in (1, 100):
            err(f"{name}: scale must be 1 (0–1 scores) or 100 (0–100 scores)")
        for m in ev["metrics"]:
            if not m.get("name"):
                err(f"{name}: a metric is missing its name")
        for i in ev["headline"]:
            if not isinstance(i, int) or not 0 <= i < n:
                err(f"{name}: headline index {i} is out of range (0–{n - 1})")
        for g in ev.get("radarGroups", []):
            bad = [i for i in g.get("idx", []) if not 0 <= i < n]
            if bad:
                err(f"{name}: radar group '{g.get('title')}' has out-of-range indexes {bad}")
            if len(g.get("idx", [])) < 3:
                err(f"{name}: radar group '{g.get('title')}' needs at least 3 metrics")
        if not ev.get("radarGroups") and n < 3:
            warn(f"{name}: fewer than 3 metrics, so its radar will be degenerate")

        def rows(kind, table, known):
            for key, vals in table.items():
                if key not in known:
                    err(f"{name}: {kind} '{key}' is not one of {sorted(known)}")
                if len(vals) != n:
                    err(f"{name}: {kind} '{key}' has {len(vals)} values for {n} metrics")
                for v in vals:
                    try:
                        x = float(v)
                    except (TypeError, ValueError):
                        err(f"{name}: {kind} '{key}' value {v!r} is not a number (write scores as strings, e.g. \"0.81\")")
                        continue
                    if not isinstance(v, str):
                        err(f"{name}: {kind} '{key}' value {v!r} should be a string so the card's precision is kept")
                    if not 0 <= x <= hi:
                        err(f"{name}: {kind} '{key}' value {v} is outside 0–{hi}")

        rows("decision model", ev["dm"], set(dm_ids))
        rows("adapter", ev["granite"], ADAPTER_KEYS)
        if not ev["dm"]:
            err(f"{name}: no decision-model rows")
        if not ev["granite"]:
            err(f"{name}: no Granite adapter rows")
        elif "30b-alora" not in ev["granite"]:
            warn(f"{name}: no 30B aLoRA row; choosing the 30B aLoRA will fall back to the 30B LoRA")

        if ev.get("pr"):
            if ev["pr"].get("num") not in pr_nums.get(ev["repo"], set()):
                err(f"{name}: cites PR #{ev['pr'].get('num')} but that PR isn't listed under PRS['{ev['repo']}']")
        else:
            touched = [c for c in d["COMMITS"].get(ev["repo"], []) if f"{ev['card']}/README.md" in c.get("files", [])]
            if not touched:
                err(f"{name}: no commit in COMMITS['{ev['repo']}'] touches {ev['card']}/README.md, so the card has no source")

    ood = d["OOD"]
    width = len(d["OOD_SETS"]) + 1  # column 0 is the average
    for kind, table in (("granite", ood.get("granite", {})), ("dm", ood.get("dm", {}))):
        for key, vals in table.items():
            if len(vals) != width:
                err(f"OOD {kind} '{key}' has {len(vals)} values; expected {width} (average + {width - 1} datasets)")
    if "prompted" in ood:
        err("OOD: has a 'prompted' block. Prompted (base) Granite is never shown; remove it")

    return errors, warnings


def main():
    d = load()
    errors, warnings = check(d)
    for w in warnings:
        print(f"warning: {w}")
    for e in errors:
        print(f"error: {e}")
    cards = len(d.get("EVALS", []))
    bench = sum(len(e.get("headline", [])) for e in d.get("EVALS", []))
    if errors:
        sys.exit(f"{len(errors)} problem(s) in {DATA.name}.")
    print(f"{DATA.name} is valid: {cards} cards, {bench} headline benchmarks.")


if __name__ == "__main__":
    main()
