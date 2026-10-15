"""Look for new decision-model results in the granitelib repos that the explorer page hasn't picked up.

Run from the repo root:

    pip install "huggingface_hub>=0.24" python-dotenv
    python scripts/check_granitelib_evals.py

For each repo in data.js's REPOS (granitelib-rag, -core, -guardian), it lists commits
newer than the ones data.js already cites, saves each diff, and prints the added table
rows that mention a decision model (prompted base-model rows are left out; the page never
shows them). It also reports open pull requests that touch
decision-model results, and whether pull requests data.js cites have been merged or
closed. Nothing is changed: you edit data.js by hand (see
spaces/granitelib-evals/UPDATING.md), then run validate_granitelib_evals.py.

Options:
  --out DIR   where to save diffs (default: a new temporary directory)
  --strict    exit 1 if anything new turned up (for CI or a cron job)

HF_TOKEN from .env or the environment is used if set; the repos are public, so it's optional.
"""

import argparse
import os
import re
import sys
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).parent))
from validate_granitelib_evals import load  # noqa: E402

ORG = "ibm-granite"
PROMPTED = re.compile(r"prompt", re.I)


def relevant_pattern(dm):
    words = {"decision"} | {m["name"].lower() for m in dm} | {m["id"].lower() for m in dm}
    words |= {"jev", "luna", "clef", "openjev"}
    return re.compile("|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)), re.I)


def summarize_diff(diff, pattern, limit=40):
    """Return (files touched, added table rows from hunks that mention a decision model)."""
    files = re.findall(r"^diff --git a/(\S+)", diff, flags=re.M)
    rows = []
    for hunk in re.split(r"^(?=@@ )", diff, flags=re.M):
        if not pattern.search(hunk):
            continue
        # prompted (base-model) rows are never used on the page, so leave them out
        rows += [line[1:].strip() for line in hunk.splitlines() if line.startswith("+|") and not PROMPTED.search(line)]
    return files, rows[:limit], max(0, len(rows) - limit)


def fetch(url, token=None):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(req) as r:
        return r.read().decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="directory for saved diffs")
    ap.add_argument("--strict", action="store_true", help="exit 1 if there is anything new")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")
    except ImportError:
        pass
    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    data = load()
    pattern = relevant_pattern(data["DM"])
    out = args.out or Path(tempfile.mkdtemp(prefix="granitelib-evals-"))
    out.mkdir(parents=True, exist_ok=True)
    found = 0

    for repo in data["REPOS"]:
        repo_id = f"{ORG}/{repo}"
        known = {c["sha"] for c in data["COMMITS"].get(repo, [])}
        cited_prs = {p["num"]: p for p in data["PRS"].get(repo, [])}
        print(f"\n== {repo_id}")

        commits = api.list_repo_commits(repo_id)
        idx = next((i for i, c in enumerate(commits) if c.commit_id in known), None)
        if idx is None:
            print(f"  None of the {len(known)} commits data.js cites are on main any more; showing the latest 5.")
            idx = 5
        new = commits[:idx]
        if not new:
            print(f"  No new commits. Latest is {commits[0].commit_id[:7]} ({commits[0].created_at:%d %b %Y %H:%M} UTC).")
        for c in new:
            diff = fetch(f"https://huggingface.co/{repo_id}/commit/{c.commit_id}.diff", token)
            path = out / f"{repo}-{c.commit_id[:7]}.diff"
            path.write_text(diff)
            files, rows, more = summarize_diff(diff, pattern)
            tag = "DECISION-MODEL ROWS" if rows else "no decision-model rows"
            found += bool(rows)
            print(f"  {c.commit_id[:7]}  {c.created_at:%d %b %Y %H:%M} UTC  {c.title}  [{tag}]")
            print(f"           sha {c.commit_id}  files: {', '.join(files) or '-'}  diff: {path}")
            for r in rows:
                print(f"           + {r}")
            if more:
                print(f"           ... {more} more rows in the saved diff")

        for d in api.get_repo_discussions(repo_id, discussion_type="pull_request"):
            if d.num in cited_prs:
                if d.status == "merged":
                    found += 1
                    print(f"  PR #{d.num} (cited by data.js) is MERGED. Cite the merge commit instead and re-check the numbers.")
                elif d.status == "closed":
                    found += 1
                    print(f"  PR #{d.num} (cited by data.js) was CLOSED without merging. Decide whether to keep its card.")
                else:
                    print(f"  PR #{d.num} (cited by data.js) is still {d.status}.")
                continue
            if d.status not in ("open", "draft"):
                continue
            details = api.get_discussion_details(repo_id, d.num)
            diff = details.diff or ""
            path = out / f"{repo}-pr{d.num}.diff"
            path.write_text(diff)
            files, rows, more = summarize_diff(diff, pattern)
            if not rows:
                continue
            found += 1
            print(f"  open PR #{d.num}  {d.created_at:%d %b %Y %H:%M} UTC  {d.title}  by {d.author}  [DECISION-MODEL ROWS]")
            print(f"           https://huggingface.co/{repo_id}/discussions/{d.num}  files: {', '.join(files)}  diff: {path}")
            for r in rows:
                print(f"           + {r}")
            if more:
                print(f"           ... {more} more rows in the saved diff")

    print(f"\nDiffs saved in {out}")
    if found:
        print(f"{found} item(s) to review. Update spaces/granitelib-evals/data.js as described in UPDATING.md.")
    else:
        print("Nothing new: data.js is up to date with the cards.")
    if args.strict and found:
        sys.exit(1)


if __name__ == "__main__":
    main()
