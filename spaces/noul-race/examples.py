# SPDX-License-Identifier: Apache-2.0
"""Example Wikipedia articles for Noul Race.

The app pulls the article text from the URL and uses it as the context. The
questions aren't written here: the app asks gpt-oss-120b to write them, with
answers, from whichever article is loaded. A long article also shows off the
endpoint batching, since every question shares it as a cached prefix.
"""

EXAMPLES = [
    ("Apollo 11", "https://en.wikipedia.org/wiki/Apollo_11"),
    ("Great Fire of London", "https://en.wikipedia.org/wiki/Great_Fire_of_London"),
    ("Honey bee", "https://en.wikipedia.org/wiki/Honey_bee"),
]
EXAMPLE_LABELS = [label for label, _ in EXAMPLES]
EXAMPLE_INPUTS = [[url] for _, url in EXAMPLES]
