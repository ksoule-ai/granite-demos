// Scores and sources for the granitelib vs decision models page.
// Everything after the "=" is plain JSON, so scripts/validate_granitelib_evals.py and
// scripts/check_granitelib_evals.py can read it. See UPDATING.md before editing.
window.GRANITELIB_DATA = {
  "DM": [
    {
      "id": "jev",
      "name": "jev-1.13",
      "marker": "triangle",
      "dash": "",
      "url": "https://openrouter.ai/typesafe/jev-1.13"
    },
    {
      "id": "msd",
      "name": "microsoft-decision-1",
      "marker": "square",
      "dash": "7 4",
      "url": "https://openrouter.ai/microsoft/microsoft-decision-1"
    },
    {
      "id": "gpt",
      "name": "gpt-6-luna-decisions",
      "marker": "diamond",
      "dash": "2 3",
      "url": "https://openrouter.ai/openai/gpt-6-luna-decisions"
    },
    {
      "id": "clef",
      "name": "clef-flash",
      "marker": "tri-down",
      "dash": "9 3 2 3",
      "url": "https://openrouter.ai/cloudflare/clef-flash"
    },
    {
      "id": "openjev",
      "name": "openjev/openjev",
      "marker": "x",
      "dash": "1 3",
      "url": null,
      "scoreboard": false
    }
  ],
  "REPOS": {
    "granitelib-rag-r1.0": {
      "short": "RAG",
      "label": "granitelib-rag"
    },
    "granitelib-core-r1.0": {
      "short": "Core",
      "label": "granitelib-core"
    },
    "granitelib-guardian-r1.0": {
      "short": "Guardian",
      "label": "granitelib-guardian"
    }
  },
  "COMMITS": {
    "granitelib-rag-r1.0": [
      {
        "sha": "167c451abc01c140c1059ed88d4bb39db9533ae9",
        "date": "14 Oct 2026",
        "time": "21:48",
        "title": "Use the cards' own granite-4.1-8b LoRA numbers as the decision models' reference row (#63)",
        "files": ["answerability/README.md", "hallucination_detection/README.md", "query_clarification/README.md"],
        "took": "Replaced the same-harness reference row with the cards' own Granite 4.1 LoRA and aLoRA rows. This is the version used here."
      },
      {
        "sha": "c1d37a2ce07da52099c7c046fcd0ff3c882c680e",
        "date": "14 Oct 2026",
        "time": "20:12",
        "title": "Add decision model results to the answerability, query clarification and hallucination detection cards (#62)",
        "files": ["answerability/README.md", "hallucination_detection/README.md", "query_clarification/README.md"],
        "took": "Added decision-model tables to the three RAG cards, each with a granite-4.1-8b LoRA row rerun on the same harness."
      }
    ],
    "granitelib-core-r1.0": [
      {
        "sha": "9771650e7d3a1c5d3cc54e2a1b5026aa108531ec",
        "date": "14 Oct 2026",
        "time": "17:59",
        "title": "Update requirement-check/README.md",
        "files": ["requirement-check/README.md"],
        "took": "Renamed the rows to IFEval, InfoBench-GPT4 and InfoBench-Human and linked the models. Scores unchanged."
      },
      {
        "sha": "4a7f51572c1ab9ee4ff0d6f4cca9c02c21b36a66",
        "date": "14 Oct 2026",
        "time": "17:45",
        "title": "added decision model comparison (#47)",
        "files": ["requirement-check/README.md"],
        "took": "Added the Performance Comparison to Decision Models table to requirement-check."
      }
    ],
    "granitelib-guardian-r1.0": [
      {
        "sha": "1a1a488a51d0d5307e30e5b3c1e66b4a5237d18a",
        "date": "14 Oct 2026",
        "time": "18:09",
        "title": "Update guardian-core/README.md",
        "files": ["guardian-core/README.md"],
        "took": "Added the four decision models to the per-dataset OOD Safety table."
      },
      {
        "sha": "03e7fef40c8777b2dde61f15a132a25fc62f78b2",
        "date": "14 Oct 2026",
        "time": "17:59",
        "title": "Update guardian-core/README.md",
        "files": ["guardian-core/README.md"],
        "took": "Linked the models and renamed rows to RAG Hallucination (LM-AggreFact) and Function Calling (FC Reward Bench). Scores unchanged."
      },
      {
        "sha": "23cd3b5f097a99bf0ccdeff7c8bbc7a8e92ee628",
        "date": "14 Oct 2026",
        "time": "17:44",
        "title": "added decision models comparison (#46)",
        "files": ["guardian-core/README.md"],
        "took": "Re-added the comparison table with the same numbers."
      },
      {
        "sha": "4e37aed6b66241148b82fc1d08007ab257f74394",
        "date": "14 Oct 2026",
        "time": "17:30",
        "title": "Update guardian-core/README.md",
        "files": ["guardian-core/README.md"],
        "took": "Removed the comparison table added a moment earlier."
      },
      {
        "sha": "17a13cfbaa150414a5147297116d717dd990f610",
        "date": "14 Oct 2026",
        "time": "17:30",
        "title": "added decision model comparison",
        "files": ["guardian-core/README.md"],
        "took": "First added the Performance Comparison to Decision Models table to guardian-core."
      }
    ]
  },
  "PRS": {
    "granitelib-guardian-r1.0": [
      {
        "num": 45,
        "url": "https://huggingface.co/ibm-granite/granitelib-guardian-r1.0/discussions/45",
        "opened": "14 Oct 17:20",
        "title": "Update factuality-detection/README.md with decision models evals",
        "files": ["factuality-detection/README.md"],
        "took": "Adds clef-flash, jev-1.13, microsoft-decision-1 and gpt-6-luna-decisions to the factuality-detection ELI5 and BIO tables. Still open and not merged; it is the source for the Factuality Detection card here."
      }
    ]
  },
  "EVALS": [
    {
      "id": "guardian-core",
      "title": "Guardian Core",
      "repo": "granitelib-guardian-r1.0",
      "card": "guardian-core",
      "anchor": "performance-comparison-to-decision-models",
      "task": "Judges whether a prompt or response meets a risk criterion: general harm, RAG groundedness, or function-call hallucination.",
      "scale": 1,
      "benchmarks": true,
      "ood": true,
      "metrics": [
        {
          "name": "OOD Safety",
          "lines": ["OOD Safety"],
          "metric": "F1 · avg of 10 datasets"
        },
        {
          "name": "RAG Hallucination Detection",
          "lines": ["RAG Hallucination", "Detection"],
          "metric": "LM-AggreFact · BAcc"
        },
        {
          "name": "Function Calling Hallucination Detection",
          "lines": ["Function Calling", "Hallucination Detection"],
          "metric": "FC Reward Bench · BAcc"
        }
      ],
      "headline": [0, 1, 2],
      "dm": {
        "jev": ["0.779", "0.783", "0.689"],
        "msd": ["0.787", "0.778", "0.741"],
        "gpt": ["0.769", "0.764", "0.731"],
        "clef": ["0.761", "0.672", "0.542"]
      },
      "granite": {
        "3b-lora": ["0.77", "0.721", "0.73"],
        "3b-alora": ["0.78", "0.745", "0.75"],
        "8b-lora": ["0.79", "0.728", "0.78"],
        "8b-alora": ["0.80", "0.741", "0.76"],
        "30b-lora": ["0.80", "0.750", "0.79"],
        "30b-alora": ["0.81", "0.757", "0.78"]
      },
      "caveats": [],
      "sourceNote": "Decision-model rows are from the card's Performance Comparison to Decision Models table. Granite rows are from its OOD Safety, LM-AggreFact and FC Reward Bench tables."
    },
    {
      "id": "factuality-detection",
      "title": "Factuality Detection",
      "repo": "granitelib-guardian-r1.0",
      "card": "factuality-detection",
      "anchor": "evaluation",
      "task": "Is a response factually incorrect? Labels come from the FactReasoner pipeline: a response is incorrect when its factuality score is below 0.8.",
      "scale": 1,
      "pr": {
        "num": 45,
        "url": "https://huggingface.co/ibm-granite/granitelib-guardian-r1.0/discussions/45",
        "by": "radum2275",
        "opened": "14 Oct 2026, 17:20 UTC",
        "title": "Update factuality-detection/README.md with decision models evals",
        "file": "https://huggingface.co/ibm-granite/granitelib-guardian-r1.0/blob/refs%2Fpr%2F45/factuality-detection/README.md#evaluation"
      },
      "metrics": [
        {
          "name": "ELI5 AUC",
          "short": "AUC",
          "metric": "ELI5 test · 1,753 samples"
        },
        {
          "name": "ELI5 F1",
          "short": "F1",
          "metric": "ELI5 test · 1,753 samples"
        },
        {
          "name": "ELI5 Precision",
          "short": "Precision",
          "metric": "ELI5 test · 1,753 samples"
        },
        {
          "name": "ELI5 Recall",
          "short": "Recall",
          "metric": "ELI5 test · 1,753 samples"
        },
        {
          "name": "ELI5 Accuracy",
          "short": "Accuracy",
          "metric": "ELI5 test · 1,753 samples"
        },
        {
          "name": "BIO AUC",
          "short": "AUC",
          "metric": "Biographies · out of distribution"
        },
        {
          "name": "BIO F1",
          "short": "F1",
          "metric": "Biographies · out of distribution"
        },
        {
          "name": "BIO Precision",
          "short": "Precision",
          "metric": "Biographies · out of distribution"
        },
        {
          "name": "BIO Recall",
          "short": "Recall",
          "metric": "Biographies · out of distribution"
        },
        {
          "name": "BIO Accuracy",
          "short": "Accuracy",
          "metric": "Biographies · out of distribution"
        }
      ],
      "headline": [1, 6],
      "radarGroups": [
        {
          "title": "ELI5 test set",
          "idx": [0, 1, 2, 3, 4]
        },
        {
          "title": "Biographies (out of distribution)",
          "idx": [5, 6, 7, 8, 9]
        }
      ],
      "dm": {
        "jev": ["0.78", "0.77", "0.68", "0.89", "0.70", "0.73", "0.72", "0.60", "0.91", "0.58"],
        "msd": ["0.79", "0.76", "0.64", "0.96", "0.66", "0.68", "0.75", "0.60", "0.99", "0.60"],
        "gpt": ["0.75", "0.75", "0.62", "0.94", "0.65", "0.74", "0.75", "0.62", "0.94", "0.62"],
        "clef": ["0.77", "0.52", "0.86", "0.37", "0.61", "0.81", "0.64", "0.85", "0.51", "0.65"]
      },
      "granite": {
        "3b-lora": ["0.88", "0.83", "0.82", "0.84", "0.80", "0.87", "0.86", "0.80", "0.93", "0.79"],
        "3b-alora": ["0.86", "0.81", "0.72", "0.92", "0.75", "0.84", "0.80", "0.73", "0.90", "0.73"],
        "8b-lora": ["0.89", "0.85", "0.81", "0.89", "0.80", "0.88", "0.85", "0.78", "0.94", "0.77"],
        "8b-alora": ["0.87", "0.82", "0.73", "0.93", "0.76", "0.89", "0.86", "0.77", "0.95", "0.80"],
        "30b-lora": ["0.89", "0.84", "0.79", "0.90", "0.80", "0.85", "0.85", "0.77", "0.94", "0.76"],
        "30b-alora": ["0.87", "0.82", "0.72", "0.94", "0.76", "0.80", "0.80", "0.70", "0.94", "0.72"]
      },
      "caveats": [
        "These results come from open pull request #45, which hasn't been merged into the card yet, so the numbers may still change.",
        "The PR lists the decision models under a \"SystemOne\" group. It does not describe how they were prompted."
      ],
      "sourceNote": "From the ELI5 and BIO tables in PR #45. The card also lists base models and granite-4.0-micro LoRA, which are left out here."
    },
    {
      "id": "requirement-check",
      "title": "Requirement Check",
      "repo": "granitelib-core-r1.0",
      "card": "requirement-check",
      "anchor": "performance-comparison-to-decision-models",
      "task": "Binary classification: does a response satisfy a stated requirement?",
      "scale": 1,
      "benchmarks": true,
      "metrics": [
        {
          "name": "IFEval",
          "lines": ["IFEval"],
          "metric": "Multi-Constraint · BAcc"
        },
        {
          "name": "InfoBench-GPT4",
          "lines": ["InfoBench-GPT4"],
          "metric": "GPT-4 labels · BAcc"
        },
        {
          "name": "InfoBench-Human",
          "lines": ["InfoBench-Human"],
          "metric": "Human labels · BAcc"
        }
      ],
      "headline": [0, 1, 2],
      "dm": {
        "jev": ["0.685", "0.768", "0.699"],
        "msd": ["0.678", "0.769", "0.698"],
        "gpt": ["0.633", "0.783", "0.713"],
        "clef": ["0.663", "0.730", "0.693"]
      },
      "granite": {
        "3b-lora": ["0.8153", "0.6960", "0.6728"],
        "3b-alora": ["0.8425", "0.7150", "0.6475"],
        "8b-lora": ["0.8360", "0.7373", "0.6546"],
        "8b-alora": ["0.8554", "0.7119", "0.6761"],
        "30b-lora": ["0.8406", "0.7512", "0.7038"],
        "30b-alora": ["0.8492", "0.7432", "0.7052"]
      },
      "caveats": [],
      "sourceNote": "The card also reports HelpSteer3, which has no decision-model results and is left out. IFEval is a held-out set."
    },
    {
      "id": "answerability",
      "title": "Answerability",
      "repo": "granitelib-rag-r1.0",
      "card": "answerability",
      "anchor": "evaluation-answerability-classification",
      "task": "Can the last user turn be answered from the supporting documents? Multi-turn MT-RAG benchmark.",
      "scale": 100,
      "metrics": [
        {
          "name": "Unanswerable F1",
          "metric": "MT-RAG"
        },
        {
          "name": "Answerable F1",
          "metric": "MT-RAG"
        },
        {
          "name": "Classification Accuracy",
          "metric": "MT-RAG"
        },
        {
          "name": "Weighted F1",
          "metric": "MT-RAG"
        }
      ],
      "headline": [3],
      "dm": {
        "openjev": ["92.1", "89.9", "91.2", "91.1"],
        "jev": ["88.9", "86.2", "87.7", "87.6"],
        "gpt": ["87.5", "81.8", "85.2", "84.9"],
        "msd": ["86.8", "83.1", "85.2", "85.1"]
      },
      "granite": {
        "3b-lora": ["91.2", "89.9", "90.6", "90.6"],
        "8b-lora": ["91.1", "90.4", "90.4", "90.8"],
        "30b-lora": ["90.4", "88.6", "89.4", "89.6"],
        "3b-alora": ["92.0", "90.5", "91.3", "91.3"],
        "8b-alora": ["91.2", "89.7", "90.5", "90.5"],
        "30b-alora": ["92.8", "91.4", "92.2", "92.2"]
      },
      "caveats": ["The card says the decision-model prompt wording was not tuned for any model, and that hosted models build their own prompts server-side."]
    },
    {
      "id": "hallucination-detection",
      "title": "Hallucination Detection",
      "repo": "granitelib-rag-r1.0",
      "card": "hallucination_detection",
      "anchor": "decision-models",
      "task": "Response-level hallucination detection on the QA portion of RAGTruth.",
      "scale": 100,
      "metrics": [
        {
          "name": "Precision",
          "metric": "RAGTruth QA"
        },
        {
          "name": "Recall",
          "metric": "RAGTruth QA"
        },
        {
          "name": "F1",
          "metric": "RAGTruth QA"
        }
      ],
      "headline": [2],
      "dm": {
        "openjev": ["71.5", "83.1", "73.1"],
        "jev": ["70.1", "81.4", "71.1"],
        "gpt": ["64.3", "72.8", "54.8"],
        "msd": ["69.8", "82.1", "69.8"]
      },
      "granite": {
        "3b-lora": ["80.9", "67.5", "70.8"],
        "8b-lora": ["75.9", "72.7", "74.1"],
        "30b-lora": ["83.4", "82.4", "82.9"]
      },
      "caveats": ["The card says the prompt wording was chosen on a held-out dev set for openjev/openjev only and not tuned for the other models.", "For the decision models, the card computes precision, recall and F1 at the response level, macro-averaged over faithful and unfaithful responses. The Granite rows are copied from the card's earlier table.", "Only LoRA adapters are published for this task, so aLoRA choices fall back to the LoRA of the same size."]
    },
    {
      "id": "query-clarification",
      "title": "Query Clarification",
      "repo": "granitelib-rag-r1.0",
      "card": "query_clarification",
      "anchor": "decision-models",
      "task": "Should the assistant ask a clarifying question? Accuracy on underspecified and clear queries.",
      "scale": 100,
      "pct": true,
      "metrics": [
        {
          "name": "Underspecified",
          "metric": "104 queries"
        },
        {
          "name": "CLEAR – random",
          "metric": "200 queries"
        },
        {
          "name": "CLEAR – hard",
          "metric": "79 queries"
        },
        {
          "name": "Overall Accuracy",
          "metric": "383 queries"
        }
      ],
      "headline": [3],
      "dm": {
        "openjev": ["78.8", "91.0", "73.4", "84.1"],
        "jev": ["87.5", "89.5", "65.8", "84.1"],
        "gpt": ["71.2", "91.0", "88.6", "85.1"],
        "msd": ["65.4", "97.0", "87.3", "86.4"]
      },
      "granite": {
        "3b-lora": ["90.0", "99.8", "94.5", "96.1"],
        "8b-lora": ["84.0", "99.8", "94.1", "94.4"],
        "30b-lora": ["85.9", "99.5", "92.8", "94.4"],
        "3b-alora": ["86.2", "100", "93.7", "95.0"],
        "8b-alora": ["87.5", "99.8", "95.4", "95.5"],
        "30b-alora": ["85.9", "99.3", "93.7", "94.5"]
      },
      "caveats": ["The card says the prompt wording was chosen on a held-out dev set for openjev/openjev only and not tuned for the other models."]
    }
  ],
  "OOD_SETS": ["AegisSafetyTest", "BeaverTails", "HarmBench", "OAI_hf", "SafeRLHF", "simpleSafety", "toxic_chat", "xstest_RH", "xstest_RR", "xstest_RR(h)"],
  "OOD": {
    "granite": {
      "3b-lora": ["0.77", "0.79", "0.75", "0.79", "0.81", "0.74", "0.98", "0.77", "0.84", "0.41", "0.78"],
      "3b-alora": ["0.78", "0.82", "0.81", "0.79", "0.78", "0.78", "0.99", "0.73", "0.89", "0.42", "0.81"],
      "8b-lora": ["0.79", "0.85", "0.80", "0.79", "0.81", "0.79", "0.98", "0.77", "0.91", "0.42", "0.82"],
      "8b-alora": ["0.80", "0.86", "0.81", "0.78", "0.83", "0.81", "0.99", "0.80", "0.91", "0.43", "0.82"],
      "30b-lora": ["0.80", "0.83", "0.79", "0.80", "0.84", "0.79", "0.99", "0.77", "0.90", "0.45", "0.84"],
      "30b-alora": ["0.81", "0.87", "0.82", "0.80", "0.82", "0.82", "0.99", "0.76", "0.92", "0.45", "0.85"]
    },
    "dm": {
      "jev": ["0.77", "0.75", "0.72", "0.76", "0.83", "0.81", "0.99", "0.75", "0.93", "0.42", "0.81"],
      "msd": ["0.78", "0.76", "0.77", "0.77", "0.83", "0.81", "0.98", "0.72", "0.91", "0.47", "0.86"],
      "gpt": ["0.77", "0.72", "0.78", "0.75", "0.84", "0.80", "0.98", "0.59", "0.90", "0.48", "0.85"],
      "clef": ["0.76", "0.75", "0.75", "0.72", "0.83", "0.70", "0.99", "0.76", "0.87", "0.44", "0.79"]
    }
  }
};
