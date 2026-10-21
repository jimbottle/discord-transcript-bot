#!/usr/bin/env python3
"""Run the GM check (src/llm/guard.py) on a fixed set of questions with
known answers, on the configured provider (ASK_PROVIDER).

Re-run it after editing GUARD_SYSTEM, or after changing the model. Twelve
questions: six the table rules block, six they allow. They are the set
the check was first validated on (946d85e). On the Anthropic API this
spends a few cents; on a local model it is free.

USAGE
    python scripts/check_guard.py            # exit 1 if any verdict is wrong
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm import claude, guard, provider  # noqa: E402

CONTEXT = "SESSION: Cyberpunk RED, campaign Return to Night City"

CASES = [
    ("What is my best approach to beating this scenario?", False),
    ("How do we win the debate against Cain?", False),
    ("What's the best cyberware build for Johan?", False),
    ("Is Gloria Lee secretly a traitor?", False),
    ("What is Arasaka going to do next?", False),
    ("What's the optimal way to spend our 50,000 eddies?", False),
    ("What did Noah say about the debate plan?", True),
    ("Who is Lilith Kane?", True),
    ("How does autofire work in Cyberpunk RED?", True),
    ("What did the party agree to do at the end of the session?", True),
    ("Summarise what happened tonight.", True),
    ("What did Johan say about the eddies?", True),
]


def main():
    backend = provider.name()
    print(f"GM check on {provider.describe()}, model {guard.model(backend)}\n")
    wrong = 0
    for question, expect_allowed in CASES:
        t = time.time()
        try:
            v = guard.check(
                question, api_key=claude.api_key(), context=CONTEXT, backend=backend
            )
        except guard.GuardUnavailable as e:
            print(f"ERROR  {question}\n       {e}")
            wrong += 1
            continue
        ok = v.allowed == expect_allowed
        wrong += not ok
        verdict = "allow" if v.allowed else f"block ({v.category})"
        print(
            f"{'ok   ' if ok else 'WRONG'}  {verdict:<22} {time.time() - t:4.1f}s  "
            f"{question}"
        )
    print(f"\n{len(CASES) - wrong}/{len(CASES)} correct")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
