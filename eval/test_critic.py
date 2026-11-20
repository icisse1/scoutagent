"""
eval/test_critic.py: checks the Critic's deterministic number check on its own (no API key needed).
This shows the second agent's core check can be verified independently of the Scout.

Run: python eval/test_critic.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import parse_verdict, unsupported_numbers  # noqa: E402

EVIDENCE = "Marcelo | Real Madrid | La Liga | LB | Brazil | 2600 | 14.67 | 0.16\nJordi Alba | 13.15 | 0.13"

tests = [
    ("correct numbers", "Marcelo makes 14.67 progressive passes per 90, Jordi Alba 13.15.", []),
    ("rounding is fine", "Marcelo 14.7 vs Alba 13.2.", []),
    ("invented number caught", "Marcelo makes 16.20 progressive passes per 90.", ["16.20"]),
    ("one bad among good", "Marcelo 14.67, Alba 12.90.", ["12.90"]),
]
ok = 0
for name, draft, expected in tests:
    got = unsupported_numbers(draft, EVIDENCE)
    passed = got == expected
    ok += passed
    print(f"{'PASS' if passed else 'FAIL'}  {name}: got {got}, expected {expected}")

verdicts = [('{"verdict": "pass", "problems": []}', "pass"),
            ('Here: {"verdict": "revise", "problems": ["16.2 not in evidence"]}', "revise"),
            ("garbage", "revise")]
for text, expected in verdicts:
    got = parse_verdict(text)[0]
    ok += got == expected
    print(f"{'PASS' if got == expected else 'FAIL'}  parse_verdict({text[:30]!r}) -> {got}")
total = len(tests) + len(verdicts)
print(f"\n{ok}/{total} passed")
sys.exit(0 if ok == total else 1)
