#!/usr/bin/env python3
"""
Turn t1_audit_answers.json (exported from t1_audit.html) into the Appendix-B table.

    PYTHONPATH= .../python summarize_audit.py ~/Downloads/t1_audit_answers.json
Prints per-primitive agreement (correct / wrong / unsure) and overall rate,
and writes results/audit/t1_audit_summary.json.
"""
import sys
import json
import pandas as pd

BASE = "."


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/results/audit/t1_audit_answers.json"
    df = pd.DataFrame(json.load(open(path)))
    answered = df[df.answer.notna()]
    print(f"answered: {len(answered)}/{len(df)}")
    rows = {}
    print(f"\n{'primitive':22} {'n':>4} {'correct':>8} {'wrong':>6} {'unsure':>7} {'agree%':>7}")
    for prim, g in answered.groupby("prim"):
        n = len(g)
        c = int((g.answer == "yes").sum()); w = int((g.answer == "no").sum())
        u = int((g.answer == "unsure").sum())
        agree = c / max(c + w, 1)
        rows[prim] = {"n": n, "correct": c, "wrong": w, "unsure": u, "agreement": agree}
        print(f"{prim:22} {n:>4} {c:>8} {w:>6} {u:>7} {agree*100:>6.1f}%")
    tot_c = sum(r["correct"] for r in rows.values())
    tot_w = sum(r["wrong"] for r in rows.values())
    print(f"\nOVERALL agreement (excl. unsure): {tot_c/(tot_c+tot_w)*100:.1f}%")
    json.dump(rows, open(f"{BASE}/results/audit/t1_audit_summary.json", "w"), indent=1)
    print("saved results/audit/t1_audit_summary.json")


if __name__ == "__main__":
    main()
