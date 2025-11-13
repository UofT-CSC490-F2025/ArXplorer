"""
Standalone evaluator for Precision@1 over a JSON predictions file.

Input JSON format:
  { "preds": [ {"label": int, "pred": int, ...}, ... ] }
Outputs a small metrics JSON to stdout.
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description="Compute Precision@1 from a predictions JSON file")
    ap.add_argument("--preds", required=True, help="Path to JSON file with a 'preds' array of {label,pred}")
    args = ap.parse_args()

    with open(args.preds, "r", encoding="utf-8") as f:
        data = json.load(f)
    preds = data.get("preds", [])

    pred_pos = sum(1 for p in preds if int(p.get("pred", 0)) == 1)
    tp = sum(1 for p in preds if int(p.get("pred", 0)) == 1 and int(p.get("label", 0)) == 1)
    fp = pred_pos - tp
    precision = tp / max(1, pred_pos)
    out = {"precision": precision, "tp": tp, "fp": fp, "pred_pos": pred_pos}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
