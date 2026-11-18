#!/usr/bin/env python3
"""Predict a supplied batch on the checkpoint's fixed drug panel."""
import argparse
import os
from pathlib import Path
import sys
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from para_drugrank import load_data, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True, help="New output NPZ containing ranking scores and the ordered drug axis")
    args = parser.parse_args()
    data, model = load_data(args.data, training=False), load_model(args.checkpoint)
    score = model.predict(data["expression"], data["source_response"], data["gene_names"], data["drug_names"])
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing prediction file")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ranking_scores=score, drug_names=data["drug_names"])
    print(f"Saved {score.shape[0]} rows x {score.shape[1]} fixed-panel ranking scores")


if __name__ == "__main__":
    main()
