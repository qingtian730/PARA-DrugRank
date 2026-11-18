"""Validate the prepared training data and ordered prediction-input axes."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np


def axis_digest(names):
    # Length-prefixing prevents ambiguous concatenation and preserves order.
    digest = hashlib.sha256()
    for item in np.asarray(names).astype(str):
        value = item.encode("utf-8")
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def load_data(path, training=True):
    with np.load(Path(path), allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    required = {"expression", "source_response", "gene_names", "drug_names"}
    if training:
        required |= {"target_response", "outer_fold", "inner_fold", "cohort"}
    if not required.issubset(data):
        raise ValueError(f"Missing fields: {sorted(required - set(data))}")
    expression, source = data["expression"], data["source_response"]
    if expression.ndim != 2 or source.ndim != 2 or len(expression) != len(source):
        raise ValueError("Expected expression[n,g] and source_response[n,d]")
    if not np.isfinite(expression).all() or np.isinf(source).any():
        raise ValueError("Expression must be finite; source permits NaN, never infinity")
    for key, width in (("gene_names", expression.shape[1]), ("drug_names", source.shape[1])):
        axis = data[key]
        if axis.ndim != 1 or len(axis) != width or len(set(axis.astype(str))) != width:
            raise ValueError(f"{key} must be a unique ordered axis of length {width}")
    if training:
        n = len(expression)
        if data["target_response"].shape != source.shape or np.isinf(data["target_response"]).any():
            raise ValueError("Target must match source shape and may only use NaN for unobserved")
        outer, inner = data["outer_fold"], data["inner_fold"]
        if outer.shape != (n,) or inner.shape != (n, 5):
            raise ValueError("Expected outer_fold[n] and inner_fold[n,5]")
        if set(outer.tolist()) != set(range(5)):
            raise ValueError("Expected all five frozen outer folds")
        for fold in range(5):
            if set(inner[outer != fold, fold].tolist()) != set(range(4)):
                raise ValueError("Expected all four inner folds within each outer training role")
            if not np.all(inner[outer == fold, fold] == -1):
                raise ValueError("Outer held-out rows must have inner_fold=-1")
    return data
