"""Original patient-macro ranking metrics and fold assignment."""
import hashlib
import numpy as np
from scipy.stats import rankdata
from . import core as old

def fixed_outer(patients, historical_patients, old_salt, extra_salt, folds=5):
    """Keep the historical 30 patients' outer roles; balance additional patients."""
    original, _ = old.hash_round_robin(historical_patients, folds, old_salt)
    lookup = dict(zip(historical_patients, original.tolist()))
    counts = np.bincount(original, minlength=folds)
    extra = sorted(set(patients) - set(historical_patients),
                   key=lambda p: hashlib.sha256(f'{extra_salt}|{p}'.encode()).hexdigest())
    for patient in extra:
        fold = int(np.argmin(counts))
        lookup[patient] = fold
        counts[fold] += 1
    assignments = np.asarray([lookup[p] for p in patients], dtype=int)
    return assignments


def new_entrant_recall(early, late, prediction, k=20):
    """Recovery of drugs newly entering the true late top-k, on a joint mask.

    No true entrant => undefined, reported as absent with its denominator.
    Minimum ranks include all boundary ties, without drug-ID tie breaking.
    Persistence has zero recovery by construction; compare to calibrated controls.
    """
    values = []
    counts = []
    for a, b, p in zip(early, late, prediction):
        mask = np.isfinite(a) & np.isfinite(b) & np.isfinite(p)
        if mask.sum() < 50:
            raise ValueError('Joint coverage below the locked minimum')
        a, b, p = a[mask], b[mask], p[mask]
        ra, rb, rp = rankdata(-a, method='min'), rankdata(-b, method='min'), rankdata(-p, method='min')
        previous = ra <= k
        true_new = (rb <= k) & ~previous
        predicted_new = (rp <= k) & ~previous
        n = int(true_new.sum())
        counts.append(n)
        values.append(float((true_new & predicted_new).sum() / n) if n else np.nan)
    return np.asarray(values), np.asarray(counts)


def all_scores(early, late, prediction):
    s, n = old.patient_scores(late, prediction, 50)
    e, _ = new_entrant_recall(early, late, prediction)
    if not np.isfinite(s).all() or not np.isfinite(n).all():
        raise ValueError('Non-finite primary or secondary scores')
    return {'Spearman': s, 'NDCG20': n, 'NewEntrantRecall20': e}
