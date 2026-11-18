"""Frozen patient candidate families and inner-fold model selection."""
import numpy as np
from itertools import product
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from . import core as old
from . import metrics as prev
OLD_EXPERTS = ('PopulationAnchor', 'Persistence', 'PersistenceRidge', 'SourceState', 'RNA', 'Combined')
MAIN = 'PrimarySelected'
DYNAMIC = 'DynamicSelected'

def rank_rows(values):
    out = np.full_like(values, np.nan, dtype=float)
    for i, row in enumerate(values):
        finite = np.isfinite(row)
        n = int(finite.sum())
        if n < 2:
            raise ValueError('At least two measured drugs required for row ranking')
        out[i, finite] = (rankdata(row[finite], method='average') - 1) / (n - 1)
    return out


def impute(train, validation, fallback=.5):
    count = np.isfinite(train).sum(axis=0)
    total = np.where(np.isfinite(train), train, 0).sum(axis=0)
    mean = np.divide(total, count, out=np.full(train.shape[1], fallback), where=count > 0)
    return np.where(np.isfinite(train), train, mean), np.where(np.isfinite(validation), validation, mean)


def weights():
    # Explicit simple vertices first; no mandatory weight on the old candidate.
    preferred = [(0., 1., 0.), (0., 0., 1.), (1., 0., 0.)]
    grid = [(a / 4, b / 4, (4 - a - b) / 4) for a in range(5) for b in range(5 - a)]
    return preferred + [w for w in grid if w not in preferred]


def choose(scores, dynamic=False, tolerance=.002, dynamic_tolerance=.005):
    """Selection sees inner scores only, with a predeclared tradeoff rule."""
    best = max(v['Spearman'] for v in scores.values())
    if not dynamic:
        return next(k for k, v in scores.items() if v['Spearman'] >= best - tolerance)
    eligible = [(k, v) for k, v in scores.items()
                if v['Spearman'] >= best - dynamic_tolerance and np.isfinite(v['NewEntrantRecall20'])]
    if not eligible:
        return choose(scores, False, tolerance)
    return max(eligible, key=lambda kv: (kv[1]['NewEntrantRecall20'], kv[1]['Spearman']))[0]


def score_pool(pool, early, late):
    return {k: {metric: float(np.nanmean(v)) for metric, v in prev.all_scores(early, late, p).items()}
            for k, p in pool.items()}


def low_rank_predictions(xtr, xva, etr, eva, ltr, c):
    """Learn train-only rank changes, sharing residual structure across drugs."""
    early_tr, early_va = rank_rows(etr), rank_rows(eva)
    early_tr, early_va = impute(early_tr, early_va)
    late_tr = rank_rows(ltr)
    delta = late_tr - early_tr
    observed = np.isfinite(delta)
    count = observed.sum(axis=0)
    total = np.where(observed, delta, 0).sum(axis=0)
    mean = np.divide(total, count, out=np.zeros(delta.shape[1]), where=count >= 8)
    residual = np.where(observed & (count >= 8), delta - mean, 0.)
    u, s, vt = np.linalg.svd(residual, full_matrices=False)
    scaler = StandardScaler().fit(early_tr)
    source_pca = PCA(n_components=c['feature_PCs'], svd_solver='full')
    source_tr = source_pca.fit_transform(scaler.transform(early_tr))
    source_va = source_pca.transform(scaler.transform(early_va))
    variance = np.var(xtr.astype(float), axis=0)
    genes = np.argsort(-variance, kind='mergesort')[:min(2000, xtr.shape[1])]
    xscale = StandardScaler().fit(xtr[:, genes])
    xpca = PCA(n_components=c['feature_PCs'], svd_solver='full')
    rna_tr = xpca.fit_transform(xscale.transform(xtr[:, genes]))
    rna_va = xpca.transform(xscale.transform(xva[:, genes]))
    base = early_va + mean

    def predictions(rna_training):
        result = {'drift': base}
        for use_rna in (False, True):
            a = np.concatenate([source_tr, rna_training], axis=1) if use_rna else source_tr
            b = np.concatenate([source_va, rna_va], axis=1) if use_rna else source_va
            scale = StandardScaler().fit(a)
            a, b = scale.transform(a), scale.transform(b)
            family = 'RNA' if use_rna else 'Source'
            for rank, alpha in product(c['residual_ranks'], c['ridge_alphas']):
                model = Ridge(alpha=alpha).fit(a, u[:, :rank] * s[:rank])
                correction = model.predict(b) @ vt[:rank]
                for gain in c['residual_gains']:
                    result[f'{family}_r{rank}_a{alpha}_g{gain}'] = base + gain * correction
        return result
    real = predictions(rna_tr)
    shuffled = [predictions(rna_tr[np.random.default_rng(seed).permutation(len(xtr))])
                for seed in c['_role_seeds']]
    return real, shuffled


def fit_role(xtr, xva, etr, eva, ltr, core, c, role_offset):
    cc = {**c, '_role_seeds': [s + role_offset for s in c['RNA_seeds']]}
    real = {}
    controls = []
    for i, seed in enumerate(cc['_role_seeds']):
        expert = old.fit_experts(xtr, xva, etr, eva, ltr, core, seed)
        if i == 0:
            real.update({k: expert[k] for k in OLD_EXPERTS})
        control = {k: expert[{'RNA': 'PermutedRNAExpert', 'Combined': 'PermutedCombinedExpert'}.get(k, k)]
                   for k in OLD_EXPERTS}
        controls.append(control)
    # Give the simple calibration its own inner alpha selection, not just alpha10.
    e0, e1 = impute(etr, eva, fallback=float(np.nanmean(etr)))
    late_mean = np.nanmean(ltr, axis=0)
    late_mean = np.where(np.isfinite(late_mean), late_mean, np.nanmean(ltr))
    for alpha in c['ridge_alphas']:
        if alpha == 10:
            pred = real['PersistenceRidge']
        else:
            pred = old.per_drug_ridge(np.zeros((len(etr), 0)), np.zeros((len(eva), 0)),
                e0, e1, e0.mean(axis=1, keepdims=True), e1.mean(axis=1, keepdims=True),
                ltr, late_mean, alpha, 8, True, False)
        real[f'cal_a{alpha}'] = pred
        for control in controls:
            control[f'cal_a{alpha}'] = pred
    rank_real, rank_controls = low_rank_predictions(xtr, xva, etr, eva, ltr, cc)
    real.update(rank_real)
    for control, rank in zip(controls, rank_controls):
        control.update(rank)
    return real, controls


def construct_pool(inner, outer, etr, ltr, core, c):
    old_choice, _ = old.select_candidate(inner, ltr, core)
    old_w = core['nested_multi_expert']['candidates_in_order'][old_choice]
    original_i = old.blend_predictions(inner, old_w)
    original_o = old.blend_predictions(outer, old_w)
    calibrations = {f'cal_a{a}': inner[f'cal_a{a}'] for a in c['ridge_alphas']}
    cal_scores = score_pool(calibrations, etr, ltr)
    cal_choice = choose(cal_scores, tolerance=c['selection_tolerance'])
    ipool = {'PopulationAnchor': inner['PopulationAnchor'], 'OriginalCandidate': original_i, **calibrations}
    opool = {'PopulationAnchor': outer['PopulationAnchor'], 'OriginalCandidate': original_o,
             **{k: outer[k] for k in calibrations}}
    for family, third_i, third_o in [('fusion', original_i, original_o),
                                      ('sourcefusion', inner['SourceState'], outer['SourceState'])]:
        for j, w in enumerate(weights()):
            ipool[f'{family}_{j}'] = w[0] * inner['PopulationAnchor'] + w[1] * inner[cal_choice] + w[2] * third_i
            opool[f'{family}_{j}'] = w[0] * outer['PopulationAnchor'] + w[1] * outer[cal_choice] + w[2] * third_o
    for family in ('RNA', 'Source'):
        ipool[family + '_drift'] = inner['drift']
        opool[family + '_drift'] = outer['drift']
        for rank, alpha, gain in product(c['residual_ranks'], c['ridge_alphas'], c['residual_gains']):
            key = f'{family}_r{rank}_a{alpha}_g{gain}'
            ipool[key], opool[key] = inner[key], outer[key]
    if len(ipool) != c['candidate_pool_size']:
        raise ValueError('Candidate pool size differs from contract')
    scores = score_pool(ipool, etr, ltr)
    choices = {'OriginalCandidate': 'OriginalCandidate', 'TunedCalibration': cal_choice,
               'RankDrift': 'RNA_drift'}
    for name, prefix in [('CalibratedFusion', 'fusion_'), ('SourceFusion', 'sourcefusion_'),
                         ('LowRankDeltaRNA', 'RNA_'), ('LowRankDeltaSource', 'Source_')]:
        choices[name] = choose({k: v for k, v in scores.items() if k.startswith(prefix)},
                               tolerance=c['selection_tolerance'])
    choices[MAIN] = choose(scores, tolerance=c['selection_tolerance'])
    choices[DYNAMIC] = choose(scores, dynamic=True, tolerance=c['selection_tolerance'],
                              dynamic_tolerance=c['dynamic_Spearman_tolerance'])
    return opool, scores, choices, {'original_mixture': old_choice, 'calibration_alpha_key': cal_choice}
