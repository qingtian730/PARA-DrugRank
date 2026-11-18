"""Refit the patient model with the original five-by-four nested protocol."""
from pathlib import Path
import argparse
import json
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import numpy as np
from longitudinal.model import fit_role, construct_pool, OLD_EXPERTS, MAIN
from longitudinal.metrics import all_scores

ROOT = Path(__file__).resolve().parents[1]


def load_patient_data(path):
    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    for name in ['expression', 'early_response', 'late_response', 'outer_fold', 'inner_fold', 'drug_names', 'gene_names']:
        if name not in data:
            raise ValueError('Missing field: ' + name)
    x, early, late = (data[k] for k in ['expression','early_response','late_response'])
    outer, inner = data['outer_fold'], data['inner_fold']
    n = len(x)
    if x.ndim != 2 or early.ndim != 2 or early.shape != late.shape or len(early) != n:
        raise ValueError('Incompatible patient arrays')
    if not np.isfinite(x).all() or np.isinf(early).any() or np.isinf(late).any():
        raise ValueError('Expression must be finite; responses may contain NaN but not infinity')
    if len(data['gene_names']) != x.shape[1] or len(data['drug_names']) != early.shape[1]:
        raise ValueError('Axis width mismatch')
    if np.any((np.isfinite(early) & np.isfinite(late)).sum(axis=1) < 50):
        raise ValueError('Each patient requires at least 50 jointly measured drugs')
    if outer.shape != (n,) or inner.shape != (n,5) or set(outer.tolist()) != set(range(5)):
        raise ValueError('Expected fixed five outer/four inner folds')
    for fold in range(5):
        if set(inner[outer != fold,fold].tolist()) != set(range(4)) or np.any(inner[outer == fold,fold] != -1):
            raise ValueError('Inner split crosses an outer held-out boundary')
    return data


def train(data, config, folds, output):
    x, early, late, outer, inner = (data[k] for k in ['expression','early_response','late_response','outer_fold','inner_fold'])
    core, c = config['core'], config['study']
    predictions = {m:np.full_like(late, np.nan) for m in [*OLD_EXPERTS, MAIN]}
    shuffled = [np.full_like(late, np.nan) for _ in c['RNA_seeds']]
    logs = []
    for fold in folds:
        train_rows, valid = np.flatnonzero(outer != fold), np.flatnonzero(outer == fold)
        assignment = inner[train_rows,fold]
        real_i, controls_i = {}, [{} for _ in c['RNA_seeds']]
        for j in range(4):
            it, iv = train_rows[assignment != j], train_rows[assignment == j]
            real, controls = fit_role(x[it], x[iv], early[it], early[iv], late[it], core, c, fold*100+j)
            for destination, values in [(real_i,real), *zip(controls_i,controls)]:
                for key, value in values.items():
                    if key not in destination:
                        destination[key] = np.full_like(late[train_rows], np.nan)
                    destination[key][assignment == j] = value
        real_o, controls_o = fit_role(x[train_rows],x[valid],early[train_rows],early[valid],late[train_rows],core,c,fold*100+90)
        pool, scores, choices, parents = construct_pool(real_i,real_o,early[train_rows],late[train_rows],core,c)
        predictions[MAIN][valid] = pool[choices[MAIN]]
        for method in OLD_EXPERTS:
            predictions[method][valid] = real_o[method]
        permutation_choices = []
        for seed_i, (ci,co) in enumerate(zip(controls_i,controls_o)):
            cp, _, cc, _ = construct_pool(ci,co,early[train_rows],late[train_rows],core,c)
            shuffled[seed_i][valid] = cp[cc[MAIN]]
            permutation_choices.append(cc[MAIN])
        logs.append({'outer_fold':fold, 'selected':choices[MAIN], 'parents':parents,
                     'inner_candidate_scores':scores, 'RNA_shuffle_choices':permutation_choices})
        print(json.dumps({'outer_fold_completed':fold, 'held_out_patients':len(valid), 'selected':choices[MAIN]}), flush=True)
    rows = np.flatnonzero(np.isin(outer,folds))
    if not all(np.isfinite(p[rows]).all() for p in [*predictions.values(),*shuffled]):
        raise ValueError('Missing held-out predictions')
    scores = {m:all_scores(early[rows],late[rows],p[rows]) for m,p in predictions.items()}
    repeated = [all_scores(early[rows],late[rows],p[rows]) for p in shuffled]
    scores[MAIN+'_RNAShuffle'] = {k:np.stack([r[k] for r in repeated]).mean(axis=0) for k in repeated[0]}
    summary = {'patients':len(rows), 'drugs':late.shape[1], 'folds':folds,
               'metrics':{m:{k:float(np.nanmean(v)) for k,v in values.items()} for m,values in scores.items()},
               'selection_logs':logs}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'held_out.npz', rows=rows, drug_names=data['drug_names'],
                        **{m:p[rows] for m,p in predictions.items()}, RNA_shuffle_predictions=np.stack([p[rows] for p in shuffled]),
                        **{m+'__'+k:v for m,values in scores.items() for k,v in values.items()})
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/patient.json')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--folds',type=int,nargs='+',choices=range(5),default=list(range(5)))
    args=p.parse_args()
    if len(set(args.folds)) != len(args.folds):
        p.error('--folds contains duplicates')
    if any((args.output_dir/name).exists() for name in ['held_out.npz','summary.json']):
        p.error('Choose a new output directory')
    result=train(load_patient_data(args.data),json.loads(args.config.read_text()),args.folds,args.output_dir)
    print(json.dumps(result['metrics'][MAIN]))
