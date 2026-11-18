"""Train all fixed outer folds and export held-out predictions and metrics."""
from pathlib import Path
import argparse
import hashlib
import json
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '2')
import numpy as np
from para_drugrank.data import load_data
from para_drugrank.model import fit_selected, save_model, score_predictions
from para_drugrank.panels import select_panel
from para_drugrank.training import select_recipe
from train import frozen_recipe

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', choices=['prism', 'gdsc2', 'ctrpv21'], required=True)
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data/processed')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--panel', choices=['common', 'full', 'custom'], default='common')
    p.add_argument('--folds', type=int, nargs='+', choices=range(5), default=list(range(5)))
    selection = p.add_mutually_exclusive_group()
    selection.add_argument('--select-recipe', action='store_true', help='Select using the four inner folds (default)')
    selection.add_argument('--recipe-file', type=Path, help='Refit a separately supplied cohort/fold recipe catalogue')
    args = p.parse_args()
    if len(set(args.folds)) != len(args.folds):
        p.error('--folds contains duplicates')
    data_path = args.data_dir / (args.cohort + '.npz')
    data = select_panel(load_data(data_path), args.panel)
    config = json.loads((ROOT / 'configs/model.json').read_text())
    cohort = str(data['cohort'].item())
    minimum = config['cohorts'][cohort]['minimum_finite_drugs']
    destinations = [args.output_dir / f'fold_{fold}.{ext}' for fold in args.folds for ext in ['joblib', 'npz', 'json']]
    if any(f.exists() for f in [*destinations, args.output_dir/'summary.json']):
        p.error('Choose a new output directory; existing results are never overwritten')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_out, predictions = [], []
    for fold in args.folds:
        if args.recipe_file:
            recipe = frozen_recipe(args.recipe_file, cohort, fold)
        else:
            recipe = select_recipe(data, fold, config, progress=lambda x: print(x, flush=True))
        test = data['outer_fold'] == fold
        model = fit_selected(data, ~test, recipe, config)
        pred = model.predict(data['expression'][test], data['source_response'][test], data['gene_names'], data['drug_names'])
        scores = score_predictions(data['target_response'][test], pred, minimum)
        digest = save_model(model, args.output_dir / f'fold_{fold}.joblib')
        np.savez_compressed(args.output_dir / f'fold_{fold}.npz', rows=np.flatnonzero(test), prediction=pred,
                            drug_names=data['drug_names'], **scores)
        report = {'fold':fold, 'recipe':recipe, 'checkpoint_sha256':digest,
                  'metrics':{k:float(v.mean()) for k,v in scores.items()}}
        (args.output_dir / f'fold_{fold}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        rows_out.extend(np.flatnonzero(test).tolist())
        predictions.append(pred)
        print(json.dumps({'cohort':cohort, **report['metrics'], 'fold':fold}), flush=True)
    order = np.argsort(rows_out)
    rows = np.asarray(rows_out)[order]
    pred = np.concatenate(predictions)[order]
    metrics = score_predictions(data['target_response'][rows], pred, minimum)
    summary = {'cohort':cohort, 'panel':args.panel, 'cells':len(rows), 'drugs':len(data['drug_names']),
               'folds':args.folds, 'selection':'frozen_recipe_refit' if args.recipe_file else 'nested_inner_selection',
               'metrics':{k:float(v.mean()) for k,v in metrics.items()},
               'input_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest()}
    np.savez_compressed(args.output_dir / 'held_out.npz', rows=rows, prediction=pred, drug_names=data['drug_names'], **metrics)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
