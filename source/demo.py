"""Synthetic example exercising all four experts; not a scientific benchmark."""
from pathlib import Path
import json
import tempfile
import numpy as np
from para_drugrank.model import fit_selected, save_model, load_model
ROOT = Path(__file__).resolve().parents[1]

def synthetic():
    rng = np.random.default_rng(23)
    n, g, d = 50, 12, 10
    e, source, target = rng.normal(size=(n, g)).astype(np.float32), rng.normal(size=(n, d)).astype(np.float32), rng.normal(size=(n, d)).astype(np.float32)
    target[::7, 0], source[::9, 1] = np.nan, np.nan
    outer = np.arange(n)%5
    inner = np.full((n, 5), -1, dtype=int)
    for f in range(5):
        inner[outer != f, f] = np.arange((outer != f).sum())%4
    data = dict(expression=e, source_response=source, target_response=target, outer_fold=outer, inner_fold=inner,
                cohort=np.asarray("PRISM"), gene_names=np.asarray([f"g{i}" for i in range(g)]), drug_names=np.asarray([f"d{i}" for i in range(d)]))
    config = json.loads((ROOT / "configs/model.json").read_text())
    config["para_search"].update(expression_components=4, source_components=3, residual_ranks=[3], alphas=[1.], gamma_scales=[.5], residual_gates=[.75, 1., 1.25], top_k_equal_ensembles=[3])
    config["baseline_representation"].update(expression_components=4, source_components=3, target_residual_rank=3)
    config["cohorts"]["PRISM"]["minimum_finite_drugs"] = 5
    config["baseline_grids"]["RF"] = [dict(n_estimators=8, max_depth=4, max_features=.3, min_samples_leaf=2, n_jobs=1)]
    config["baseline_grids"]["SRMF"] = [dict(rank=3, neighbors=5, similarity_regularization=.01, source_weight=.25, ridge=.001, iterations=10, learning_rate=.02)]
    recipe = dict(cohort="PRISM", outer_fold=0, para_members=[dict(rank=3, alpha=1., gamma_scale=.5, gate=1.)],
                  baselines={"RF_SourceAug": config["baseline_grids"]["RF"][0], "SRMF_Original": config["baseline_grids"]["SRMF"][0], "SRMF_SourceAug": config["baseline_grids"]["SRMF"][0]}, transform="row_zscore", weights=[.4, .2, .2, .2])
    return data, config, recipe


if __name__ == '__main__':
    data, config, recipe = synthetic()
    train = data['outer_fold'] != 0
    model = fit_selected(data, train, recipe, config)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'synthetic_model.joblib'
        save_model(model, path)
        restored = load_model(path)
        score = restored.predict(data['expression'][~train], data['source_response'][~train],
                                 data['gene_names'], data['drug_names'])
    assert score.shape == (10, 10) and np.isfinite(score).all()
    print(json.dumps({'example': 'synthetic', 'training_rows': int(train.sum()),
                      'prediction_shape': list(score.shape), 'all_finite': True}))
