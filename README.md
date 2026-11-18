# PARA-DrugRank

PARA-DrugRank uses basal gene expression and drug responses from a source screen to rank drugs for a cell line in a target screen. Four experts generate predictions. Within each outer training set, four-fold inner validation selects their configurations, the within-cell score transformation and the weights used to combine their predictions. A separate longitudinal model predicts later ex vivo rankings from earlier samples in BeatAML.

This release contains training and evaluation code, search settings and a synthetic example. Supply the study inputs locally. Training writes new model files and predictions to the output directory you choose. [中文运行说明](RUNNING_CN.md)

## Installation and example

Use Python 3.12 in a fresh virtual environment. For the common drug panels in the cell-line comparison, run the following from the repository root.

```sh
python -m pip install -r requirements.txt
python source/demo.py
python -m unittest discover -s tests -v
```

The example generates synthetic inputs, fits all four cell-line experts, reloads the fitted model and checks a held-out prediction. The tests also check axis alignment and the exclusion of outer-fold targets from model selection.

## Cell-line training

Place a prepared NPZ on your machine. The commands below train and evaluate PRISM outer fold 0. The first command selects the model on the four inner folds and then refits it on the outer training cells.

```sh
python source/train.py --data data/processed/prism.npz --outer-fold 0 --output outputs/prism_fold0.joblib
python source/evaluate.py --data data/processed/prism.npz --checkpoint outputs/prism_fold0.joblib
python source/run_all_folds.py --cohort prism --data-dir data/processed --output-dir outputs/prism_cv
```

Use `gdsc2` or `ctrpv21` for the other screens. The common panels contain 105 drugs for PRISM, 100 for GDSC2 and 92 for CTRPv2.1. The common GDSC2 comparison excludes `iap5620` from the 101-drug full panel. The common-panel loader checks the ordered drug names against the study panel. For the full-panel experiments, install `requirements-full-panel.txt` in a separate environment and pass `--panel full` to training and evaluation. For a locally defined axis, use `--panel custom`.

| NPZ field | Contents |
|---|---|
| `expression` | Finite basal expression matrix, cells × genes |
| `source_response`, `target_response` | Matched cells × drugs matrices of negative AUC, with higher values denoting greater sensitivity and NaN marking missing measurements |
| `gene_names`, `drug_names` | Unique ordered names matching the matrix columns |
| `cohort` | Scalar string `PRISM`, `GDSC2` or `CTRPv2.1` |
| `outer_fold` | One outer-fold index from 0 to 4 per cell |
| `inner_fold` | Cells × 5. Column *k* holds the four inner-fold indices within outer training set *k*, with −1 on its held-out cells |

To predict with a locally trained checkpoint, provide a query NPZ containing `expression`, `source_response`, `gene_names` and `drug_names`. Use the same ordered axes as the checkpoint. When reproducing a held-out prediction, keep the original query batch together because SRMF uses similarities within the batch.

```sh
python source/predict.py --checkpoint outputs/prism_fold0.joblib --data data/query.npz --output outputs/query_scores.npz
```

## Longitudinal training

Use a separate Python 3.12 environment for the patient model.

```sh
python -m pip install -r requirements-patient.txt
python source/train_patient.py --data data/processed/beataml.npz --output-dir outputs/beataml
```

Each row represents one patient. The NPZ contains earlier `expression`, paired `early_response` and `late_response` matrices on the same drug axis, `gene_names`, `drug_names`, `outer_fold` and `inner_fold`. Expression uses within-patient inverse-normal gene ranks. Responses use negative fitted AUC. Each pair has at least 50 drugs measured at both times. Both samples from a patient share the same split. Training selects among 73 candidates by inner-validation Spearman correlation and evaluates held-out patients on Spearman correlation, NDCG@20 and recall of newly entering top-20 drugs.

Cell-line search grids and seeds are in [model.json](configs/model.json). Patient-model settings and RNA shuffle seeds are in [patient.json](configs/patient.json).
