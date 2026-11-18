"""Patient experts and nested selection, preserving the study computations."""
from __future__ import annotations
import hashlib
import math
from typing import Any, Mapping
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def safe_spearman(x: np.ndarray, y: np.ndarray, minimum: int = 3) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < minimum:
        return math.nan
    x_use = x[mask]
    y_use = y[mask]
    if float(np.std(x_use)) <= 0.0 or float(np.std(y_use)) <= 0.0:
        return 0.0
    return float(spearmanr(x_use, y_use).statistic)


class LongitudinalModelError(RuntimeError):
    """Raised when a frozen longitudinal model invariant drifts."""


def hash_round_robin(values: list[str], folds: int, salt: str) -> tuple[np.ndarray, str]:
    ordered = sorted(
        range(len(values)),
        key=lambda index: hashlib.sha256(f"{salt}|{values[index]}".encode()).hexdigest(),
    )
    assignment = np.empty(len(values), dtype=int)
    for position, index in enumerate(ordered):
        assignment[index] = position % int(folds)
    lines = "\n".join(
        sorted(
            hashlib.sha256(f"{salt}|{values[index]}|{assignment[index]}".encode()).hexdigest()
            for index in range(len(values))
        )
    )
    return assignment, hashlib.sha256((lines + "\n").encode()).hexdigest()


def ndcg_at_k(truth: np.ndarray, prediction: np.ndarray, k: int, minimum: int) -> float:
    mask = np.isfinite(truth) & np.isfinite(prediction)
    if int(mask.sum()) < int(minimum):
        return math.nan
    y = truth[mask].astype(float)
    p = prediction[mask].astype(float)
    relevance = y - float(np.min(y))
    if float(np.max(relevance)) <= 0.0:
        return 0.0
    use = min(int(k), len(y))
    discount = 1.0 / np.log2(np.arange(2, use + 2, dtype=float))
    order = np.argsort(-p, kind="mergesort")[:use]
    ideal = np.argsort(-relevance, kind="mergesort")[:use]
    denominator = float(np.sum(relevance[ideal] * discount))
    return float(np.sum(relevance[order] * discount) / denominator) if denominator > 0 else 0.0


def patient_scores(
    truth: np.ndarray,
    prediction: np.ndarray,
    minimum: int,
) -> tuple[np.ndarray, np.ndarray]:
    spearman = np.asarray(
        [safe_spearman(truth[row], prediction[row], minimum) for row in range(len(truth))],
        dtype=float,
    )
    ndcg = np.asarray(
        [ndcg_at_k(truth[row], prediction[row], 20, minimum) for row in range(len(truth))],
        dtype=float,
    )
    return spearman, ndcg


def blend_predictions(
    experts: Mapping[str, np.ndarray], weights: Mapping[str, float]
) -> np.ndarray:
    if not weights or abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-8:
        raise LongitudinalModelError("longitudinal ensemble weights do not sum to one")
    shape = next(iter(experts.values())).shape
    value = np.zeros(shape, dtype=float)
    for name, weight in weights.items():
        if name not in experts or experts[name].shape != shape:
            raise LongitudinalModelError("longitudinal ensemble expert drift")
        value += float(weight) * experts[name]
    return value


def train_latents(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    early_train: np.ndarray,
    early_validation: np.ndarray,
    contract: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    train_drug_mean = np.nanmean(early_train, axis=0)
    global_mean = float(np.nanmean(early_train))
    train_drug_mean = np.where(np.isfinite(train_drug_mean), train_drug_mean, global_mean)
    early_train_imputed = np.where(np.isfinite(early_train), early_train, train_drug_mean)
    early_validation_imputed = np.where(
        np.isfinite(early_validation), early_validation, train_drug_mean
    )
    source_scaler = StandardScaler().fit(early_train_imputed)
    source_train_scaled = source_scaler.transform(early_train_imputed)
    source_validation_scaled = source_scaler.transform(early_validation_imputed)
    source_components = int(contract["preprocessing"]["source_response_PCA_components"])
    source_pca = PCA(n_components=source_components, svd_solver="full")
    source_train = source_pca.fit_transform(source_train_scaled)
    source_validation = source_pca.transform(source_validation_scaled)

    top_genes = min(int(contract["preprocessing"]["expression_gene_selection"].split("_")[1]), x_train.shape[1])
    variance = np.var(x_train.astype(float), axis=0)
    selected = np.argsort(-variance, kind="mergesort")[:top_genes]
    expression_scaler = StandardScaler().fit(x_train[:, selected])
    expression_train_scaled = expression_scaler.transform(x_train[:, selected])
    expression_validation_scaled = expression_scaler.transform(x_validation[:, selected])
    expression_components = int(contract["preprocessing"]["expression_PCA_components"])
    expression_pca = PCA(n_components=expression_components, svd_solver="full")
    expression_train = expression_pca.fit_transform(expression_train_scaled)
    expression_validation = expression_pca.transform(expression_validation_scaled)
    return {
        "early_train": early_train_imputed,
        "early_validation": early_validation_imputed,
        "source_train": source_train,
        "source_validation": source_validation,
        "expression_train": expression_train,
        "expression_validation": expression_validation,
        "patient_mean_train": early_train_imputed.mean(axis=1, keepdims=True),
        "patient_mean_validation": early_validation_imputed.mean(axis=1, keepdims=True),
    }


def per_drug_ridge(
    base_train: np.ndarray,
    base_validation: np.ndarray,
    early_train: np.ndarray,
    early_validation: np.ndarray,
    patient_mean_train: np.ndarray,
    patient_mean_validation: np.ndarray,
    target_train: np.ndarray,
    fallback: np.ndarray,
    alpha: float,
    minimum: int,
    include_early: bool,
    include_patient_mean: bool,
) -> np.ndarray:
    prediction = np.tile(fallback, (len(base_validation), 1)).astype(float)
    for drug in range(target_train.shape[1]):
        train_blocks = [base_train]
        validation_blocks = [base_validation]
        if include_early:
            train_blocks.append(early_train[:, drug : drug + 1])
            validation_blocks.append(early_validation[:, drug : drug + 1])
        if include_patient_mean:
            train_blocks.append(patient_mean_train)
            validation_blocks.append(patient_mean_validation)
        train_features = np.concatenate(train_blocks, axis=1)
        validation_features = np.concatenate(validation_blocks, axis=1)
        observed = np.isfinite(target_train[:, drug])
        if int(observed.sum()) < int(minimum):
            continue
        scaler = StandardScaler().fit(train_features[observed])
        model = Ridge(alpha=float(alpha))
        model.fit(scaler.transform(train_features[observed]), target_train[observed, drug])
        prediction[:, drug] = model.predict(scaler.transform(validation_features))
    return prediction


def fit_experts(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    early_train: np.ndarray,
    early_validation: np.ndarray,
    late_train: np.ndarray,
    contract: Mapping[str, Any],
    seed: int,
) -> dict[str, np.ndarray]:
    latent = train_latents(x_train, x_validation, early_train, early_validation, contract)
    population = np.nanmean(late_train, axis=0)
    population = np.where(np.isfinite(population), population, float(np.nanmean(late_train)))
    alpha = float(contract["learner"]["alpha"])
    minimum = int(contract["learner"]["minimum_training_patients_per_drug"])
    empty_train = np.zeros((len(x_train), 0), dtype=float)
    empty_validation = np.zeros((len(x_validation), 0), dtype=float)
    persistence_ridge = per_drug_ridge(
        empty_train,
        empty_validation,
        latent["early_train"],
        latent["early_validation"],
        latent["patient_mean_train"],
        latent["patient_mean_validation"],
        late_train,
        population,
        alpha,
        minimum,
        True,
        False,
    )
    source = per_drug_ridge(
        latent["source_train"],
        latent["source_validation"],
        latent["early_train"],
        latent["early_validation"],
        latent["patient_mean_train"],
        latent["patient_mean_validation"],
        late_train,
        population,
        alpha,
        minimum,
        True,
        True,
    )
    rna = per_drug_ridge(
        latent["expression_train"],
        latent["expression_validation"],
        latent["early_train"],
        latent["early_validation"],
        latent["patient_mean_train"],
        latent["patient_mean_validation"],
        late_train,
        population,
        alpha,
        minimum,
        False,
        False,
    )
    combined_train = np.concatenate(
        [latent["source_train"], latent["expression_train"]], axis=1
    )
    combined_validation = np.concatenate(
        [latent["source_validation"], latent["expression_validation"]], axis=1
    )
    combined = per_drug_ridge(
        combined_train,
        combined_validation,
        latent["early_train"],
        latent["early_validation"],
        latent["patient_mean_train"],
        latent["patient_mean_validation"],
        late_train,
        population,
        alpha,
        minimum,
        True,
        True,
    )
    permutation = np.random.default_rng(int(seed)).permutation(len(x_train))
    permuted_expression_train = latent["expression_train"][permutation]
    permuted_rna = per_drug_ridge(
        permuted_expression_train,
        latent["expression_validation"],
        latent["early_train"],
        latent["early_validation"],
        latent["patient_mean_train"],
        latent["patient_mean_validation"],
        late_train,
        population,
        alpha,
        minimum,
        False,
        False,
    )
    permuted_combined = per_drug_ridge(
        np.concatenate([latent["source_train"], permuted_expression_train], axis=1),
        combined_validation,
        latent["early_train"],
        latent["early_validation"],
        latent["patient_mean_train"],
        latent["patient_mean_validation"],
        late_train,
        population,
        alpha,
        minimum,
        True,
        True,
    )
    return {
        "PopulationAnchor": np.tile(population, (len(x_validation), 1)),
        "Persistence": latent["early_validation"],
        "PersistenceRidge": persistence_ridge,
        "SourceState": source,
        "RNA": rna,
        "Combined": combined,
        "PermutedRNAExpert": permuted_rna,
        "PermutedCombinedExpert": permuted_combined,
    }


def select_candidate(
    inner_experts: Mapping[str, np.ndarray],
    truth: np.ndarray,
    contract: Mapping[str, Any],
) -> tuple[str, dict[str, float]]:
    candidates = contract["nested_multi_expert"]["candidates_in_order"]
    scores: dict[str, float] = {}
    minimum = int(contract["cohort_freeze"]["minimum_joint_finite_drugs_per_patient"])
    for name, weights in candidates.items():
        prediction = blend_predictions(inner_experts, weights)
        values, _ = patient_scores(truth, prediction, minimum)
        scores[name] = float(np.nanmean(values))
    best = max(scores.values())
    tolerance = float(contract["nested_multi_expert"]["selection_tolerance"])
    selected = next(name for name in candidates if scores[name] >= best - tolerance)
    return selected, scores


def permuted_candidate(
    experts: Mapping[str, np.ndarray], weights: Mapping[str, float]
) -> np.ndarray:
    mapped = dict(experts)
    mapped["RNA"] = experts["PermutedRNAExpert"]
    mapped["Combined"] = experts["PermutedCombinedExpert"]
    return blend_predictions(mapped, weights)
