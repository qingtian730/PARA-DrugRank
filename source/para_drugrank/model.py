"""Frozen PARA/RF/SRMF components with fitted-state serialization.

This is a compact extraction of the terminal architecture, not a neural rewrite.
Both PCA fit_transform and transform paths, float conversions, seed derivation,
and query-batch SRMF bandwidth behavior follow the experiment implementation.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from .data import axis_digest

EXPERTS = ("PARA", "RF_SourceAug", "SRMF_Original", "SRMF_SourceAug")


def finite_mean(matrix):
    counts = np.isfinite(matrix).sum(axis=0)
    if np.any(counts == 0):
        raise ValueError("A training drug has no observed value")
    return np.nansum(matrix, axis=0, dtype=np.float64) / counts


def stable_seed(base, *parts):
    text = "|".join([str(int(base)), *(str(x) for x in parts)])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def factor_basis(residual, rank):
    finite = np.isfinite(residual)
    counts = finite.sum(axis=0).astype(float)
    center = np.divide(np.nansum(residual, axis=0, dtype=np.float64), counts,
                       out=np.zeros(residual.shape[1]), where=counts > 0)
    _, singular, right = np.linalg.svd(np.where(finite, residual - center, 0.0), full_matrices=False)
    if int(rank) > len(right):
        raise ValueError("Requested residual rank unavailable")
    return center, right[:int(rank)], singular[:int(rank)]


def project_scores(residual, center, basis):
    result = np.zeros((len(residual), len(basis)), dtype=np.float64)
    for row in range(len(residual)):
        observed = np.isfinite(residual[row])
        if observed.sum() < len(basis):
            continue
        design = basis[:, observed].T
        gram = design.T @ design
        gram.flat[::len(gram) + 1] += 1e-3
        result[row] = np.linalg.solve(gram, design.T @ (residual[row, observed] - center[observed]))
    return result


@dataclass
class PCAState:
    scaler: object
    projector: object
    state_scaler: object
    whiten: bool
    components_strides: tuple

    @classmethod
    def fit(cls, matrix, components, seed, standardized):
        if standardized:
            if min(int(components), len(matrix)-1, matrix.shape[1]) != int(components):
                raise ValueError("Requested standardized PCA rank unavailable")
            scaler = StandardScaler().fit(matrix)
            scaled = scaler.transform(matrix)
            projector = PCA(n_components=int(components), svd_solver="randomized", random_state=int(seed))
            state = projector.fit_transform(scaled).astype(np.float64)
            state_scaler = StandardScaler().fit(state)
            return cls(scaler, projector, state_scaler, False, projector.components_.strides), state_scaler.transform(state)
        components = min(int(components), len(matrix)-1, matrix.shape[1])
        if components < 1:
            raise ValueError("PCA rank unavailable")
        projector = PCA(n_components=components, whiten=True, svd_solver="randomized", random_state=int(seed))
        state = projector.fit_transform(matrix.astype(np.float64, copy=False)).astype(np.float32)
        return cls(None, projector, None, True, projector.components_.strides), state

    def restore_layout(self):
        # sklearn's randomized PCA components can be a sliced, noncontiguous
        # view. Joblib otherwise restores it as contiguous. Changing strides
        # changes the BLAS path and can perturb nearly tied percentile ranks.
        value = self.projector.components_
        strides = tuple(self.components_strides)
        if value.strides == strides:
            return
        if any(step < 0 for step in strides):
            raise ValueError("Unsupported negative PCA component stride")
        size = value.itemsize+sum((width-1)*step for width, step in zip(value.shape, strides))
        restored = np.ndarray(value.shape, dtype=value.dtype, buffer=bytearray(size), strides=strides)
        restored[...] = value
        self.projector.components_ = restored

    def transform(self, matrix):
        value = self.scaler.transform(matrix) if self.scaler is not None else matrix.astype(np.float64, copy=False)
        value = self.projector.transform(value)
        if self.state_scaler is not None:
            value = self.state_scaler.transform(value.astype(np.float64))
        return value.astype(np.float32) if self.whiten else value.astype(np.float64)


@dataclass
class ParaRepresentation:
    source_imputation: np.ndarray
    target_mean: np.ndarray
    center: np.ndarray
    basis: np.ndarray
    expression_pca: PCAState
    source_pca: PCAState
    source_mean: np.ndarray
    projection_scaler: object
    final_scaler: object
    interactions: bool

    @staticmethod
    def parts(e, s, p, interactions):
        values = [e, s, p]
        if interactions:
            matched = min(8, e.shape[1], s.shape[1])
            values.append(e[:, :matched] * s[:, :matched])
            matched = min(p.shape[1], e.shape[1])
            values.append(p[:, :matched] * e[:, :matched])
        return np.hstack(values).astype(np.float64)

    @classmethod
    def fit(cls, expression, source, target, search, rank, seed):
        imputation = finite_mean(source)
        source = np.where(np.isfinite(source), source, imputation).astype(np.float32)
        mean = finite_mean(target).astype(np.float32)
        residual = target.astype(np.float64) - mean[None, :].astype(np.float64)
        center, basis, _ = factor_basis(residual, rank)
        scores = project_scores(residual, center, basis)
        epca, e = PCAState.fit(expression, search["expression_components"], seed+1, True)
        spca, s = PCAState.fit(source, search["source_components"], seed+2, True)
        source_mean = source.mean(axis=0, dtype=np.float64)
        p = (source.astype(np.float64)-source_mean) @ basis.T
        projection_scaler = StandardScaler().fit(p)
        p = projection_scaler.transform(p)
        parts = cls.parts(e, s, p, search["interactions"])
        scaler = StandardScaler().fit(parts)
        state = cls(imputation, mean, center, basis, epca, spca, source_mean,
                    projection_scaler, scaler, bool(search["interactions"]))
        return state, scaler.transform(parts).astype(np.float32), scores

    def transform(self, expression, source):
        source = np.where(np.isfinite(source), source, self.source_imputation).astype(np.float32)
        e, s = self.expression_pca.transform(expression), self.source_pca.transform(source)
        p = self.projection_scaler.transform((source.astype(np.float64)-self.source_mean) @ self.basis.T)
        return self.final_scaler.transform(self.parts(e, s, p, self.interactions)).astype(np.float32)


@dataclass
class ParaMember:
    representation: ParaRepresentation
    kernel_model: object
    factor_mean: np.ndarray
    factor_scale: np.ndarray
    gate: float

    @classmethod
    def fit(cls, representation, features, scores, configuration):
        mean = scores.mean(axis=0, dtype=np.float64)
        scale = scores.std(axis=0, dtype=np.float64)
        scale = np.where(scale > 1e-8, scale, 1.0)
        # Preserve the mixed float64 norms / float32 matmul in the frozen gamma.
        squared = np.sum(features.astype(np.float64)**2, axis=1, keepdims=True)
        distance = squared+squared.T-2.0*(features @ features.T)
        upper = distance[np.triu_indices(len(features), 1)]
        positive = upper[upper > 1e-12]
        median = float(np.median(positive)) if len(positive) else 1.0
        gamma = float(configuration["gamma_scale"])/median
        model = KernelRidge(alpha=float(configuration["alpha"]), kernel="rbf", gamma=gamma)
        model.fit(features, (scores-mean)/scale)
        return cls(representation, model, mean, scale, float(configuration["gate"]))

    def predict(self, expression, source):
        factors = self.kernel_model.predict(self.representation.transform(expression, source))
        factors = factors.astype(np.float64)*self.factor_scale + self.factor_mean
        residual = self.representation.center + factors @ self.representation.basis
        return (self.representation.target_mean.astype(np.float64)+self.gate*residual).astype(np.float32)


@dataclass
class BaselineRepresentation:
    source_imputation: np.ndarray
    target_mean: np.ndarray
    expression_pca: PCAState
    source_pca: PCAState
    source_mean: np.ndarray
    basis: np.ndarray

    @classmethod
    def fit(cls, expression, source, target, representation, seed):
        imputation = finite_mean(source)
        source = np.where(np.isfinite(source), source, imputation).astype(np.float32)
        epca, e = PCAState.fit(expression, representation["expression_components"], seed, False)
        spca, s = PCAState.fit(source, representation["source_components"], seed+1, False)
        mean = finite_mean(target.astype(np.float32)).astype(np.float32)
        _, basis, _ = factor_basis(target.astype(np.float32).astype(np.float64)-mean, representation["target_residual_rank"])
        source_mean = source.mean(axis=0, dtype=np.float64)
        p = ((source.astype(np.float64)-source_mean) @ basis.T).astype(np.float32)
        return cls(imputation, mean, epca, spca, source_mean, basis), (e, s, source, p)

    def transform(self, expression, source):
        source = np.where(np.isfinite(source), source, self.source_imputation).astype(np.float32)
        e, s = self.expression_pca.transform(expression), self.source_pca.transform(source)
        p = ((source.astype(np.float64)-self.source_mean) @ self.basis.T).astype(np.float32)
        return e, s, source, p


def pair_features(values, mean, variant="SourceAug"):
    e, s, source, projection = values
    n, d = source.shape
    cell = e if variant == "Original" else np.concatenate([e, s], axis=1)
    blocks = [np.repeat(cell.astype(np.float32), d, axis=0)]
    if variant == "SourceAug":
        blocks.append(np.repeat(projection.astype(np.float32), d, axis=0))
    blocks.append(np.tile(np.eye(d, dtype=np.float32), (n, 1)))
    if variant == "SourceAug":
        blocks.append(source.reshape(-1, 1).astype(np.float32))
    blocks.append(np.tile(mean.astype(np.float32), n)[:, None])
    return np.concatenate(blocks, axis=1)


def squared_distances(left, right):
    left, right = left.astype(np.float64, copy=False), right.astype(np.float64, copy=False)
    return np.maximum(np.sum(left*left, axis=1)[:, None]+np.sum(right*right, axis=1)[None, :]-2.*left @ right.T, 0.)


def kernel_weights(query, reference, neighbors, exclude_diagonal=False):
    distance = squared_distances(query, reference)
    if exclude_diagonal and query.shape == reference.shape and np.array_equal(query, reference):
        np.fill_diagonal(distance, np.inf)
    use = min(int(neighbors), reference.shape[0]-(1 if exclude_diagonal else 0))
    if use < 1:
        raise ValueError("Similarity has no neighbor")
    indices = np.argpartition(distance, use-1, axis=1)[:, :use]
    selected = np.take_along_axis(distance, indices, axis=1)
    finite = selected[np.isfinite(selected) & (selected > 0)]
    bandwidth = max(float(np.median(finite)) if finite.size else 1., 1e-8)
    weights = np.exp(-selected/bandwidth)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    result = np.zeros_like(distance, dtype=np.float64)
    np.put_along_axis(result, indices, weights, axis=1)
    return result


def laplacian(state, neighbors):
    weights = kernel_weights(state, state, neighbors, True)
    weights = np.maximum(weights, weights.T)
    degree = weights.sum(axis=1)
    inverse = np.zeros_like(degree)
    positive = degree > 0
    inverse[positive] = 1./np.sqrt(degree[positive])
    return np.eye(len(state), dtype=np.float64)-inverse[:, None]*weights*inverse[None, :]


def adam_step(value, gradient, first, second, step, learning_rate):
    first = .9*first+.1*gradient
    second = .999*second+.001*gradient*gradient
    value = value-float(learning_rate)*(first/(1.-.9**step))/(np.sqrt(second/(1.-.999**step))+1e-8)
    return value, first, second


@dataclass
class SRMFState:
    variant: str
    config: dict
    expression_reference: np.ndarray
    source_reference: object
    cell_factors: np.ndarray
    drug_factors: np.ndarray
    target_mean: np.ndarray

    @classmethod
    def fit(cls, values, target, mean, variant, config, seed):
        e, s, source, _ = values
        residual = target.astype(np.float64)-mean[None, :]
        mask = np.isfinite(residual)
        filled = np.where(mask, residual, 0.)
        left, singular, right = np.linalg.svd(filled, full_matrices=False)
        rank = min(int(config["rank"]), len(singular))
        scale = np.sqrt(np.maximum(singular[:rank], 1e-8))
        u, v = left[:, :rank]*scale[None, :], right[:rank].T*scale[None, :]
        rng = np.random.default_rng(int(seed))
        u += rng.normal(0., 1e-5, size=u.shape)
        v += rng.normal(0., 1e-5, size=v.shape)
        elap = laplacian(e, config["neighbors"])
        if variant == "SourceAug":
            weight = float(config["source_weight"])
            cell_lap = (1.-weight)*elap+weight*laplacian(s, config["neighbors"])
            drug_lap = laplacian(source.T, min(int(config["neighbors"]), source.shape[1]-1))
        else:
            cell_lap, drug_lap = elap, np.zeros((len(v), len(v)), dtype=np.float64)
        first_u, second_u, first_v, second_v = (np.zeros_like(u), np.zeros_like(u), np.zeros_like(v), np.zeros_like(v))
        rows, columns = np.maximum(mask.sum(axis=1), 1)[:, None], np.maximum(mask.sum(axis=0), 1)[:, None]
        reg, ridge = float(config["similarity_regularization"]), float(config["ridge"])
        for step in range(1, int(config["iterations"])+1):
            error = np.where(mask, u @ v.T-filled, 0.)
            gu = 2.*(error @ v)/rows+reg*(cell_lap @ u)+ridge*u
            gv = 2.*(error.T @ u)/columns+reg*(drug_lap @ v)+ridge*v
            u, first_u, second_u = adam_step(u, gu, first_u, second_u, step, config["learning_rate"])
            v, first_v, second_v = adam_step(v, gv, first_v, second_v, step, config["learning_rate"])
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            raise ValueError("SRMF optimization diverged")
        return cls(variant, dict(config), e.copy(), s.copy() if variant == "SourceAug" else None, u, v, mean.copy())

    def predict(self, values):
        e, s, _, _ = values
        cross = kernel_weights(e, self.expression_reference, self.config["neighbors"])
        if self.variant == "SourceAug":
            weight = float(self.config["source_weight"])
            cross = (1.-weight)*cross+weight*kernel_weights(s, self.source_reference, self.config["neighbors"])
            cross /= np.maximum(cross.sum(axis=1, keepdims=True), 1e-12)
        return (self.target_mean[None, :]+(cross @ self.cell_factors) @ self.drug_factors.T).astype(np.float32)


def transform_prediction(array, transform):
    value = np.asarray(array, dtype=np.float64)
    if transform == "row_zscore":
        scale = value.std(axis=1, keepdims=True)
        result = (value-value.mean(axis=1, keepdims=True))/np.where(scale > 1e-12, scale, 1.)
    elif transform == "row_percentile_rank":
        result = np.array([(rankdata(row, method="average")-1.)/max(value.shape[1]-1, 1) for row in value])
    else:
        raise ValueError(f"Unknown transform: {transform}")
    return result.astype(np.float32)


def blend_predictions(predictions, transform, weights):
    if len(weights) != 4 or min(weights) < -1e-12 or abs(sum(weights)-1.) > 1e-8:
        raise ValueError("Fusion weights must lie on the four-expert simplex")
    transformed = [transform_prediction(predictions[name], transform) for name in EXPERTS]
    return sum(float(weight)*value.astype(np.float64) for weight, value in zip(weights, transformed)).astype(np.float32)


def score_predictions(truth, prediction, minimum=1):
    """Current multi-K evaluation: truth-only mask and stable mergesort ties.

    Returns per-row metrics so callers aggregate by the independent ModelID
    unit. Scores are not automatically written or included in public releases.
    """
    from .training import ndcg_rows
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or truth.ndim != 2 or not np.isfinite(prediction).all():
        raise ValueError("Invalid truth/prediction shape or nonfinite prediction")
    result = {f"NDCG{k}": ndcg_rows(truth, prediction, minimum, k) for k in (5, 10, 20, 50)}
    result.update({f"Recall{k}": np.empty(len(truth), dtype=np.float64) for k in (5, 10, 20)})
    result.update(Spearman=np.empty(len(truth), dtype=np.float64), Top20Enrichment=np.empty(len(truth), dtype=np.float64))
    for row in range(len(truth)):
        mask = np.isfinite(truth[row])
        y, p = truth[row, mask].astype(float), prediction[row, mask].astype(float)
        if len(y) < minimum:
            raise ValueError("Insufficient observed target drugs")
        true_order, pred_order = np.argsort(-y, kind="mergesort"), np.argsort(-p, kind="mergesort")
        for k in (5, 10, 20):
            use = min(k, len(y))
            overlap = len(set(true_order[:use]) & set(pred_order[:use]))
            result[f"Recall{k}"][row] = overlap/use
            if k == 20:
                result["Top20Enrichment"][row] = result["Recall20"][row]*len(y)/use
        if np.std(p) == 0 or np.std(y) == 0:
            result["Spearman"][row] = 0.
        else:
            rho = float(spearmanr(y, p).statistic)
            result["Spearman"][row] = rho if np.isfinite(rho) else 0.
    return result


@dataclass
class DrugRankModel:
    para_members: list
    baseline_representation: BaselineRepresentation
    random_forest: object
    srmf_original: SRMFState
    srmf_source: SRMFState
    transform: str
    fusion_weights: tuple
    gene_axis_sha256: str
    drug_axis_sha256: str
    metadata: dict

    def predict(self, expression, source_response, gene_names, drug_names, return_experts=False):
        if axis_digest(gene_names) != self.gene_axis_sha256 or axis_digest(drug_names) != self.drug_axis_sha256:
            raise ValueError("Ordered gene/drug identity axes differ from the checkpoint")
        if expression.ndim != 2 or source_response.ndim != 2 or len(expression) != len(source_response):
            raise ValueError("Prediction input shapes differ")
        if not np.isfinite(expression).all() or np.isinf(source_response).any():
            raise ValueError("Invalid prediction inputs")
        values = self.baseline_representation.transform(expression, source_response)
        experts = {"PARA": np.mean([m.predict(expression, source_response).astype(np.float64) for m in self.para_members], axis=0).astype(np.float32),
                   "RF_SourceAug": self.random_forest.predict(pair_features(values, self.baseline_representation.target_mean)).reshape(source_response.shape).astype(np.float32),
                   "SRMF_Original": self.srmf_original.predict(values), "SRMF_SourceAug": self.srmf_source.predict(values)}
        if any(not np.isfinite(x).all() for x in experts.values()):
            raise ValueError("Nonfinite expert prediction")
        prediction = blend_predictions(experts, self.transform, self.fusion_weights)
        return (prediction, experts) if return_experts else prediction


def fit_selected(data, train_rows, recipe, config):
    """Refit the exact inner-selected experts without consulting held-out targets."""
    train_rows = np.asarray(train_rows, dtype=bool)
    expression = data["expression"][train_rows]
    source = data["source_response"][train_rows]
    target = data["target_response"][train_rows]
    fold = int(recipe["outer_fold"])
    if np.any(data["outer_fold"][train_rows] == fold):
        raise ValueError("Outer held-out outcomes cannot enter fit")
    representations, members = {}, []
    for candidate in recipe["para_members"]:
        rank = int(candidate["rank"])
        if rank not in representations:
            seed = int(config["cohorts"][str(data["cohort"].item())]["para_partition_seed_base"])+fold+rank
            representations[rank] = ParaRepresentation.fit(expression, source, target, config["para_search"], rank, seed)
        representation, features, scores = representations[rank]
        members.append(ParaMember.fit(representation, features, scores, candidate))
    cohort = config["cohorts"][str(data["cohort"].item())]
    representation, values = BaselineRepresentation.fit(expression, source, target, config["baseline_representation"], cohort["baseline_partition_seed_base"]+fold)
    rf_config = recipe["baselines"]["RF_SourceAug"]
    rf_seed = stable_seed(config["baseline_seed_base"]+10000, fold, "outer", "SourceAug")
    rf = RandomForestRegressor(n_estimators=int(rf_config["n_estimators"]), max_depth=None if rf_config["max_depth"] == "None" else int(rf_config["max_depth"]), max_features=rf_config["max_features"], min_samples_leaf=int(rf_config["min_samples_leaf"]), bootstrap=True, n_jobs=int(rf_config["n_jobs"]), random_state=rf_seed)
    observed = np.isfinite(target.astype(np.float32)).reshape(-1)
    rf.fit(pair_features(values, representation.target_mean)[observed], target.astype(np.float32).reshape(-1)[observed])
    original = SRMFState.fit(values, target.astype(np.float32), representation.target_mean, "Original", recipe["baselines"]["SRMF_Original"], stable_seed(config["baseline_seed_base"]+100000, fold, "outer", "Original"))
    augmented = SRMFState.fit(values, target.astype(np.float32), representation.target_mean, "SourceAug", recipe["baselines"]["SRMF_SourceAug"], stable_seed(config["baseline_seed_base"]+110000, fold, "outer", "SourceAug"))
    return DrugRankModel(members, representation, rf, original, augmented, recipe["transform"], tuple(recipe["weights"]), axis_digest(data["gene_names"]), axis_digest(data["drug_names"]), {"cohort": str(data["cohort"].item()), "outer_fold": fold, "training_rows": int(train_rows.sum()), "recipe": recipe, "schema": "PARA_DRUGRANK_FITTED_STATE_V1", "contains_training_derived_reference_states": True, "contains_raw_input_matrices": False})


def save_model(model, path):
    path = Path(path)
    if path.exists() or path.with_suffix(path.suffix+".json").exists():
        raise FileExistsError("Refusing to overwrite an existing checkpoint or sidecar")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix+".json").write_text(json.dumps({**model.metadata, "checkpoint_sha256": digest, "bytes": path.stat().st_size}, indent=2)+"\n")
    return digest


def load_model(path):
    """Load only a checkpoint from a trusted source; joblib uses Python pickle."""
    model = joblib.load(path)
    if not isinstance(model, DrugRankModel) or model.metadata.get("schema") != "PARA_DRUGRANK_FITTED_STATE_V1":
        raise ValueError("Not a supported fitted-state checkpoint")
    representations = [m.representation for m in model.para_members]+[model.baseline_representation]
    for representation in representations:
        representation.expression_pca.restore_layout()
        representation.source_pca.restore_layout()
    return model
