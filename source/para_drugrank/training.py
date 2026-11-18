"""Four-fold inner selection followed by an outer-training-only fitted model."""
from __future__ import annotations
from itertools import product
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor
from .model import (EXPERTS, ParaRepresentation, ParaMember, BaselineRepresentation,
                    SRMFState, pair_features, stable_seed, blend_predictions)


def ndcg_rows(truth, prediction, minimum=1, k=50):
    if truth.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("Metric shapes or prediction finiteness invalid")
    values = []
    for target, estimate in zip(truth, prediction):
        mask = np.isfinite(target)
        y, p = target[mask].astype(np.float64), estimate[mask].astype(np.float64)
        if len(y) < minimum:
            raise ValueError("Insufficient observed drugs for frozen metric")
        relevance = (rankdata(y, method="average")-1.)/max(len(y)-1, 1)
        use = min(k, len(y))
        gains = np.expm1(np.log(2.)*relevance)
        discount = 1./np.log2(np.arange(2, use+2, dtype=np.float64))
        order, ideal = np.argsort(-p, kind="mergesort")[:use], np.argsort(-relevance, kind="mergesort")[:use]
        dcg, idcg = float(np.sum(gains[order]*discount)), float(np.sum(gains[ideal]*discount))
        values.append(dcg/idcg if idcg > 0 else 0.)
    return np.asarray(values)


def candidate_key(value):
    return "|".join(f"{key}={value[key]}" for key in sorted(value))


def para_key(c):
    return f"rank={c['rank']}|alpha={c['alpha']:g}|gamma={c['gamma_scale']:g}|gate={c['gate']:g}"


def para_tie(c):
    return c["rank"], abs(c["gate"]-1.), -c["alpha"], c["gamma_scale"]


def select_stack(predictions, truth, stacking, minimum=1):
    weights = []
    total = int(stacking["integer_weight_total"])
    for a in range(int(stacking["minimum_para_integer_weight"]), total+1):
        for b in range(total-a+1):
            for c in range(total-a-b+1):
                weights.append(tuple(float(x)*float(stacking["weight_step"]) for x in (a, b, c, total-a-b-c)))
    if len(weights) != stacking["simplex_configuration_count_per_transform"]:
        raise ValueError("Frozen simplex cardinality mismatch")
    def weight_tie(w):
        return (-w[0], sum(x > 1e-12 for x in w), tuple(round(x, 12) for x in w))
    candidates = []
    transforms = stacking["prediction_transforms"]
    for transform in transforms:
        singles = [{"transform": transform, "mode": "single", "weights": w,
                    "score": float(ndcg_rows(truth, blend_predictions(predictions, transform, w), minimum).mean())} for w in weights]
        ordered = sorted(singles, key=lambda x: (-x["score"], weight_tie(x["weights"])))
        candidates.extend(singles)
        for top in stacking["stability_ensembles_per_transform"]:
            w = tuple(float(np.mean([entry["weights"][j] for entry in ordered[:top]])) for j in range(4))
            candidates.append({"transform": transform, "mode": f"top{top}_weight_average", "weights": w,
                               "score": float(ndcg_rows(truth, blend_predictions(predictions, transform, w), minimum).mean())})
    best = max(x["score"] for x in candidates)
    eligible = [x for x in candidates if x["score"] >= best-float(stacking["selection_tolerance_NDCG50"])]
    preference = {"top5_weight_average": 0, "top3_weight_average": 1, "single": 2}
    return sorted(eligible, key=lambda x: (preference[x["mode"]], weight_tie(x["weights"]), transforms.index(x["transform"]), -x["score"]))[0]


def select_recipe(data, fold, config, progress=None):
    """Outer target values are never used in selection, fitting, or transforms."""
    outer = data["outer_fold"] != int(fold)
    rows = np.flatnonzero(outer)
    assignments = data["inner_fold"][rows, int(fold)]
    cohort = str(data["cohort"].item())
    cfg = config["cohorts"][cohort]
    search = config["para_search"]
    candidates = [dict(zip(("rank", "alpha", "gamma_scale", "gate"), values)) for values in product(search["residual_ranks"], search["alphas"], search["gamma_scales"], search["residual_gates"])]
    shape = len(rows), data["target_response"].shape[1]
    para_oof = {para_key(c): np.full(shape, np.nan, dtype=np.float32) for c in candidates}
    baseline_oof, baseline_scores = {}, {}
    for name, family in (("RF_SourceAug", "RF"), ("SRMF_Original", "SRMF"), ("SRMF_SourceAug", "SRMF")):
        baseline_oof[name] = {candidate_key(c): np.full(shape, np.nan, dtype=np.float32) for c in config["baseline_grids"][family]}
        baseline_scores[name] = {key: [] for key in baseline_oof[name]}
    for inner in range(4):
        local = assignments == inner
        query = rows[local]
        train = outer.copy()
        train[query] = False
        e, s, y = data["expression"][train], data["source_response"][train], data["target_response"][train]
        qe, qs = data["expression"][query], data["source_response"][query]
        if progress:
            progress(f"outer={fold} inner={inner}: PARA and baseline candidates")
        for rank in search["residual_ranks"]:
            seed = cfg["para_partition_seed_base"]+fold*100+inner*10+rank
            representation, features, scores = ParaRepresentation.fit(e, s, y, search, rank, seed)
            for alpha, gamma in product(search["alphas"], search["gamma_scales"]):
                candidate = {"rank": rank, "alpha": alpha, "gamma_scale": gamma, "gate": 1.}
                member = ParaMember.fit(representation, features, scores, candidate)
                factors = member.kernel_model.predict(representation.transform(qe, qs)).astype(np.float64)*member.factor_scale+member.factor_mean
                residual = representation.center+factors @ representation.basis
                for gate in search["residual_gates"]:
                    c = {**candidate, "gate": gate}
                    para_oof[para_key(c)][local] = (representation.target_mean.astype(np.float64)+gate*residual).astype(np.float32)
        seed = cfg["baseline_partition_seed_base"]+fold*100+inner
        representation, values = BaselineRepresentation.fit(e, s, y, config["baseline_representation"], seed)
        query_values = representation.transform(qe, qs)
        for name, family, variant, family_index, variant_index in (("RF_SourceAug", "RF", "SourceAug", 0, 1), ("SRMF_Original", "SRMF", "Original", 1, 0), ("SRMF_SourceAug", "SRMF", "SourceAug", 1, 1)):
            for index, candidate in enumerate(config["baseline_grids"][family]):
                fit_seed = stable_seed(config["baseline_seed_base"]+family_index*100000+variant_index*10000, fold, inner, index, variant)
                if family == "RF":
                    rf = RandomForestRegressor(n_estimators=int(candidate["n_estimators"]), max_depth=None if candidate["max_depth"] == "None" else int(candidate["max_depth"]), max_features=candidate["max_features"], min_samples_leaf=int(candidate["min_samples_leaf"]), n_jobs=int(candidate["n_jobs"]), bootstrap=True, random_state=fit_seed)
                    observed = np.isfinite(y.astype(np.float32)).reshape(-1)
                    rf.fit(pair_features(values, representation.target_mean, variant)[observed], y.astype(np.float32).reshape(-1)[observed])
                    pred = rf.predict(pair_features(query_values, representation.target_mean, variant)).reshape((len(query), shape[1])).astype(np.float32)
                else:
                    state = SRMFState.fit(values, y.astype(np.float32), representation.target_mean, variant, candidate, fit_seed)
                    pred = state.predict(query_values)
                key = candidate_key(candidate)
                baseline_oof[name][key][local] = pred
                baseline_scores[name][key].append(float(ndcg_rows(data["target_response"][query], pred, cfg["minimum_finite_drugs"]).mean()))
    truth = data["target_response"][rows]
    para_scores = {key: float(ndcg_rows(truth, pred, cfg["minimum_finite_drugs"]).mean()) for key, pred in para_oof.items()}
    ordered = sorted(candidates, key=lambda c: (-para_scores[para_key(c)], para_tie(c)))
    selections = [("single", ordered[:1], para_scores[para_key(ordered[0])])]
    for top in search["top_k_equal_ensembles"]:
        members = ordered[:top]
        pred = np.mean([para_oof[para_key(c)].astype(np.float64) for c in members], axis=0).astype(np.float32)
        selections.append((f"top{top}_equal", members, float(ndcg_rows(truth, pred, cfg["minimum_finite_drugs"]).mean())))
    best = max(x[2] for x in selections)
    selected = sorted([x for x in selections if x[2] >= best-search["selection_tolerance"]], key=lambda x: ({"top5_equal": 0, "top3_equal": 1, "single": 2}[x[0]], -x[2]))[0]
    inner_predictions = {"PARA": np.mean([para_oof[para_key(c)].astype(np.float64) for c in selected[1]], axis=0).astype(np.float32)}
    baselines = {}
    for name, family in (("RF_SourceAug", "RF"), ("SRMF_Original", "SRMF"), ("SRMF_SourceAug", "SRMF")):
        means = {key: float(np.mean(value)) for key, value in baseline_scores[name].items()}
        key = sorted(means, key=lambda k: (-means[k], k))[0]
        baselines[name] = next(c for c in config["baseline_grids"][family] if candidate_key(c) == key)
        inner_predictions[name] = baseline_oof[name][key]
    stack = select_stack(inner_predictions, truth, config["stacking"], cfg["minimum_finite_drugs"])
    return {"cohort": cohort, "outer_fold": int(fold), "para_members": selected[1], "baselines": baselines,
            "transform": stack["transform"], "weights": list(stack["weights"]), "selection_mode": stack["mode"],
            "selected_inner_NDCG50": stack["score"], "para_selection_mode": selected[0],
            "data_release": "USER_SUPPLIED_PREPARED_DATA", "outer_targets_used_for_selection": False}
