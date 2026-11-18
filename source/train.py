#!/usr/bin/env python3
"""Select a model on four inner folds and refit it on the outer training set."""
import argparse
import json
import os
from pathlib import Path
import sys
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from para_drugrank.data import load_data
from para_drugrank.model import fit_selected, save_model
from para_drugrank.training import select_recipe
from para_drugrank.panels import select_panel


def frozen_recipe(path, cohort, outer_fold):
    """Load one cohort/fold recipe from the compact frozen catalogue."""
    catalogue = json.loads(Path(path).read_text())
    if catalogue.get("schema") != "PARA_DRUGRANK_SELECTED_RECIPES_V1":
        raise ValueError("Unsupported selected-recipe catalogue")
    try:
        return catalogue["recipes"][cohort][str(outer_fold)]
    except KeyError as error:
        raise ValueError(f"No frozen recipe for {cohort}, outer fold {outer_fold}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs/model.json"))
    parser.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--panel", choices=["common", "full", "custom"], default="common")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--recipe", type=Path, help="Refit a separately supplied single-recipe JSON")
    selection.add_argument("--recipe-file", type=Path, help="Refit from a separately supplied cohort/fold recipe catalogue")
    selection.add_argument("--select-recipe", action="store_true", help="Select using the four inner folds (default)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data, config = select_panel(load_data(args.data), args.panel), json.loads(Path(args.config).read_text())
    cohort = str(data["cohort"].item())
    if args.recipe:
        recipe = json.loads(args.recipe.read_text())
    elif args.recipe_file:
        recipe = frozen_recipe(args.recipe_file, cohort, args.outer_fold)
    else:
        recipe = select_recipe(data, args.outer_fold, config, progress=lambda s: print(s, flush=True))
    if recipe["outer_fold"] != args.outer_fold or recipe["cohort"] != str(data["cohort"].item()):
        parser.error("Recipe cohort/fold differs from data/command")
    model = fit_selected(data, data["outer_fold"] != args.outer_fold, recipe, config)
    digest = save_model(model, args.output)
    selection_mode = "frozen_recipe_refit" if args.recipe or args.recipe_file else "nested_inner_selection"
    print(json.dumps({"status": "SAVED_FITTED_MODEL", "sha256": digest, "selection": selection_mode}))


if __name__ == "__main__":
    main()
