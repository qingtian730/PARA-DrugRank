"""Small meaningful tests for state export, axis binding and target exclusion."""
import copy
import contextlib
import json
import io
import os
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from demo import synthetic
import train as training_cli
from para_drugrank.model import fit_selected, save_model, load_model, transform_prediction
from para_drugrank.data import axis_digest
from para_drugrank.training import select_recipe


class TrainingTests(unittest.TestCase):
    def test_user_supplied_recipe_catalogue(self):
        _, _, recipe = synthetic()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "catalogue.json"
            path.write_text(json.dumps({
                "schema": "PARA_DRUGRANK_SELECTED_RECIPES_V1",
                "recipes": {"PRISM": {"0": recipe}},
            }))
            self.assertEqual(training_cli.frozen_recipe(path, "PRISM", 0), recipe)
            with self.assertRaisesRegex(ValueError, "No frozen recipe"):
                training_cli.frozen_recipe(path, "GDSC2", 0)



    def test_entrypoint_help_works_outside_repository(self):
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        with tempfile.TemporaryDirectory() as folder:
            for name in ("train.py", "predict.py", "evaluate.py"):
                result = subprocess.run([sys.executable, str(ROOT / "source" / name), "--help"],
                    cwd=folder, env=environment, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--data", result.stdout)

    def test_train_default_reselection_and_legacy_recipe_interfaces(self):
        data, _, recipe = synthetic()
        with tempfile.TemporaryDirectory() as folder:
            single = Path(folder) / "single_recipe.json"
            single.write_text(json.dumps(recipe))
            catalogue = Path(folder) / "catalogue.json"
            catalogue.write_text(json.dumps({"schema": "PARA_DRUGRANK_SELECTED_RECIPES_V1", "recipes": {"PRISM": {"0": recipe}}}))
            for options, expected in (([], "nested_inner_selection"),
                    (["--recipe-file", str(catalogue)], "frozen_recipe_refit"),
                    (["--recipe", str(single)], "frozen_recipe_refit"),
                    (["--select-recipe"], "nested_inner_selection")):
                command = ["train.py", "--data", "supplied.npz", "--outer-fold", "0",
                           "--output", str(Path(folder) / "not_written.joblib"), "--panel", "custom", *options]
                output = io.StringIO()
                with patch.object(sys, "argv", command), \
                     patch.object(training_cli, "load_data", return_value=data), \
                     patch.object(training_cli, "select_recipe", return_value=recipe) as select, \
                     patch.object(training_cli, "fit_selected") as fit, \
                     patch.object(training_cli, "save_model", return_value="test_digest"), \
                     contextlib.redirect_stdout(output):
                    training_cli.main()
                self.assertEqual(json.loads(output.getvalue())["selection"], expected)
                self.assertEqual(select.call_count, int(expected == "nested_inner_selection"))
                self.assertEqual(fit.call_args.args[2], recipe)
                np.testing.assert_array_equal(fit.call_args.args[1], data["outer_fold"] != 0)
                self.assertFalse((Path(folder) / "not_written.joblib").exists())

    def test_serialization_keeps_every_expert_prediction(self):
        data, config, recipe = synthetic()
        model = fit_selected(data, data["outer_fold"] != 0, recipe, config)
        query = data["outer_fold"] == 0
        arguments = (data["expression"][query], data["source_response"][query], data["gene_names"], data["drug_names"])
        prediction, experts = model.predict(*arguments, return_experts=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"model.joblib"
            save_model(model, path)
            with self.assertRaises(FileExistsError):
                save_model(model, path)
            restored = load_model(path)
            after, after_experts = restored.predict(*arguments, return_experts=True)
            self.assertEqual(restored.para_members[0].representation.expression_pca.projector.components_.strides,
                             model.para_members[0].representation.expression_pca.projector.components_.strides)
        np.testing.assert_array_equal(prediction, after)
        for name in experts:
            np.testing.assert_array_equal(experts[name], after_experts[name])
        self.assertEqual(len(restored.random_forest.estimators_), 8)
        self.assertGreater(restored.para_members[0].kernel_model.dual_coef_.size, 0)
        self.assertGreater(restored.srmf_original.drug_factors.size, 0)

    def test_outer_targets_do_not_affect_fit_or_selection(self):
        data, config, recipe = synthetic()
        changed = copy.deepcopy(data)
        changed["target_response"][changed["outer_fold"] == 0] = 9999.
        selected = select_recipe(data, 0, config)
        selected_changed = select_recipe(changed, 0, config)
        self.assertEqual(selected, selected_changed)
        a = fit_selected(data, data["outer_fold"] != 0, recipe, config)
        b = fit_selected(changed, changed["outer_fold"] != 0, recipe, config)
        args = (data["expression"][:4], data["source_response"][:4], data["gene_names"], data["drug_names"])
        np.testing.assert_array_equal(a.predict(*args), b.predict(*args))

    def test_prediction_rejects_reordered_axes(self):
        data, config, recipe = synthetic()
        model = fit_selected(data, data["outer_fold"] != 0, recipe, config)
        with self.assertRaisesRegex(ValueError, "identity axes"):
            model.predict(data["expression"][:3], data["source_response"][:3], data["gene_names"][::-1], data["drug_names"])
        self.assertNotEqual(axis_digest(["a", "bc"]), axis_digest(["ab", "c"]))

    def test_rank_transform_uses_average_ties(self):
        actual = transform_prediction(np.asarray([[4., 2., 2., 1.]]), "row_percentile_rank")
        np.testing.assert_allclose(actual, [[1., .5, .5, 0.]])


if __name__ == "__main__":
    unittest.main()
