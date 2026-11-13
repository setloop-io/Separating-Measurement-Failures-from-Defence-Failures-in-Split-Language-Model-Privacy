"""Stdlib-only contract fixtures. No subprocess, package install or ML execution."""

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup = load_module("original_at_setup_fixture", "setup.py")
saved = load_module("original_at_saved_fixture", "rescore_saved.py")
headers = load_module("original_at_headers_fixture", "private_headers.py")


def config():
    return json.loads((HERE / "run.json").read_text())


def utility():
    call = {"references": ["def f(x): return x"], "predictions": ["def f(x): return x"],
            "lang": "python", "weights": [0.25, 0.25, 0.25, 0.25], "result": {"codebleu": 0.5}}
    return [{"dataset": "CodeAlpaca", "max_new_tokens": 200, "test_rows": 1,
             "predicted_ids": [[1, 2]], "target_ids": [[1, 2]], "scoring_calls": [copy.deepcopy(call) for _ in range(10)],
             "native_result": {"acc": 0.5}}]


class StaticContracts(unittest.TestCase):
    def test_selfcontained_source_and_model_allowlist(self):
        result = setup.validate(config())
        self.assertEqual(result["source_files_verified"], 254)
        self.assertEqual(result["config_git_blobs_verified"], 2)
        self.assertEqual(result["model_files"], 12)
        self.assertEqual(result["model_bytes"], 13479516728)
        self.assertEqual(result["model_bin_files"], 0)

    def test_mid_environment_rejected(self):
        value = config()
        value["host"]["private_environment"] = value["host"]["forbidden_environment_paths"][0]
        with self.assertRaisesRegex(ValueError, "MID"):
            setup.validate(value)

    def test_changed_metric_pin_rejected(self):
        value = config()
        value["pins"]["tree-sitter"] = "0.26.0"
        with self.assertRaisesRegex(ValueError, "pin"):
            setup.validate(value)

    def test_execution_stage_not_allowed(self):
        value = config()
        value["stage"] = "native-model-full"
        with self.assertRaisesRegex(ValueError, "schema/stage"):
            setup.validate(value)

    def test_unlabelled_metric_replacement_rejected(self):
        value = config()
        value["native_metric_policy"] = "replace-with-whitespace"
        with self.assertRaisesRegex(ValueError, "silently replace"):
            setup.validate(value)

    def test_asset_byte_budget_changed_rejected(self):
        value = config()
        value["guards"]["maximum_model_asset_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "byte budget"):
            setup.validate(value)

    def test_torch_install_plan_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plan.json"
            path.write_text(json.dumps({"install": [{"metadata": {"name": "torch"}}]}))
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                setup.reject_torch_plan(path)

    def test_metric_only_plan_allowed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plan.json"
            path.write_text(json.dumps({"install": [{"metadata": {"name": "CodeBLEU"}}]}))
            setup.reject_torch_plan(path)

    def test_saved_generation_contract_preserves_exact_text(self):
        documents = saved.extract_saved_texts(utility())
        self.assertEqual(documents[0]["reference"], "def f(x): return x")
        self.assertEqual(documents[0]["native_codebleu"], 0.5)

    def test_saved_calls_with_different_text_rejected(self):
        value = utility()
        value[0]["scoring_calls"][3]["predictions"] = ["different"]
        with self.assertRaisesRegex(ValueError, "do not agree"):
            saved.extract_saved_texts(value)

    def test_missing_generation_row_rejected(self):
        value = utility()
        value[0]["predicted_ids"] = []
        with self.assertRaisesRegex(ValueError, "coverage"):
            saved.extract_saved_texts(value)

    def test_missing_native_call_rejected(self):
        value = utility()
        value[0]["scoring_calls"].pop()
        with self.assertRaisesRegex(ValueError, "ten exact"):
            saved.extract_saved_texts(value)

    def test_native_aggregate_mismatch_rejected(self):
        value = utility()
        value[0]["native_result"]["acc"] = 0.6
        with self.assertRaisesRegex(ValueError, "aggregation differs"):
            saved.extract_saved_texts(value)

    def test_earlier_admission_required_before_assets(self):
        with mock.patch.object(setup, "check_receipt", side_effect=RuntimeError("missing dependency gate")):
            with mock.patch.object(setup, "run_logged") as launch:
                with self.assertRaisesRegex(RuntimeError, "missing dependency gate"):
                    setup.assets(config(), {}, Path("/must-not-create"), {})
                launch.assert_not_called()

    def test_only_exact_unadmitted_owner_repair_allowed(self):
        value = config()
        old = {"contract_sha256": setup.REPAIR_OWNER}
        record = setup.recoverable_owner(value, old, {"contract_sha256": "new"})
        self.assertEqual(record["failed_orx_run_id"], setup.REPAIR_RUN)
        self.assertFalse(record["successful_prior_admission_present"])
        self.assertIsNone(setup.recoverable_owner(value, {"contract_sha256": "new"}, {"contract_sha256": "new"}))
        with self.assertRaisesRegex(RuntimeError, "another source/config"):
            setup.recoverable_owner(value, {"contract_sha256": "unrecognised"}, {"contract_sha256": "new"})

    def test_repair_cannot_expand_to_other_failed_runs(self):
        value = config()
        value["provisional_software_repair"]["failed_orx_run_id"] = "other"
        with self.assertRaisesRegex(ValueError, "exact failed"):
            setup.validate(value)

    def test_header_download_hash_failure_is_atomic(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "headers.deb"
            spec = {"url": "https://ports.ubuntu.com/ubuntu-ports/pool/main/p/python3.12/fixture.deb",
                    "bytes": 4, "sha256": hashlib.sha256(b"good").hexdigest()}
            with self.assertRaisesRegex(ValueError, "SHA256"):
                headers.checked_download(spec, path, opener=lambda *a, **kw: io.BytesIO(b"bad!"))
            self.assertFalse(path.exists())
            self.assertFalse(path.with_name("headers.deb.part").exists())

    def test_cached_headers_still_rehashed_without_download(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "headers.deb"
            path.write_bytes(b"good")
            spec = {"url": "https://ports.ubuntu.com/ubuntu-ports/pool/main/p/python3.12/fixture.deb",
                    "bytes": 4, "sha256": hashlib.sha256(b"good").hexdigest()}
            opener = mock.Mock(side_effect=AssertionError("network not expected"))
            self.assertFalse(headers.checked_download(spec, path, opener))
            opener.assert_not_called()

    def test_nonofficial_header_source_rejected(self):
        spec = {"url": "https://example.com/headers.deb", "bytes": 4, "sha256": "unused"}
        with self.assertRaisesRegex(ValueError, "official Ubuntu"):
            headers.checked_download(spec, Path("must-not-create"))

    def test_header_paths_include_multiarch_dispatch_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "Python.h").write_bytes(b"fixture")
            spec = {"required_headers": {"Python.h": {"bytes": 7, "sha256": hashlib.sha256(b"fixture").hexdigest()}},
                    "include_directories": ["usr/include/python3.12", "usr/include/aarch64-linux-gnu/python3.12", "usr/include"]}
            env, _ = headers.header_environment(spec, root, {"CPATH": "/unreviewed", "C_INCLUDE_PATH": "/unreviewed"})
            self.assertTrue(env["CPATH"].endswith("/usr/include"))
            self.assertNotIn("/unreviewed", env["CPATH"])
            self.assertNotIn("C_INCLUDE_PATH", env)
            (root / "Python.h").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                headers.header_environment(spec, root, {})

    def test_wrong_python_package_fails_before_download_or_extraction(self):
        with mock.patch.object(headers, "checked_download") as download:
            with self.assertRaisesRegex(RuntimeError, "installed Python"):
                headers.prepare({"version": "expected"}, Path("must-not-create"), Path("unused"), {},
                                mock.Mock(), lambda command: "different")
            download.assert_not_called()

    def test_private_compat_wheels_are_exact_and_do_not_replace_cuda(self):
        proof = json.loads((HERE / "inputs/private-compat-provenance.json").read_text())
        self.assertEqual({item["name"] for item in proof["packages"]}, {"cffi", "pycparser", "nvidia-ml-py"})
        requirements = (HERE / "requirements-private-compat.txt").read_text()
        for item in proof["packages"]:
            self.assertEqual(len(item["wheels"]), 1)
            wheel = item["wheels"][0]
            self.assertIn(wheel["url"] + "#sha256=" + wheel["sha256"], requirements)


if __name__ == "__main__":
    unittest.main()
