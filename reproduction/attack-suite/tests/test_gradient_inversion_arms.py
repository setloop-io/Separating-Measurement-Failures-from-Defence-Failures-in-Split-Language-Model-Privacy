"""gradient-inversion Phase C arms: iDLG analytic label recovery and the
GradInversion-style prior arm. Toy-level and torch-free (the new arms'
cores are pure python); the historical dlgpp arm keeps its torch guard and
dlgpp.py delegation, checked here for exactly that."""

import json

import pytest

from attacker.attacks import gradient_inversion as gi


def parse(*argv):
    args = gi.build_parser().parse_args(list(argv))
    args.mode = "training"
    return args


def test_parser_builds_torchless_with_new_flags():
    args = gi.build_parser().parse_args([])
    assert args.arm == "dlgpp"          # historical default, unchanged
    assert args.prior == "none"
    assert args.prior_lambda == 0.3
    assert args.smooth == 6.0
    assert args.evidence is False
    with pytest.raises(SystemExit):     # unknown arm rejected
        gi.build_parser().parse_args(["--arm", "nope"])
    for arm in ("dlgpp", "idlg", "prior-arm", "all"):
        assert gi.build_parser().parse_args(["--arm", arm]).arm == arm
    for prior in ("none", "tv", "group-lasso"):
        assert gi.build_parser().parse_args(["--prior", prior]).prior \
            == prior


def test_default_arm_behavior_unchanged():
    # without torch the historical arm keeps its guard message
    if gi.torch is None:
        with pytest.raises(SystemExit, match="requires torch"):
            gi.run(parse("--toy"))
    # delegation path still resolves split-training/dlgpp.py first
    with pytest.raises(SystemExit, match="dlgpp.py not found"):
        gi.run(parse("--dlgpp", "--model x", "--output", "/tmp/x.json"))


def test_new_arms_real_path_is_a_descriptive_integration_point():
    with pytest.raises(SystemExit, match="real-capture path"):
        gi.run(parse("--arm", "idlg"))
    with pytest.raises(SystemExit, match="real-capture path"):
        gi.run(parse("--arm", "prior-arm"))


def test_idlg_world_is_a_real_linear_model():
    world = gi.toy_classification_world(0, hidden=16, vocab=8, seq_len=4)
    # bias-gradient rows are the model's true CE gradients: db = p - y
    for i, row in enumerate(world["db"]):
        y = world["y"][i]
        assert row[y] == pytest.approx(
            -sum(v for c, v in enumerate(row) if c != y))  # rows sum to 0
        assert row[y] < 0.0 < max(v for c, v in enumerate(row) if c != y)
    # g* = db @ A^T consistency with the stored weight matrix
    import math
    a_t = [list(col) for col in zip(*world["weights_out"])]
    g_check = gi._matmul(world["db"], a_t)
    assert all(abs(a - b) < 1e-9
               for ra, rb in zip(g_check, world["g_star"])
               for a, b in zip(ra, rb))
    assert math.isfinite(world["h_star"][0][0])


def test_idlg_accuracy_bound_on_toy():
    # iDLG is exact on a per-row CE bias gradient (the sign argument); the
    # sane bound below leaves margin for float degeneracies only.
    args = parse("--toy", "--arm", "idlg", "--seq-len", "6", "--hidden",
                 "32", "--vocab", "16", "--rounds", "50", "--restarts", "1")
    for seed in (0, 1, 2):
        rec = gi.toy_idlg(args, seed)
        assert rec["label_accuracy"] >= 99.0
        assert rec["guided_token_recovery"] \
            >= rec["dlgpp_arm_token_recovery"]
        assert rec["improvement_pp"] == pytest.approx(
            rec["guided_token_recovery"] - rec["dlgpp_arm_token_recovery"])


def test_idlg_arm_runs_and_journals(tmp_path):
    out = str(tmp_path / "idlg.json")
    assert gi.run(parse("--toy", "--arm", "idlg", "--quick",
                        "--output", out)) == 0
    artifact = json.load(open(out))
    cells = artifact["results"]
    assert len(cells) == 2              # --quick seeds [0, 1]
    for rec in cells:
        assert rec["arm"] == "idlg"
        for key in ("label_accuracy", "guided_token_recovery",
                    "dlgpp_arm_token_recovery", "improvement_pp",
                    "guided_psnr_db", "dlgpp_arm_psnr_db",
                    "random_baseline"):
            assert key in rec
        assert rec["label_accuracy"] >= 99.0
    journal = [json.loads(l) for l in open(out + ".jsonl")]
    assert [j["seed"] for j in journal] == [0, 1]
    summary = artifact["summary"][0]
    assert summary["arm"] == "idlg"
    assert summary["label_accuracy_mean"] >= 99.0
    assert summary["improvement_pp_mean"] >= 0.0


def test_prior_arm_artifact_fields_and_measured_gain(tmp_path):
    out = str(tmp_path / "prior.json")
    assert gi.run(parse("--toy", "--arm", "prior-arm", "--quick",
                        "--output", out)) == 0
    artifact = json.load(open(out))
    cells = artifact["results"]
    assert len(cells) == 4              # 2 seeds x {none, group-lasso}
    for rec in cells:
        assert rec["arm"] == "prior-arm"
        for key in ("prior", "prior_lambda", "token_recovery", "rmse",
                    "psnr_db", "smooth"):
            assert key in rec
    assert sorted(r["prior"] for r in cells) == \
        ["group-lasso", "group-lasso", "none", "none"]
    # the smooth toy world makes the smoothness prior a TRUE prior: the
    # group-lasso PSNR gain over prior=none is measured, and positive
    gains = [s["psnr_gain_db"] for s in artifact["summary"]]
    assert len(gains) == 2
    assert all(g >= 0.0 for g in gains)
    assert sum(gains) / len(gains) > 0.0
    journal = [json.loads(l) for l in open(out + ".jsonl")]
    assert len(journal) == 4


def test_arm_all_runs_both_arms(tmp_path):
    out = str(tmp_path / "all.json")
    assert gi.run(parse("--toy", "--arm", "all", "--quick",
                        "--output", out)) == 0
    artifact = json.load(open(out))
    assert sorted(set(r["arm"] for r in artifact["results"])) == \
        ["idlg", "prior-arm"]


def test_prior_math_fixtures():
    z = [[0.0, 1.0], [0.4, 0.6], [0.8, 0.2]]
    import math
    assert gi._prior_value(z, "group-lasso") == pytest.approx(
        2 * math.sqrt(0.32))
    assert gi._prior_value(z, "tv") == pytest.approx(1.6)
    assert gi._prior_value(z, "none") == 0.0
    rmse, psnr = gi._rmse_psnr(z, z)
    assert rmse == 0.0 and psnr == 99.0  # capped, json-safe


def test_evidence_flag_wires_evidence_logger(tmp_path):
    from attacker.evidence import verify_evidence
    out = str(tmp_path / "ev.json")
    assert gi.run(parse("--toy", "--arm", "idlg", "--quick", "--evidence",
                        "--output", out)) == 0
    ok, problems = verify_evidence(out + ".evidence.json")
    assert ok and problems == []
    # --evidence without --output is a descriptive error
    with pytest.raises(SystemExit, match="--evidence requires --output"):
        gi.run(parse("--toy", "--arm", "idlg", "--quick", "--evidence"))


def test_module_self_test_passes():
    assert gi.self_test() == 0
