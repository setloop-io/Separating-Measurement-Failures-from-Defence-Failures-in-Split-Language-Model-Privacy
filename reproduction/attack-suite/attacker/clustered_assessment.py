"""Opt-in retrospective paired-cluster assessment; legacy gates are untouched.

This module is not registered in the application's CLI or default scorecard.
NumPy/SciPy are loaded only when the optional numerical routines are called.
All statements concern the supplied contract, recorded attacks and explicitly
declared independent-cluster assumptions. No privacy certification is implied.
"""
from dataclasses import asdict, dataclass
import math


POLICY_VERSION = "paired-cluster-assessment.v1-retrospective"
REQUIRED_CONTRACT_FIELDS = ("model", "corpus", "mode", "surface", "grant", "budget", "metric")


@dataclass(frozen=True)
class ClusterPolicy:
    version: str = POLICY_VERSION
    privacy_margin_pp: float = 1.0
    reference_margin_pp: float = 1.0
    family_alpha: float = 0.05
    planned_cases: int = 40
    components_per_case: int = 2  # privacy and reference competence
    minimum_seeds: int = 3
    utility_max_ppl_ratio: float = 1.25
    allow_serial_acceptance: bool = False

    def __post_init__(self):
        if not self.version or not 0 < self.family_alpha < 1:
            raise ValueError("a named policy and alpha in (0,1) are required")
        if not all(math.isfinite(x) for x in (self.privacy_margin_pp,
                self.reference_margin_pp, self.utility_max_ppl_ratio)):
            raise ValueError("finite policy thresholds are required")
        if (self.privacy_margin_pp <= 0 or self.reference_margin_pp < 0
                or self.planned_cases < 1 or self.components_per_case < 1
                or self.minimum_seeds < 2 or self.utility_max_ppl_ratio < 1):
            raise ValueError("invalid policy thresholds or sampling requirements")

    @property
    def test_alpha(self):
        return self.family_alpha / (self.planned_cases * self.components_per_case)


def _libraries():
    import numpy as np
    from scipy import stats
    return np, stats


def cluster_records(records, *, serial=False, block_length=10):
    """Aggregate paired counts by shared document identity, or ordered blocks.

    Document clusters collect every occurrence across fixed victim seeds.
    Serial records are grouped into non-overlapping, seed-specific blocks;
    their ordering is retained. Such blocks do not assert stationarity and
    cannot support an acceptance verdict under the default policy.
    """
    np, _ = _libraries()
    if not records or block_length < 1:
        raise ValueError("nonempty records and positive block length required")
    seen, grouped, seeds = set(), {}, set()
    for r in records:
        for name in ("seed", "document_id", "n_tokens", "correct", "control_correct"):
            if name not in r:
                raise ValueError(f"missing paired record field: {name}")
        seed = str(r["seed"])
        n, correct, control = (float(r[x]) for x in ("n_tokens", "correct", "control_correct"))
        if not all(math.isfinite(x) for x in (n, correct, control)):
            raise ValueError("nonfinite paired counts")
        if n <= 0 or min(correct, control) < 0 or max(correct, control) > n:
            raise ValueError("paired counts outside their token denominator")
        identity = (seed, str(r["document_id"]))
        if identity in seen:
            raise ValueError("duplicate seed/document record; average attack seeds first")
        seen.add(identity); seeds.add(seed)
        if serial:
            if "step" not in r or int(r["step"]) != r["step"] or r["step"] < 0:
                raise ValueError("serial records require a nonnegative integer step")
            key = f"seed={seed}:block={int(r['step']) // block_length}"
        else:
            key = str(r["document_id"])
        item = grouped.setdefault(key, [0.0, 0.0, 0.0, 0.0])
        item[0] += correct-control
        item[1] += n
        item[2] += correct
        item[3] += control
    labels = sorted(grouped)
    a = np.asarray([grouped[x] for x in labels], dtype=float)
    return {"difference_counts": a[:, 0], "tokens": a[:, 1],
            "correct": a[:, 2], "control_correct": a[:, 3],
            "labels": labels, "seeds": sorted(seeds),
            "records": len(records), "serial": serial,
            "block_length": block_length if serial else None,
            "scope": ("conditional ordered-block sensitivity; stationarity unestablished"
                      if serial else "document clusters conditional on the recorded victim seeds")}


def _bounded_p(distance, sum_squared_weights):
    """Hoeffding one-sided p-value bound for fixed weighted [-1,1] clusters."""
    return 1.0 if distance <= 0 else math.exp(-distance*distance / (2*sum_squared_weights))


def paired_cluster_test(cluster, *, margin_pp=1.0, alpha=0.05):
    """Upper-margin test and genuine two-one-sided equivalence sensitivity.

    Main calculation is a cluster-t/delta-method sensitivity for the paired
    ratio of counts. TOST tests H01: theta<=-margin and H02: theta>=margin,
    each at alpha; p_TOST is max(p_lower,p_upper). A lower-than-baseline
    attack may pass the upper-margin test and fail two-sided equivalence.

    Zero empirical variance never yields a zero-width certainty interval:
    a bounded-cluster Hoeffding calculation replaces the t calculation.
    Its validity still assumes independent supplied clusters and fixed weights.
    """
    np, stats = _libraries()
    x, n = (np.asarray(cluster[k], dtype=float) for k in ("difference_counts", "tokens"))
    if x.ndim != 1 or n.shape != x.shape or len(x) < 2 or not 0 < alpha < 0.5:
        raise ValueError("at least two paired clusters and alpha in (0,0.5) required")
    if margin_pp <= 0 or not np.isfinite(x).all() or not np.isfinite(n).all() or (n <= 0).any():
        raise ValueError("invalid margin or cluster arrays")
    if (np.abs(x) > n+1e-10).any():
        raise ValueError("paired cluster differences outside [-n,n]")
    margin = margin_pp/100.0
    mean = float(x.sum()/n.sum())
    influence = (x-mean*n)/n.mean()
    se = float(np.std(influence, ddof=1)/math.sqrt(len(x)))
    weights = n/n.sum()
    weight_squares = float(np.sum(weights**2))
    degenerate = se <= 1e-14
    if degenerate:
        radius = math.sqrt(2*weight_squares*math.log(1/alpha))
        lower, upper = max(-1.0, mean-radius), min(1.0, mean+radius)
        p_lower = _bounded_p(mean+margin, weight_squares)
        p_upper = _bounded_p(margin-mean, weight_squares)
        p_leakage = _bounded_p(mean-margin, weight_squares)
        method = "bounded-Hoeffding guard for zero empirical variance"
    else:
        critical = float(stats.t.ppf(1-alpha, len(x)-1))
        lower, upper = max(-1.0, mean-critical*se), min(1.0, mean+critical*se)
        p_lower = float(stats.t.sf((mean+margin)/se, len(x)-1))
        p_upper = float(stats.t.cdf((mean-margin)/se, len(x)-1))
        p_leakage = float(stats.t.sf((mean-margin)/se, len(x)-1))
        method = "paired cluster-t sensitivity with ratio linearisation"
    return {"mean_excess_pp": 100*mean, "standard_error_pp": 100*se,
            "lower_one_sided_pp": 100*lower, "upper_one_sided_pp": 100*upper,
            "central_interval_level": 1-2*alpha, "alpha_per_one_sided_test": alpha,
            "clusters": len(x), "effective_fixed_weight_clusters": 1/weight_squares,
            "degrees_of_freedom": len(x)-1, "method": method,
            "zero_variance_guard_used": degenerate,
            "upper_margin": {"null": "excess >= +margin", "alternative": "excess < +margin",
                "margin_pp": margin_pp, "p_value": p_upper,
                "supported_below_margin": bool(p_upper < alpha)},
            "leakage_above_margin": {"null": "excess <= +margin", "p_value": p_leakage,
                "supported_above_margin": bool(p_leakage < alpha)},
            "tost": {"lower_null": "excess <= -margin", "upper_null": "excess >= +margin",
                "margin_pp": margin_pp, "p_lower": p_lower, "p_upper": p_upper,
                "p_tost": max(p_lower, p_upper),
                "equivalent": bool(max(p_lower, p_upper) < alpha)},
            "assumptions": ["independent document clusters, or explicitly conditional ordered blocks",
                "ratio linearisation/t approximation for nonzero-variance clusters",
                "fixed recorded seeds, model, corpus and attacker contract",
                "no guarantee for untested attacks, new training corpora or arbitrary temporal drift"]}


def paired_bootstrap(cluster, *, replicates=100_000, seed=93281, alpha=0.05):
    """Paired identity/block bootstrap, a descriptive sensitivity only.

    Every shared document identity is resampled once, retaining all seed
    occurrences. Degenerate percentile outputs are explicitly not promoted
    to evidence of a zero-width population bound.
    """
    np, _ = _libraries()
    x, n = (np.asarray(cluster[k], dtype=float) for k in ("difference_counts", "tokens"))
    if replicates < 100 or not 0 < alpha < 0.5:
        raise ValueError("at least100 draws and valid one-sided alpha required")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for start in range(0, replicates, 1000):
        end = min(replicates, start+1000)
        idx = rng.integers(len(x), size=(end-start, len(x)))
        draws[start:end] = x[idx].sum(axis=1)/n[idx].sum(axis=1)
    lo, hi = np.quantile(draws, [alpha, 1-alpha])
    return {"mean_excess_pp": float(100*x.sum()/n.sum()),
            "lower_pp": float(100*lo), "upper_pp": float(100*hi),
            "central_interval_level": 1-2*alpha, "replicates": replicates, "seed": seed,
            "expected_draws_per_tail": replicates*alpha,
            "degenerate_percentile_distribution": bool(np.ptp(draws) <= 1e-14),
            "interpretation": "retrospective conditional percentile sensitivity; never overrides the zero-variance guard"}


def utility_eligibility(utility, *, max_ppl_ratio=1.25):
    """Fixed retrospective utility criterion using matched cross-entropies.

    This is an explicit empirical utility filter, not a confidence test or a
    universal tolerance. Training-loss and held-out-loss scopes stay labelled.
    """
    if not math.isfinite(max_ppl_ratio) or max_ppl_ratio < 1:
        raise ValueError("finite utility ratio at least one is required")
    if not utility or not utility.get("pairs") or not utility.get("metric"):
        return {"status": "INCONCLUSIVE", "reason": "missing matched task-utility arm"}
    pairs = utility["pairs"]
    try:
        seeds = [str(x["seed"]) for x in pairs]
        values = [float(x[k]) for x in pairs for k in ("candidate_ce", "reference_ce")]
    except (KeyError, TypeError, ValueError):
        return {"status": "INCONCLUSIVE", "reason": "malformed matched utility values"}
    if len(seeds) != len(set(seeds)):
        return {"status": "INCONCLUSIVE", "reason": "duplicate utility seed identities"}
    if not all(math.isfinite(x) and x >= 0 for x in values):
        return {"status": "INCONCLUSIVE", "reason": "nonfinite or negative utility values"}
    delta = sum(float(x["candidate_ce"])-float(x["reference_ce"]) for x in pairs)/len(pairs)
    ratio = math.exp(delta) if delta < 700 else None
    return {"status": "ELIGIBLE" if delta <= math.log(max_ppl_ratio) else "NON_VIABLE",
            "metric": utility["metric"], "matched_pairs": len(pairs),
            "seeds": sorted(seeds),
            "mean_ce_increase": delta, "geometric_ppl_ratio": ratio,
            "ppl_ratio_overflow": ratio is None,
            "maximum_ppl_ratio": max_ppl_ratio,
            "interpretation": "declared retrospective utility tolerance; empirical point criterion"}


def _paired_incidence(records, *, serial):
    """Match the actual victims/probes, not merely their protocol labels."""
    return {(str(r["seed"]), str(r["document_id"]),
             int(r["step"]) if serial else None):
            (float(r["n_tokens"]), float(r["control_correct"])) for r in records}


def assess(records, *, reference_records=None, protocol=None, reference_protocol=None,
           utility=None, policy=None, serial=False, block_length=10):
    """Explicit opt-in assessment; absent/weak controls cannot produce PASS."""
    policy = policy or ClusterPolicy()
    cl = cluster_records(records, serial=serial, block_length=block_length)
    candidate = paired_cluster_test(cl, margin_pp=policy.privacy_margin_pp, alpha=policy.test_alpha)
    reasons = []
    ref_test = None
    alignment = {"matched": False, "requires": "identical seed/document/token/control incidence"}
    if not reference_records:
        reasons.append("missing linked undefended reference")
    elif (not protocol or any(k not in protocol for k in REQUIRED_CONTRACT_FIELDS)
          or protocol != reference_protocol):
        reasons.append("reference protocol mismatch or absent contract")
    else:
        ref_cl = cluster_records(reference_records, serial=serial, block_length=block_length)
        ref_test = paired_cluster_test(ref_cl, margin_pp=policy.reference_margin_pp, alpha=policy.test_alpha)
        same_seeds = cl["seeds"] == ref_cl["seeds"]
        same_incidence = (_paired_incidence(records, serial=serial)
                          == _paired_incidence(reference_records, serial=serial))
        alignment.update(matched=same_seeds and same_incidence,
                         same_seed_identities=same_seeds,
                         same_probe_denominators_and_controls=same_incidence)
        if not same_seeds:
            reasons.append("candidate/reference victim seed identities do not match")
        if not same_incidence:
            reasons.append("candidate/reference seed-document-token-control incidence does not match")
        if len(ref_cl["seeds"]) < policy.minimum_seeds:
            reasons.append("reference lacks required independent seed records")
        if not ref_test["leakage_above_margin"]["supported_above_margin"]:
            reasons.append("reference lower bound does not establish competence above margin")
    if len(cl["seeds"]) < policy.minimum_seeds:
        reasons.append("candidate lacks required independent seed records")
    util = utility_eligibility(utility, max_ppl_ratio=policy.utility_max_ppl_ratio)
    if util["status"] == "INCONCLUSIVE":
        reasons.append(util["reason"])
    elif util["seeds"] != cl["seeds"]:
        reasons.append("utility arm is not matched to every recorded victim seed")
    if serial and not policy.allow_serial_acceptance:
        reasons.append("evolving training trajectories: block results are conditional sensitivity, not acceptance evidence")
    if reasons:
        verdict = "INCONCLUSIVE"
    elif util["status"] == "NON_VIABLE":
        verdict = "NON_VIABLE"
    elif candidate["upper_margin"]["supported_below_margin"]:
        verdict = "PASS_TESTED_CONTRACT"
    elif candidate["leakage_above_margin"]["supported_above_margin"]:
        verdict = "FAIL_ABOVE_MARGIN"
    else:
        verdict = "INCONCLUSIVE"
        reasons.append("paired uncertainty does not resolve the declared margin")
    return {"policy": asdict(policy), "verdict": verdict, "reasons": reasons,
            "candidate": candidate, "reference": ref_test, "utility": util,
            "reference_alignment": alignment,
            "cluster_scope": cl["scope"], "seeds": cl["seeds"],
            "records": cl["records"], "serial": serial,
            "block_length": cl["block_length"],
            "retrospective": True,
            "legacy_gate_and_scorecard_changed": False}


def simulation_power(profile, *, simulation, alpha, margin_pp=1.0):
    """Bounded paired-count planning simulation; no learned model is executed.

    Profile supplies the *actual* seed/document incidence and tokens per
    independent document or hypothetical stationary block. A baseline token
    vector is sampled once per document, then reused by every seed observing
    that document. Seed effects are shared across that seed's documents.
    This preserves those two sources of dependence, rather than multiplying
    a token count and pretending the resulting tokens are independent units.
    """
    np, stats = _libraries()
    incidence = np.asarray(profile["incidence"], dtype=bool)
    lengths = np.asarray(profile["tokens_per_document"], dtype=int)
    if incidence.ndim != 2 or incidence.shape[1] != len(lengths) or len(lengths) < 2:
        raise ValueError("invalid simulation incidence or document lengths")
    denominator = incidence.sum(axis=0)*lengths
    if (denominator <= 0).any():
        raise ValueError("simulation has an unobserved document")
    p0 = float(profile["baseline_rate"])
    if not 0 <= p0 <= 1:
        raise ValueError("invalid simulation baseline")
    reps = int(simulation["replicates_per_scenario"])
    if reps < 100:
        raise ValueError("planning simulation requires at least100 replicates")
    total_tokens = denominator.sum()
    weight_sq = float(np.sum((denominator/total_tokens)**2))
    g = len(lengths)
    critical = float(stats.t.ppf(1-alpha, g-1))
    margin = margin_pp/100
    rng = np.random.default_rng(simulation["seed"])
    rows = []
    for rho in simulation["document_beta_rho"]:
        for tau in simulation["seed_probability_sd"]:
            for coupling in simulation["paired_coupling"]:
                for requested_pp in simulation["requested_excess_pp"]:
                    accepted, equivalent, detected, competent = [], [], [], []
                    expected_values, measured_values, clips = [], [], []
                    for start in range(0, reps, 64):
                        b = min(64, reps-start)
                        if 0 < rho < 1 and 0 < p0 < 1:
                            kappa = 1/rho-1
                            q = rng.beta(p0*kappa, (1-p0)*kappa, size=(b, g))
                        else:
                            q = np.full((b, g), p0)
                        baseline = rng.binomial(lengths[None, :], q)
                        offsets = rng.normal(0, tau, size=(b, incidence.shape[0]))
                        attack_sum = np.zeros((b, g), dtype=float)
                        expected_sum = np.zeros((b, g), dtype=float)
                        clipped = np.zeros(b)
                        for s in range(incidence.shape[0]):
                            raw = q+requested_pp/100+offsets[:, s, None]
                            a = np.clip(raw, 0, 1)
                            clipped += ((raw != a)*incidence[s]).sum(axis=1)
                            # Mixture of maximal and independent paired coupling.
                            p11 = coupling*np.minimum(a, q)+(1-coupling)*a*q
                            given_true = np.divide(p11, q, out=np.zeros_like(a), where=q > 0)
                            given_false = np.divide(a-p11, 1-q, out=np.zeros_like(a), where=q < 1)
                            counts = (rng.binomial(baseline, np.clip(given_true, 0, 1))
                                      + rng.binomial(lengths[None, :]-baseline,
                                                     np.clip(given_false, 0, 1)))
                            attack_sum += counts*incidence[s]
                            expected_sum += (a-q)*lengths[None, :]*incidence[s]
                        x = attack_sum-baseline*incidence.sum(axis=0)
                        mean = x.sum(axis=1)/total_tokens
                        influence = (x-mean[:, None]*denominator[None, :])/denominator.mean()
                        se = np.std(influence, axis=1, ddof=1)/math.sqrt(g)
                        radius = critical*se
                        zero = se <= 1e-14
                        radius[zero] = math.sqrt(2*weight_sq*math.log(1/alpha))
                        lower, upper = mean-radius, mean+radius
                        accepted.extend((upper < margin).tolist())
                        equivalent.extend(((lower > -margin) & (upper < margin)).tolist())
                        detected.extend((lower > 0).tolist())
                        competent.extend((lower > margin).tolist())
                        measured_values.extend(mean.tolist())
                        expected_values.extend((expected_sum.sum(axis=1)/total_tokens).tolist())
                        clips.extend((clipped/incidence.sum()).tolist())
                    row = {"profile": profile["id"], "rho": rho,
                           "seed_probability_sd": tau, "paired_coupling": coupling,
                           "requested_excess_pp": requested_pp,
                           "achieved_expected_excess_pp": 100*float(np.mean(expected_values)),
                           "mean_sample_excess_pp": 100*float(np.mean(measured_values)),
                           "boundary_clipping_fraction": float(np.mean(clips)),
                           "replicates": reps, "clusters": g,
                           "alpha": alpha, "privacy_margin_pp": margin_pp}
                    for name, observations in (("upper_margin_acceptance", accepted),
                            ("tost_equivalence", equivalent), ("detect_excess_above_zero", detected),
                            ("reference_competence_above_margin", competent)):
                        prob = sum(observations)/reps
                        row[name+"_probability"] = prob
                        row[name+"_monte_carlo_se"] = math.sqrt(prob*(1-prob)/reps)
                    rows.append(row)
    return rows
