#!/usr/bin/env python3
"""Build the domain-shifted seam-inversion corpora (domain-shift-v1).

Rebuttal commitment: a code corpus and a synthetic clinical-style corpus,
each reported with its own label-free majority control, against the
WikiText-2 pilot reference. Stdlib-only and seeded; writes:

  results/domain-shift-v1/inputs/code-corpus-v1.jsonl
  results/domain-shift-v1/inputs/clinical-synth-corpus-v1.jsonl
  experiments/domain_corpora_manifest.json   (SHA-256 pins, schema style
      mirrors experiments/validation_corpus_manifest.json)

The WikiText-2 reference corpus is NOT rebuilt: the run reuses the pinned
pilot inputs at results/pilot-gpt2-wikitext2/inputs/pilot-corpus.jsonl in
place (the same basis as the paper's campaign; the manifest cross-pins its
SHA-256). results/validation-corpus-v1/ is a frozen held-out input and is
never read here; all three corpora are disjoint from it by construction.

Document floors: every emitted document is >= MIN_CHARS characters, a
conservative proxy for the protocol's >= 32-GPT-2-token floor (the driver
attacker.experiment hard-rejects any shorter document at encode time; the
token-level check is repeated with the HF tokenizer after generation —
transformers is intentionally not imported here, keeping this script
stdlib-only).
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

SCHEMA = "llm_attacker.domain_corpora.v1"
SALT = "domain-shift-v1-20260916"
DOCS_PER_CORPUS = 96          # matches the pilot's 32/32/32 partitions
MIN_CHARS = 800               # >> 32 GPT-2 tokens, even for dense code
CODE_TARGET_CHARS = 2400
CODE_SOURCE_GLOB = "attacker/*.py"
WIKITEXT_REFERENCE = Path(
    "results/pilot-gpt2-wikitext2/inputs/pilot-corpus.jsonl")
WIKITEXT_REFERENCE_MANIFEST = Path(
    "results/pilot-gpt2-wikitext2/inputs/corpus-manifest.json")


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Code corpus: this repository's own MIT-licensed Python source, chunked on
# line boundaries into document-sized pieces.

def code_chunks(root, target=CODE_TARGET_CHARS, min_chars=MIN_CHARS):
    chunks = []
    for path in sorted(Path(root).glob(CODE_SOURCE_GLOB)):
        lines = path.read_text().splitlines(keepends=True)
        buf, idx = "", 0
        for line in lines:
            buf += line
            if len(buf) >= target:
                chunks.append((f"code-{path.stem}-{idx:04d}", buf.strip()))
                idx += 1
                buf = ""
        if buf.strip():
            if chunks and len(buf.strip()) < min_chars:
                prev_id, prev = chunks.pop()
                chunks.append((prev_id, (prev + "\n" + buf).strip()))
            else:
                chunks.append((f"code-{path.stem}-{idx:04d}", buf.strip()))
    # exact-text dedupe (repeated boilerplate across files), keep first
    seen, out = set(), []
    for cid, text in chunks:
        key = sha256_text(text)
        if key not in seen:
            seen.add(key)
            out.append((cid, text))
    return [c for c in out if len(c[1]) >= min_chars]


# ---------------------------------------------------------------------------
# Synthetic clinical corpus: templated, slot-filled notes. Clearly
# synthetic (banner in every note, fabricated patient ids, fixed small
# vocabularies). No real PHI exists here by construction.

SEXES = ["male", "female"]
COMPLAINTS = ["chest pain", "shortness of breath", "persistent cough",
              "lower back pain", "intermittent dizziness", "abdominal pain",
              "knee swelling", "morning headaches", "fatigue", "palpitations"]
CONDITIONS = ["type 2 diabetes mellitus", "essential hypertension",
              "hyperlipidemia", "chronic kidney disease stage 2",
              "mild persistent asthma", "osteoarthritis",
              "gastroesophageal reflux disease", "hypothyroidism",
              "atrial fibrillation", "iron deficiency anemia"]
MEDS = ["metformin", "lisinopril", "atorvastatin", "amlodipine",
        "levothyroxine", "omeprazole", "albuterol", "apixaban",
        "ferrous sulfate", "hydrochlorothiazide"]
FREQS = ["once daily", "twice daily", "every morning", "at bedtime",
         "every 12 hours"]
DEPTS = ["internal medicine", "family practice", "cardiology",
         "pulmonology", "endocrinology", "urgent care"]
ASSESSMENTS = [
    "symptoms most consistent with poorly controlled {cond}",
    "presentation suggests an exacerbation of {cond}",
    "findings are compatible with stable {cond} and deconditioning",
    "clinical picture is most consistent with new-onset {cond}",
    "symptoms remain unexplained; {cond} is well controlled at present"]
STUDIES = ["a basic metabolic panel", "a complete blood count",
           "a lipid panel", "a chest radiograph", "an ECG",
           "a hemoglobin A1c", "a urinalysis", "a thyroid panel"]
DISPOSITIONS = ["discharged home in stable condition",
                "observed for four hours and released",
                "referred to outpatient follow-up",
                "advised to return if symptoms worsen"]
FOLLOWUPS = ["one week", "two weeks", "one month", "six weeks"]

TEMPLATE = (
    "SYNTHETIC RECORD - fabricated for a model-inversion benchmark; this "
    "is not a real patient and contains no real PHI. Pt {pid} is a "
    "{age}-year-old {sex} presenting to the {dept} clinic with {complaint} "
    "for the past {duration} days. Past medical history is notable for "
    "{cond1} and {cond2}. Current medications include {med1} {dose1} mg "
    "{freq1} and {med2} {dose2} mg {freq2}. Vital signs on arrival: blood "
    "pressure {sys}/{dia} mmHg, heart rate {hr} beats per minute, "
    "temperature {temp} F, respiratory rate {rr} breaths per minute, "
    "oxygen saturation {spo2}% on room air. Laboratory studies show "
    "hemoglobin {hgb} g/dL, white cell count {wbc} K/uL, creatinine {cr} "
    "mg/dL, and glucose {glu} mg/dL. Assessment: {assess}. Plan: start "
    "{med3} {dose3} mg {freq3}, order {study}, and schedule follow-up in "
    "{fup}. The patient verbalized understanding of the plan and all "
    "questions were answered. Disposition: {disp}.")


def clinical_doc(rng, index):
    letters = "ABCDEFGHJKMNPQRSTVWXYZ"
    pid = (f"SYN-{rng.randrange(1000, 9999)}-"
           f"{rng.choice(letters)}{rng.choice(letters)}")
    cond1, cond2 = rng.sample(CONDITIONS, 2)
    med1, med2, med3 = rng.sample(MEDS, 3)
    text = TEMPLATE.format(
        pid=pid, age=rng.randrange(19, 89), sex=rng.choice(SEXES),
        dept=rng.choice(DEPTS), complaint=rng.choice(COMPLAINTS),
        duration=rng.randrange(2, 21), cond1=cond1, cond2=cond2,
        med1=med1, dose1=rng.choice([5, 10, 20, 40, 500, 850]),
        freq1=rng.choice(FREQS), med2=med2,
        dose2=rng.choice([5, 10, 25, 50, 100]), freq2=rng.choice(FREQS),
        sys=rng.randrange(98, 168), dia=rng.randrange(58, 99),
        hr=rng.randrange(52, 108), temp=round(rng.uniform(96.8, 100.4), 1),
        rr=rng.randrange(12, 24), spo2=rng.randrange(92, 100),
        hgb=round(rng.uniform(10.5, 16.5), 1),
        wbc=round(rng.uniform(3.5, 12.5), 1),
        cr=round(rng.uniform(0.6, 1.6), 1),
        glu=rng.randrange(72, 210),
        assess=rng.choice(ASSESSMENTS).format(cond=cond1),
        med3=med3, dose3=rng.choice([5, 10, 20, 40, 81]),
        freq3=rng.choice(FREQS), study=rng.choice(STUDIES),
        fup=rng.choice(FOLLOWUPS), disp=rng.choice(DISPOSITIONS))
    return {"id": f"clinical-synth-{index:04d}", "text": text}


def clinical_docs(rng, n):
    docs, seen = [], set()
    index = 0
    while len(docs) < n:
        index += 1
        doc = clinical_doc(rng, index)
        if doc["text"] in seen:  # pid makes this vanishingly unlikely
            continue
        seen.add(doc["text"])
        docs.append(doc)
    return docs


# ---------------------------------------------------------------------------
# The driver's partition rule (attacker.experiment.partition), replicated
# stdlib-side so the manifest can pre-register partition membership.

def partition_ids(docs, seed):
    ids = sorted(d["id"] for d in docs)
    random.Random(seed).shuffle(ids)
    n = len(ids) // 3
    return {"calibration": ids[:n],
            "victim_train": ids[n:2 * n],
            "utility_eval": ids[2 * n:]}


def corpus_entry(docs, path, source, limits, partition_seed):
    parts = partition_ids(docs, partition_seed)
    by_id = {d["id"]: d for d in docs}
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "n_docs": len(docs),
        "min_doc_chars": min(len(d["text"]) for d in docs),
        "max_doc_chars": max(len(d["text"]) for d in docs),
        "source": source,
        "limits": limits,
        "partition_counts": {k: len(v) for k, v in parts.items()},
        "selected_ids": parts,
        "document_sha256": {
            part: {i: sha256_text(by_id[i]["text"]) for i in ids}
            for part, ids in parts.items()}}


def write_jsonl(path, docs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for doc in docs:
            fh.write(json.dumps(doc) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--output-dir",
                    default="results/domain-shift-v1/inputs")
    ap.add_argument("--manifest",
                    default="experiments/domain_corpora_manifest.json")
    ap.add_argument("--docs", type=int, default=DOCS_PER_CORPUS)
    ap.add_argument("--seed", type=int, default=20260916,
                    help="selection/generation seed")
    ap.add_argument("--partition-seed", type=int, default=2026,
                    help="the experiment driver's default --data-seed")
    args = ap.parse_args(argv)
    root = Path(args.repo_root)
    if args.docs < 9:
        ap.error("the protocol requires >= 9 documents per corpus")
    out_dir = Path(args.output_dir)
    rng = random.Random(args.seed)

    # --- code corpus: chunk, then a seeded sample of the eligible chunks
    chunks = code_chunks(root)
    if len(chunks) < args.docs:
        raise SystemExit(
            f"only {len(chunks)} eligible code chunks under "
            f"{CODE_SOURCE_GLOB}; need {args.docs}")
    picked = sorted(rng.sample(chunks, args.docs))
    code_docs = [{"id": cid, "text": text} for cid, text in picked]
    code_path = out_dir / "code-corpus-v1.jsonl"
    write_jsonl(code_path, code_docs)

    # --- synthetic clinical corpus
    clin_docs = clinical_docs(rng, args.docs)
    clin_path = out_dir / "clinical-synth-corpus-v1.jsonl"
    write_jsonl(clin_path, clin_docs)

    # --- wikitext reference: reuse the pinned pilot inputs in place
    ref_path = WIKITEXT_REFERENCE
    if not ref_path.is_file():
        raise SystemExit(f"missing pinned pilot corpus: {ref_path}")
    ref_docs = [json.loads(line) for line in
                ref_path.read_text().splitlines() if line.strip()]
    ref_source = {"reuse": "pinned pilot inputs, used in place",
                  "dataset": "Salesforce/wikitext",
                  "configuration": "wikitext-2-raw-v1",
                  "source_split": "train"}
    ref_pin = None
    if WIKITEXT_REFERENCE_MANIFEST.is_file():
        pin = json.loads(WIKITEXT_REFERENCE_MANIFEST.read_text())
        ref_pin = pin.get("files", {}).get("pilot-corpus.jsonl")
        ref_source.update({k: pin.get(k) for k in
                           ("revision", "selection_seed")})
    actual = sha256_file(ref_path)
    if ref_pin is not None and ref_pin != actual:
        raise SystemExit(
            f"pilot corpus pin mismatch: manifest {ref_pin} != {actual}")

    manifest = {
        "schema": SCHEMA,
        "created": "2026-09-16",
        "purpose": ("domain-shifted seam-inversion evaluation (GPT-2 "
                    "split 6/12, vmi arm): leak magnitude and label-free "
                    "majority control rate across three domains, each "
                    "corpus with its own calibration partition"),
        "generator_script": "experiments/build_domain_corpora.py",
        "generator_script_sha256": sha256_file(__file__),
        "selection_seed": args.seed,
        "selection_salt": SALT,
        "partition_seed": args.partition_seed,
        "partition_rule": ("attacker.experiment.partition: sort by id, "
                           "random.Random(partition_seed).shuffle, "
                           "contiguous thirds"),
        "docs_per_corpus": args.docs,
        "sequence_length": 32,
        "token_floor": (f"every document >= {MIN_CHARS} chars (>> 32 "
                        "GPT-2 tokens); attacker.experiment re-checks "
                        ">= seq-len tokens at encode time"),
        "model": "openai-community/gpt2",
        "heldout_disjointness": (
            "results/validation-corpus-v1/ (frozen held-out) is never "
            "read: the code corpus is this repository's own source, the "
            "clinical corpus is templated synthetic text, and the "
            "reference corpus is the pilot WikiText-2 file whose sha256 "
            "the validation manifest records as excluded_pilot_sha256 — "
            "validation selection excluded these pilot articles."),
        "corpora": {
            "code": corpus_entry(
                code_docs, code_path,
                source={"kind": "repository source code",
                        "files": CODE_SOURCE_GLOB,
                        "license": "MIT (this repository)",
                        "eligible_chunks": len(chunks),
                        "chunking": (f"line-boundary accumulate to "
                                     f"{CODE_TARGET_CHARS} chars, "
                                     f"tail merge, exact-text dedupe, "
                                     f">= {MIN_CHARS} chars")},
                limits=["single repository, single language (Python)",
                        "public source stands in for private code",
                        "docstrings/comments included as written"],
                partition_seed=args.partition_seed),
            "clinical": corpus_entry(
                clin_docs, clin_path,
                source={"kind": "templated synthetic clinical-style notes",
                        "template": "single slot-filled note template "
                                    "with a synthetic-record banner",
                        "vocabularies": "fixed small slot vocabularies "
                                        "(complaints, conditions, "
                                        "medications, measurements)",
                        "patient_ids": "fabricated (SYN-####-XX); no "
                                       "real PHI exists by construction"},
                limits=["template regularity inflates the majority "
                        "control relative to natural clinical text — "
                        "that shift is the measurement, not a bug",
                        "no real clinical distribution is claimed"],
                partition_seed=args.partition_seed),
            "wikitext_reference": corpus_entry(
                ref_docs, ref_path,
                source=ref_source,
                limits=["identical to the campaign's pilot corpus "
                        "(comparability); not freshly curated here",
                        "pretraining exposure cannot be excluded; a "
                        "reconstruction study, not membership inference"],
                partition_seed=args.partition_seed)}}
    man_path = Path(args.manifest)
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for name, entry in manifest["corpora"].items():
        print(f"[domain-corpora] {name}: {entry['n_docs']} docs, "
              f"min {entry['min_doc_chars']} chars, "
              f"sha256 {entry['sha256'][:16]}... -> {entry['path']}")
    print(f"[domain-corpora] manifest -> {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
