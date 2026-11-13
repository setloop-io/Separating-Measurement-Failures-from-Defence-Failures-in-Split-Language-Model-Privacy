#!/usr/bin/env python3
"""Seal a CC-on/off workload bundle after independent external collection."""

import argparse

from p4d.cc_workload import seal_with_evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trusted-journal", required=True)
    parser.add_argument("--external-journal", required=True)
    parser.add_argument("--trusted-auth", required=True)
    parser.add_argument("--journal-public-key", required=True)
    parser.add_argument("--external-auth", required=True)
    parser.add_argument("--collector-manifest", required=True)
    parser.add_argument("--collector-public-key", required=True)
    parser.add_argument("--collector-journal-id", required=True)
    parser.add_argument("--evidence", required=True,
                        help="CC-on admission or CC-off arrival inventory")
    parser.add_argument("--artifact-manifest", required=True)
    parser.add_argument("--workload-config", required=True)
    parser.add_argument("--protection-mode", choices=("cc-on", "cc-off"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--paired-cc-on-admission",
                        help="Required for CC-off comparator sealing")
    args = parser.parse_args()
    seal_with_evidence(
        args.trusted_journal, args.external_journal, args.trusted_auth,
        args.journal_public_key, args.external_auth, args.collector_manifest,
        args.collector_public_key, args.collector_journal_id,
        args.evidence, args.artifact_manifest,
        args.output, args.protection_mode, args.workload_config,
        paired_cc_on_admission_path=args.paired_cc_on_admission,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
