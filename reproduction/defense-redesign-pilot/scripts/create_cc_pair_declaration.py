#!/usr/bin/env python3
"""Create the signed, pre-arrival CC pair declaration."""
import argparse

from p4d.cc_campaign import create_pair_declaration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--deployment-manifest", required=True)
    parser.add_argument("--hardware-packet", required=True)
    parser.add_argument("--workload-config", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--checkpoint-artifact", required=True)
    parser.add_argument("--cc-off-journal-public-key", required=True)
    parser.add_argument("--cc-on-journal-public-key", required=True)
    parser.add_argument("--lifecycle-controller-public-key", required=True)
    parser.add_argument("--campaign-private-key", required=True)
    parser.add_argument("--campaign-public-key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--auth-output", required=True)
    args = parser.parse_args()
    create_pair_declaration(
        args.output, args.auth_output,
        campaign_id=args.campaign_id, pair_id=args.pair_id,
        deployment_manifest_path=args.deployment_manifest,
        hardware_packet_path=args.hardware_packet,
        workload_config_path=args.workload_config,
        checkpoint_manifest_path=args.checkpoint_manifest,
        checkpoint_artifact_path=args.checkpoint_artifact,
        cc_off_journal_public_key_path=args.cc_off_journal_public_key,
        cc_on_journal_public_key_path=args.cc_on_journal_public_key,
        lifecycle_controller_public_key_path=(
            args.lifecycle_controller_public_key),
        campaign_private_key_path=args.campaign_private_key,
        campaign_public_key_path=args.campaign_public_key,
    )
    print(args.output)
    print(args.auth_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
