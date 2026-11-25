#!/usr/bin/env python3
"""Seal the exact CC-off/CC-on engineering campaign."""
import argparse

from p4d.cc_campaign import seal_campaign


ARM_ARGUMENTS = (
    "session-id", "evidence-record", "arrival-record", "repetition-set",
    "repetition-set-auth", "journal-public-key", "collector-manifest",
    "collector-public-key", "lifecycle-receipt", "teardown-argv",
    "absence-verifier-argv",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-declaration", required=True)
    parser.add_argument("--pair-declaration-auth", required=True)
    parser.add_argument("--campaign-private-key", required=True)
    parser.add_argument("--campaign-public-key", required=True)
    parser.add_argument("--runtime-binding", required=True)
    parser.add_argument("--active-security-manifest", required=True)
    parser.add_argument("--bootstrap-lifecycle", required=True)
    parser.add_argument("--active-security-transition", required=True)
    parser.add_argument("--cc-on-lifecycle-launch-manifest", required=True)
    parser.add_argument("--cc-off-lifecycle-launch-manifest", required=True)
    parser.add_argument("--lifecycle-controller-public-key", required=True)
    parser.add_argument("--deployment-manifest", required=True)
    parser.add_argument("--hardware-packet", required=True)
    parser.add_argument("--workload-config", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--checkpoint-artifact", required=True)
    parser.add_argument("--campaign-ledger", required=True)
    for arm in ("cc-on", "cc-off"):
        for name in ARM_ARGUMENTS:
            parser.add_argument(f"--{arm}-{name}", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--auth-output", required=True)
    args = parser.parse_args()

    def arm_inputs(arm: str) -> dict[str, str]:
        return {
            name.replace("-", "_"): getattr(
                args, f"{arm}_{name}".replace("-", "_"))
            for name in ARM_ARGUMENTS
        }

    seal_campaign(
        args.output, args.auth_output,
        pair_declaration_path=args.pair_declaration,
        pair_declaration_auth_path=args.pair_declaration_auth,
        campaign_private_key_path=args.campaign_private_key,
        campaign_public_key_path=args.campaign_public_key,
        runtime_binding_path=args.runtime_binding,
        active_security_manifest_path=args.active_security_manifest,
        bootstrap_lifecycle_path=args.bootstrap_lifecycle,
        active_security_transition_path=args.active_security_transition,
        lifecycle_launch_manifest_paths={
            "cc-on": args.cc_on_lifecycle_launch_manifest,
            "cc-off": args.cc_off_lifecycle_launch_manifest,
        },
        lifecycle_controller_public_key_path=(
            args.lifecycle_controller_public_key),
        deployment_manifest_path=args.deployment_manifest,
        hardware_packet_path=args.hardware_packet,
        workload_config_path=args.workload_config,
        checkpoint_manifest_path=args.checkpoint_manifest,
        checkpoint_artifact_path=args.checkpoint_artifact,
        campaign_ledger_path=args.campaign_ledger,
        arms={arm: arm_inputs(arm) for arm in ("cc-on", "cc-off")},
    )
    print(args.output)
    print(args.auth_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
