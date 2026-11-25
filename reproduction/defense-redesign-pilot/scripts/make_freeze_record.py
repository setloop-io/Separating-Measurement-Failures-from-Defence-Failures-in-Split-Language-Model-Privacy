"""Generate a pilot freeze record (test-mode); the real record is produced at freeze time."""
import argparse
import json
from p4d.freeze import make_freeze_record

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Pilot config path to bind")
    parser.add_argument("--protocol-doc", required=True, help="Freeze document path to bind")
    parser.add_argument("--seed-audit", required=True, help="Passed seed-audit JSON to bind")
    parser.add_argument("--qualification-verification", required=True,
                        help="Qualification verification.json to bind")
    parser.add_argument("--out", required=True, help="Freeze record output path")
    parser.add_argument("--freeze-id", default=None)
    args = parser.parse_args()
    kwargs = {"freeze_id": args.freeze_id} if args.freeze_id else {}
    record = make_freeze_record(args.config, args.protocol_doc, args.seed_audit,
                                args.qualification_verification, args.out, **kwargs)
    print(json.dumps(record, indent=2))
