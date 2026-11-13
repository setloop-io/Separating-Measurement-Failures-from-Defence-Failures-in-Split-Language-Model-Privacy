"""Verify the pinned inherited implementation without running experiments."""
import json
from p4d.experiment import verify_inherited_source

if __name__ == "__main__":
    print(json.dumps(verify_inherited_source(), indent=2))
