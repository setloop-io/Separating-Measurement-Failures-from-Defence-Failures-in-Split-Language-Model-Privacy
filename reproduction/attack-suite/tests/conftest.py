import os
import sys

# allow `import attacker` regardless of how pytest was invoked
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# reporting plugin: every run writes tests/last_run.{json,md}
pytest_plugins = ["conftest_plugin"]
