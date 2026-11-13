#!/usr/bin/env bash
# Read-only confidential-compute arrival inventory.  This script never admits a
# host and never sets hardware_attested=true; admission is a separate, evidence-
# bound decision validated by scripts/validate_cc_arrival.py.
# Usage: bash scripts/cc_arrival_check.sh --context PATH --cuda-device INDEX [--out PATH]
set -uo pipefail

usage() {
  echo "usage: bash scripts/cc_arrival_check.sh --context PATH --cuda-device INDEX [--out PATH]" >&2
  exit 2
}

OUT="cc-arrival-$(date -u +%Y%m%dT%H%M%SZ).json"
CONTEXT=""
CUDA_DEVICE_INDEX=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --context) [ "$#" -ge 2 ] || usage; CONTEXT="$2"; shift 2 ;;
    --cuda-device) [ "$#" -ge 2 ] || usage; CUDA_DEVICE_INDEX="$2"; shift 2 ;;
    --out) [ "$#" -ge 2 ] || usage; OUT="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$CONTEXT" ] || usage
[ -n "$CUDA_DEVICE_INDEX" ] || usage
[ "$CUDA_DEVICE_INDEX" -ge 0 ] 2>/dev/null \
  || { echo "CUDA device index must be a nonnegative integer" >&2; exit 2; }
case "$CUDA_DEVICE_INDEX" in
  *[!0-9]*) echo "CUDA device index must be a nonnegative integer" >&2; exit 2 ;;
esac
[ -f "$CONTEXT" ] || { echo "deployment context is missing: $CONTEXT" >&2; exit 2; }

if [ -e "$OUT" ]; then
  echo "refusing to overwrite existing arrival record: $OUT" >&2
  exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to emit and validate the arrival record" >&2
  exit 2
fi

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  RUNTIME_PY="$REPO_ROOT/.venv/bin/python"
else
  RUNTIME_PY="python3"
fi
WORK="$(mktemp -d)"
PENDING="${OUT}.pending.$$"
cleanup() {
  rm -f "$PENDING"
  rm -r "$WORK"
}
trap cleanup EXIT

capture_command() {
  local key="$1"
  shift
  local code
  if "$@" >"$WORK/${key}.out" 2>&1; then
    code=0
  else
    code=$?
  fi
  printf '%s\n' "$code" >"$WORK/${key}.exit"
}

if command -v nvidia-smi >/dev/null 2>&1; then
  : >"$WORK/nvidia.present"
  capture_command gpu_identity nvidia-smi \
    --query-gpu=index,uuid,pci.bus_id,name,driver_version \
    --format=csv,noheader,nounits
  capture_command cuda_runtime_identity "$RUNTIME_PY" -c \
    'import json, sys, torch
i = int(sys.argv[1])
if not torch.cuda.is_available() or i < 0 or i >= torch.cuda.device_count():
    raise SystemExit("selected CUDA ordinal is unavailable")
nvml = int(torch.cuda._get_nvml_device_index(i))
print(json.dumps({"cuda_index": i, "nvidia_index": nvml,
                  "name": torch.cuda.get_device_name(i)}, sort_keys=True))' \
    "$CUDA_DEVICE_INDEX"
  capture_command gpu_cc_ready nvidia-smi conf-compute -grs
  # NVIDIA's -f option prints protected-memory information; it takes no filename.
  capture_command gpu_protected_memory nvidia-smi conf-compute -f
fi

if command -v dmesg >/dev/null 2>&1; then
  capture_command dmesg dmesg
fi
capture_command nvtrust python3 -c 'import nvtrust; print(getattr(nvtrust, "__version__", "import-ok"))'
capture_command nvidia_attestation_sdk python3 -c 'import nv_attestation_sdk; print(getattr(nv_attestation_sdk, "__version__", "import-ok"))'
capture_command python_version python3 --version
if command -v uv >/dev/null 2>&1; then
  capture_command uv_version uv --version
fi

python3 - "$WORK" "$PENDING" "$SCRIPT_PATH" "$REPO_ROOT" "$CONTEXT" \
  "$CUDA_DEVICE_INDEX" <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys


work, output_path, script_path, repo_root, context_path = map(Path, sys.argv[1:6])
selected_cuda_index = int(sys.argv[6])


def text(path: Path, limit: int = 16384) -> str:
    if not path.is_file():
        return ""
    value = path.read_text(errors="replace")
    if len(value) > limit:
        value = value[:limit] + "\n[truncated by collector]"
    return value.strip()


def command(name: str, attempted: bool) -> dict:
    if not attempted:
        return {"attempted": False, "exit_code": None, "output": "command not available"}
    return {
        "attempted": True,
        "exit_code": int(text(work / f"{name}.exit")),
        "output": text(work / f"{name}.out"),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filtered_dmesg(pattern: str) -> dict:
    base = command("dmesg", (work / "dmesg.exit").is_file())
    if not base["attempted"] or base["exit_code"] != 0:
        return base
    lines = [line for line in base["output"].splitlines()
             if re.search(pattern, line, flags=re.IGNORECASE)]
    base["output"] = "\n".join(lines[-20:]) or "no matching kernel messages"
    return base


nvidia_present = (work / "nvidia.present").is_file()
identity = command("gpu_identity", nvidia_present)
cuda_runtime_identity = command("cuda_runtime_identity", nvidia_present)
gpu_cuda_index = None
gpu_nvidia_index = None
gpu_uuid = None
gpu_pci_bus_id = None
gpu_name = None
gpu_driver = None
if identity["exit_code"] == 0 and cuda_runtime_identity["exit_code"] == 0:
    try:
        runtime = json.loads(cuda_runtime_identity["output"])
        if (set(runtime) != {"cuda_index", "nvidia_index", "name"}
                or isinstance(runtime["cuda_index"], bool)
                or isinstance(runtime["nvidia_index"], bool)
                or not isinstance(runtime["cuda_index"], int)
                or not isinstance(runtime["nvidia_index"], int)
                or runtime["cuda_index"] != selected_cuda_index):
            raise ValueError("CUDA runtime identity fields differ")
        rows = [[field.strip() for field in row]
                for row in csv.reader(identity["output"].splitlines())]
        matches = [row for row in rows
                   if len(row) == 5 and row[0].isdigit()
                   and int(row[0]) == runtime["nvidia_index"]]
        if len(matches) != 1:
            raise ValueError("selected NVIDIA index is not unique")
        row = matches[0]
        if runtime["name"].strip() != row[3]:
            raise ValueError("CUDA and NVIDIA names differ")
        gpu_cuda_index = runtime["cuda_index"]
        gpu_nvidia_index = runtime["nvidia_index"]
        gpu_uuid = row[1]
        gpu_pci_bus_id = row[2]
        gpu_name = row[3]
        gpu_driver = row[4]
    except (json.JSONDecodeError, TypeError, ValueError):
        # Keep identity null.  Strict schema/semantic validation below then
        # rejects the record instead of guessing a physical GPU.
        pass

try:
    git_commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
except (FileNotFoundError, subprocess.CalledProcessError):
    git_commit = None
if not git_commit or not re.fullmatch(r"[a-f0-9]{40}", git_commit):
    git_commit = None

cpu_model = platform.processor().strip()
cpuinfo = Path("/proc/cpuinfo")
if cpuinfo.is_file():
    for line in cpuinfo.read_text(errors="replace").splitlines():
        if line.lower().startswith("model name") and ":" in line:
            cpu_model = line.split(":", 1)[1].strip()
            break
cpu_model = cpu_model or platform.machine() or "unknown"

tdx_path = Path("/sys/module/tdx_module")
if tdx_path.is_dir():
    version_path = tdx_path / "version"
    tdx_module = "present: " + (text(version_path) or "version unavailable")
else:
    tdx_module = "not present or not readable"

nvtrust = command("nvtrust", True)
nvidia_sdk = command("nvidia_attestation_sdk", True)
python_version = command("python_version", True)
uv_attempted = (work / "uv_version.exit").is_file()
uv_version = command("uv_version", uv_attempted)
lock_path = repo_root / "uv.lock"

try:
    context = json.loads(context_path.read_text())
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid deployment context: {exc}")
context_keys = {
    "provider", "candidate_id", "sku", "region", "zone", "instance_id",
    "image_reference", "protection_mode", "pair_id",
    "cuda_device_index", "deployment_manifest_sha256", "pair_manifest_sha256",
    "provider_metadata_path",
}
if not isinstance(context, dict) or set(context) != context_keys:
    raise SystemExit("deployment context fields differ")
if context["provider"] not in {"azure", "gcp"}:
    raise SystemExit("deployment context provider is invalid")
if context["protection_mode"] not in {"cc-on", "cc-off"}:
    raise SystemExit("deployment context protection_mode is invalid")
if not re.fullmatch(r"[A-Za-z0-9_.:-]+", str(context["pair_id"])):
    raise SystemExit("deployment context pair_id is invalid")
if not re.fullmatch(r"[a-f0-9]{64}", str(context["deployment_manifest_sha256"])):
    raise SystemExit("deployment manifest digest is invalid")
if not re.fullmatch(r"[a-f0-9]{64}", str(context["pair_manifest_sha256"])):
    raise SystemExit("pair manifest digest is invalid")
if context["pair_manifest_sha256"] == context["deployment_manifest_sha256"]:
    raise SystemExit("pair and arm deployment manifests must be distinct")
if (isinstance(context["cuda_device_index"], bool)
        or not isinstance(context["cuda_device_index"], int)
        or context["cuda_device_index"] != selected_cuda_index):
    raise SystemExit("deployment context CUDA device differs from --cuda-device")
provider_metadata_path = Path(context["provider_metadata_path"])
if not provider_metadata_path.is_absolute():
    provider_metadata_path = (context_path.parent / provider_metadata_path).resolve()
if not provider_metadata_path.is_file() or provider_metadata_path.stat().st_size < 1:
    raise SystemExit("provider metadata evidence is missing")
for key in ("candidate_id", "sku", "region", "zone", "instance_id", "image_reference"):
    if not isinstance(context[key], str) or not context[key].strip():
        raise SystemExit(f"deployment context {key} is missing")

boot_id_path = Path("/proc/sys/kernel/random/boot_id")
try:
    boot_id = boot_id_path.read_text(encoding="ascii").strip().lower()
except (OSError, UnicodeError) as exc:
    raise SystemExit(f"kernel boot identity is unavailable: {exc}")
if not re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        boot_id):
    raise SystemExit("kernel boot identity has an invalid format")
boot_id_sha256 = hashlib.sha256(boot_id.encode("ascii")).hexdigest()

record = {
    "schema_version": "p4-cc-arrival-inventory-0.4",
    "record_type": "inventory_only",
    "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "host": platform.node() or "unknown",
    "collector": {"script_sha256": sha256(script_path), "git_commit": git_commit},
    "claim_scope": ("Arrival inventory only; this record cannot admit hardware or set "
                    "hardware_attested true."),
    "hardware_attested": False,
    "admission_status": "pending_separate_validation",
    "deployment": {
        "provider": context["provider"],
        "candidate_id": context["candidate_id"],
        "sku": context["sku"],
        "region": context["region"],
        "zone": context["zone"],
        "instance_id": context["instance_id"],
        "boot_id_sha256": boot_id_sha256,
        "image_reference": context["image_reference"],
        "protection_mode": context["protection_mode"],
        "pair_id": context["pair_id"],
        "cuda_device_index": context["cuda_device_index"],
        "deployment_manifest_sha256": context["deployment_manifest_sha256"],
        "pair_manifest_sha256": context["pair_manifest_sha256"],
        "context_sha256": sha256(context_path),
        "provider_metadata_sha256": sha256(provider_metadata_path),
    },
    "gpu": {
        "nvidia_smi_present": nvidia_present,
        "identity_query": identity,
        "cuda_runtime_query": cuda_runtime_identity,
        "cuda_index": gpu_cuda_index,
        "nvidia_index": gpu_nvidia_index,
        "uuid": gpu_uuid,
        "pci_bus_id": gpu_pci_bus_id,
        "name": gpu_name,
        "driver": gpu_driver,
        "cc_readiness": command("gpu_cc_ready", nvidia_present),
        "protected_memory": command("gpu_protected_memory", nvidia_present),
    },
    "cpu": {
        "model": cpu_model,
        "tdx_evidence": filtered_dmesg(r"\btdx\b"),
        "sev_snp_evidence": filtered_dmesg(r"\bsev(?:-snp)?\b|\bsnp\b"),
        "tdx_module": tdx_module,
    },
    "attestation_tools": {
        "nvtrust": {"available": nvtrust["exit_code"] == 0,
                    "detail": nvtrust["output"] or "import failed without output"},
        "nvidia_attestation_sdk": {"available": nvidia_sdk["exit_code"] == 0,
                                   "detail": nvidia_sdk["output"] or
                                   "import failed without output"},
    },
    "environment": {
        "python": python_version["output"] or "version unavailable",
        "uv": uv_version["output"] if uv_attempted and uv_version["exit_code"] == 0 else None,
        "uv_lock_sha256": sha256(lock_path) if lock_path.is_file() else None,
    },
    "advisory_inventory": [
        "CVE-2025-31356: Intel TDX module release mapping",
        "CVE-2025-9613: TDX Connect/IDE mapping",
        "CVE-2024-56161: AMD SEV-SNP microcode mapping",
        "physical DDR5 interposition exposure",
        "host-side microarchitectural tracing exposure",
    ],
    "unassessed_from_guest": [
        "physical interposition attacks (TEE.fail/WireTap/BatteringRAM)",
        "host-side microarchitectural tracing (TDXRay class)",
        "hypervisor-internal behaviour not exposed to the guest",
    ],
}
output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY

if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  VALIDATOR_PY="$REPO_ROOT/.venv/bin/python"
else
  VALIDATOR_PY="python3"
fi
if ! PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$VALIDATOR_PY" "$REPO_ROOT/scripts/validate_cc_arrival.py" --inventory "$PENDING"; then
  echo "arrival inventory failed strict validation; no final record was written" >&2
  exit 1
fi
if ! ln "$PENDING" "$OUT"; then
  echo "refusing to replace an arrival record created concurrently: $OUT" >&2
  exit 2
fi
rm -f "$PENDING"
echo "wrote validated inventory-only record: $OUT"
