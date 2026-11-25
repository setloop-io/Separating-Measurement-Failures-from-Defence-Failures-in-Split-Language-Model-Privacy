"""One-shot controller for six precommitted, explicitly authorised SSH runs.

Only ORX launches/cancels models. SSH and rsync read hardware/evidence. No retries,
repairs, experiment creation, source mutation, paid cloud or manuscript work.
"""
import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
TASK = "experiments/vflair_native_replication_20261022"
DEFAULT_MANIFEST = ROOT / "artifacts/paper-b-vflair-replication-20261022/bounded-study/study-manifest.json"
ACTIVE = {"running", "queued", "pending", "preparing", "starting", "submitted"}
FAILED = {"failed", "cancelled", "canceled"}
UUID = r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}"
RESERVE = 20 * 2**30
LIVE_COMPLETION_SECONDS = 27000 + 600 + 120


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def exclusive_lock(path):
    with Path(path).open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def command(args, timeout=120):
    result = subprocess.run([str(x) for x in args], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Command failed {args[:3]}: {result.stdout[-4000:]}")
    return result.stdout


def parse_runs(text):
    result,diagnostics = {},set()
    for line in text.splitlines():
        fields = line.split()
        if fields and re.fullmatch(UUID, fields[0]):
            if len(fields)>=3 and fields[1]=="reason:":
                diagnostics.add(fields[0]);continue
            if len(fields)<2 or fields[0] in result or fields[1] not in ACTIVE | FAILED | {"done"}:
                raise ValueError("Unknown/duplicate ORX run status row")
            result[fields[0]] = fields[1]
    if not diagnostics.issubset(result):raise ValueError("ORX reason diagnostic has no corresponding run row")
    return result


def validate_manifest(manifest):
    auth, entries = manifest["authorization"], manifest["entries"]
    if (manifest["schema"] != "bounded-native-study-launch-v1" or len(entries) != 6
            or auth["maximum_aggregate_gpu_hours"] != 48 or auth["reserved_gpu_hours"] != 45
            or auth["per_run_seconds"] != 27000 or auth["paid_cloud_authorized"] is not False
            or auth["automatic_repairs_or_additional_nodes_authorized"] is not False
            or manifest["fixed_run_command"] != f"bash {TASK}/entry.sh"):
        raise ValueError("Unexpected six-run authorization/reservation/command")
    expected = {(host, condition, seed) for host, condition in [("toula", "mid"), ("koula", "none")] for seed in [11,22,33]}
    if {(x["host"],x["condition"],x["seed"]) for x in entries} != expected:
        raise ValueError("Study cell/host assignment changed")
    if len({x["experiment_id"] for x in entries}) != 6:
        raise ValueError("Duplicate experiment in six-run manifest")
    for entry in entries:
        if not re.fullmatch(UUID, entry["experiment_id"]) or not re.fullmatch(r"[a-f0-9]{40}", entry["commit"]) or entry["wall_seconds"] != 27000:
            raise ValueError("Invalid frozen node identity or guard")


def verify_indices(out, expected_captures=400, expected_total=None):
    out=Path(out).resolve();hashed=[];seen=set()
    for index in ["checkpoint-index.json","capture-index.json"]:
        records=json.loads((out/index).read_text())
        if not records or (index=="capture-index.json" and len(records)!=expected_captures):
            raise ValueError("Incomplete retained artifact index")
        for item in records:
            path=(out/item["path"]).resolve()
            if (item["path"] in seen or not path.is_relative_to(out) or path.stat().st_size!=item["bytes"]
                    or sha(path)!=item["sha256"]):
                raise ValueError("Retained full artifact hash/path mismatch")
            seen.add(item["path"]);hashed.append({"path":item["path"],"sha256":item["sha256"],"bytes":item["bytes"]})
    if expected_total is not None and len(hashed)!=expected_total:raise ValueError("Verified artifact count/index mismatch")
    return hashed


class Controller:
    def __init__(self, manifest_path, state_path, pair_module, pair_sha, execute=command, clock=time.time, free=None,
                 capture_module=None,capture_sha=None,audit_python=None):
        self.manifest_path, self.state_path = Path(manifest_path), Path(state_path)
        self.manifest = json.loads(self.manifest_path.read_text()); validate_manifest(self.manifest)
        self.digest = sha(self.manifest_path)
        self.execute, self.clock = execute, clock
        self.free = free or (lambda: shutil.disk_usage(self.state_path.parent).free)
        self.pair_module, self.pair_sha = Path(pair_module), pair_sha
        self.capture_module=Path(capture_module) if capture_module else None
        self.capture_sha=capture_sha;self.audit_python=Path(audit_python) if audit_python else ROOT/".venv/bin/python"
        self.stop_request=self.manifest_path.with_name("STOP_REQUESTED")
        self.journal = self.state_path.with_name("controller-journal.jsonl")
        self.specs = {x["experiment_id"]:x for x in self.manifest["entries"]}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        else:
            self.state = {"schema":"bounded-study-controller-v1", "manifest_sha256":self.digest, "stop":False,
                          "entries":{key:{"state":"pending", "run_id":None, "host":x["host"], "seed":x["seed"],
                                          "condition":x["condition"], "intent_unix":None} for key,x in self.specs.items()}}
        if self.state["schema"] != "bounded-study-controller-v1" or self.state["manifest_sha256"] != self.digest or set(self.state["entries"]) != set(self.specs):
            raise ValueError("Controller state/manifest binding differs")
        for key, entry in self.state["entries"].items():
            if entry["state"] not in {"pending","launch_intent","running","verified"} or any(entry[k] != self.specs[key][k] for k in ["host","seed","condition"]):
                raise ValueError("State cell identity differs from manifest")
            if entry.get("run_id") and not re.fullmatch(UUID,entry["run_id"]):
                raise ValueError("Invalid recorded run ID")
        if self.state.get("pair_audit_sha256",pair_sha) != pair_sha or sha(self.pair_module) != pair_sha:
            raise ValueError("Pair-audit source changed")
        self.state["pair_audit_sha256"] = pair_sha
        if capture_sha:
            if (not self.capture_module or sha(self.capture_module)!=capture_sha
                    or self.state.get("capture_audit_sha256",capture_sha)!=capture_sha):
                raise ValueError("Capture-audit source changed")
            self.state["capture_audit_sha256"]=capture_sha

    def check_operator_stop(self):
        if self.stop_request.exists():raise RuntimeError("Operator STOP_REQUESTED file is present")

    def save(self, event, **values):
        self.state["updated_unix"] = self.clock()
        record = {"event":event,"unix_time":self.clock(),**values}
        self.journal.parent.mkdir(parents=True,exist_ok=True)
        with self.journal.open("a") as handle:
            handle.write(json.dumps(record,allow_nan=False)+"\n");handle.flush();os.fsync(handle.fileno())
        atomic_write(self.state_path,self.state)
        print(json.dumps(record,allow_nan=False),flush=True)

    def historical(self):
        total, ids = 0.0,set()
        for attempt in self.state.get("failed_attempts",[]):
            seconds = attempt["charged_seconds"]
            if (attempt["experiment_id"] not in self.specs or not re.fullmatch(UUID,attempt["run_id"])
                    or attempt["run_id"] in ids or isinstance(seconds,bool)
                    or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<0):
                raise ValueError("Invalid historical failure budget record")
            ids.add(attempt["run_id"]);total += seconds
        return total,ids

    def budget(self):
        historical,_ = self.historical();spent=historical;reserved=historical
        for entry in self.state["entries"].values():
            if entry["state"] == "verified":
                value = entry["charged_seconds"]
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                    raise ValueError("Invalid completed charge")
                spent += value;reserved += value
            else:
                intent=entry.get("intent_unix")
                if intent is not None and (isinstance(intent,bool) or not isinstance(intent,(int,float)) or not math.isfinite(intent)):
                    raise ValueError("Invalid launch/accounting timestamp")
                elapsed = max(0,self.clock()-intent) if intent is not None else 0
                spent += elapsed;reserved += max(27000,elapsed)
        if spent>=48*3600 or reserved>48*3600:
            raise RuntimeError("Fresh 48 aggregate GPU-hour spend/reservation ceiling reached")
        self.state["budget"] = {"spent_seconds_conservative":spent,"remaining_reservation_seconds":reserved-spent,
                                "total_actual_plus_reserved_seconds":reserved,"cap_seconds":48*3600,"historical_failure_seconds":historical}
        return spent,reserved

    def runs(self, key):
        return parse_runs(self.execute(["orx","runs",self.manifest["project_id"],"--experiment",key]))

    def binding(self, key, run_id=None):
        spec = self.specs[key]
        text = self.execute(["orx","exp","status",key])
        if not re.search(r"^\s*branch:\s*"+re.escape(spec["branch"])+r"\s*$",text,re.M) or not re.search(r"^\s*command:\s*"+re.escape(self.manifest["fixed_run_command"])+r"\s*$",text,re.M):
            raise ValueError("ORX branch/fixed command mismatch")
        if run_id:
            if not re.search(r"last run:\s*"+re.escape(run_id)+r"\b",text) or not re.search(r"^\s*commit:\s*"+spec["commit"]+r"\s*$",text,re.M):
                raise ValueError("ORX recorded run/commit mismatch")

    def reconcile(self):
        _,historical = self.historical()
        observations = {}
        for key, entry in self.state["entries"].items():
            rows = {rid:status for rid,status in self.runs(key).items() if rid not in historical}
            if len(rows)>1:raise RuntimeError("Multiple current runs on a one-shot study node")
            if entry["state"] == "pending":
                if rows:raise RuntimeError("Unrecorded run on pending node; refusing blind adoption")
                continue
            if entry["state"] == "launch_intent" and not entry.get("run_id"):
                if len(rows)!=1:raise RuntimeError("Ambiguous launch intent; never retry submission automatically")
                rid = next(iter(rows));self.binding(key,rid)
                host = entry["host"]
                exists = self.execute(["ssh","-o","BatchMode=yes",host,"test","-d",f"/home/geo/.orx/runs/{rid}"])
                entry.update(state="running",run_id=rid)
                self.save("launch_intent_reconciled",experiment_id=key,run_id=rid,host=host)
            rid = entry.get("run_id")
            if rid not in rows:raise RuntimeError("Recorded run missing from ORX truth")
            self.binding(key,rid);observations[key]=rows[rid]
            if rows[rid] in FAILED:raise RuntimeError(f"Required study run failed/cancelled: {rid}")
            if entry["state"] == "verified" and rows[rid]!="done":raise RuntimeError("Verified run is no longer ORX terminal-successful")
        return observations

    def progress(self,key,enforce_live=True):
        entry=self.state["entries"][key]
        text=self.execute(["orx","logs",entry["run_id"]])
        found=re.search(r"full log file:\s*([^\n]+)",text)
        if not found:raise RuntimeError("Cannot locate the persisted ORX evidence log")
        log=Path(found.group(1).strip())
        size=log.stat().st_size
        if size < entry.get("log_bytes",0):raise RuntimeError("ORX log shrank; progress provenance is ambiguous")
        if size > entry.get("log_bytes",0):entry["last_progress_unix"]=self.clock()
        entry.update(log_bytes=size,local_orx_log=str(log))
        if enforce_live and self.clock()-entry.get("last_progress_unix",entry["intent_unix"])>1800:
            raise TimeoutError("No persisted run progress for 1800 seconds")
        if enforce_live and self.clock()-entry["intent_unix"]>LIVE_COMPLETION_SECONDS:
            raise TimeoutError("Run exceeded its 27000-second guard plus bounded cleanup allowance")

    def source_config(self,key):
        spec=self.specs[key];tree=Path(spec["worktree"])
        if self.execute(["git","-C",str(tree),"rev-parse","HEAD"]).strip()!=spec["commit"]:
            raise ValueError("Precommitted worktree HEAD changed")
        if self.execute(["git","-C",str(tree),"status","--porcelain"]).strip():raise ValueError("Precommitted worktree is dirty")
        kit=tree/TASK
        if sha(kit/"run.json")!=spec["run_config_sha256"]:raise ValueError("Frozen run config bytes changed")
        for name,digest in spec["source_files"].items():
            if sha(kit/name)!=digest:raise ValueError("Frozen source bytes changed: "+name)
        cfg=json.loads((kit/"run.json").read_text());relative=Path(cfg["output_root"])
        if relative.is_absolute() or ".." in relative.parts:raise ValueError("Output root escaped archive")
        return cfg

    def launch(self,key):
        self.check_operator_stop()
        entry=self.state["entries"][key];spec=self.specs[key]
        if entry["state"]!="pending" or entry.get("run_id"):raise RuntimeError("Only a never-submitted pending node may launch")
        self.budget();self.source_config(key);self.binding(key)
        if self.runs(key):raise RuntimeError("Never launch a node with any previous run automatically")
        if any(x["state"] in {"running","launch_intent"} and x["host"]==entry["host"] for x in self.state["entries"].values()):
            raise RuntimeError("A model job is already recorded on this host")
        busy=self.execute(["ssh","-o","BatchMode=yes",entry["host"],"nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"]).strip()
        if busy:raise RuntimeError("Host has an existing GPU compute process; no concurrent model job admitted")
        entry.update(state="launch_intent",intent_unix=self.clock())
        self.save("LAUNCH_INTENT",experiment_id=key,host=entry["host"],commit=spec["commit"])
        self.check_operator_stop()
        # Never repeat this command on a CLI failure/timeout or unreadable reply.
        text=self.execute(["orx","exp","run",key,"--backend","ssh","--host",entry["host"],"--no-container"],timeout=900)
        entry["launch_stdout"]=text
        found=re.search(r"^\s*run\s+("+UUID+r")\b",text,re.M)
        if not found:raise RuntimeError("Ambiguous launch reply; reconcile journal/ORX, never relaunch")
        entry.update(state="running",run_id=found.group(1),last_progress_unix=self.clock())
        self.save("run_submitted",experiment_id=key,run_id=entry["run_id"],host=entry["host"])
        self.binding(key,entry["run_id"])

    def retain_verify(self,key):
        entry=self.state["entries"][key];spec=self.specs[key];cfg=self.source_config(key)
        rid=entry["run_id"];out=self.manifest_path.parent/"retained-runs"/rid;out.mkdir(parents=True,exist_ok=True)
        remote=f'{entry["host"]}:/home/geo/.orx/runs/{rid}/repo/{cfg["output_root"]}/{rid}/'
        excludes=["--exclude=/runtime-upstream/","--exclude=/dataset-cache/","--exclude=/model-input/"]
        stats=self.execute(["rsync","-an","--stats",*excludes,remote,str(out)+"/"],timeout=300)
        match=re.search(r"Total file size:\s*([\d,]+)",stats)
        if not match:raise RuntimeError("Cannot reserve disk for full evidence transfer")
        if self.free()<RESERVE+int(match.group(1).replace(",","")):raise RuntimeError("Full evidence would violate the 20 GiB local free reserve")
        self.execute(["rsync","-a","--partial",*excludes,remote,str(out)+"/"],timeout=1800)
        if self.free()<RESERVE:raise RuntimeError("Local evidence reserve fell below 20 GiB")
        read=lambda name:json.loads((out/name).read_text())
        terminal,verification,actual=read("terminal.json"),read("verification.json"),read("effective-run.json")
        if actual!=cfg or terminal["status"]!="complete" or terminal["exit_code"]!=0:
            raise ValueError("Run terminal/effective configuration mismatch")
        if (verification["status"]!="PASS" or verification["phase"]!="bounded-study" or verification["condition"]!=entry["condition"]
                or verification["utility_rows_verified"]!=5153 or verification["prediction_rows_verified"]!=400 or verification["capture_files_verified"]!=400):
            raise ValueError("Complete bounded evidence verification did not pass")
        provenance=read("launcher-provenance.json")
        if provenance["run_id"]!=rid or provenance["run_json_sha256"]!=spec["run_config_sha256"]:
            raise ValueError("Launcher run/config provenance mismatch")
        for name,digest in spec["source_files"].items():
            if provenance["task_files"].get(name)!=digest:raise ValueError("Executed source hash differs: "+name)
        hashed=verify_indices(out,expected_total=verification["artifact_hashes_verified"])
        if sha(out/"native-final-optimizer-rng.pt")!=verification["optimizer_rng_sha256"]:
            raise ValueError("Retained optimizer/RNG payload hash mismatch")
        wire_receipt=self.wire_audit(out)
        self.progress(key,enforce_live=False)
        log=out/"actual-orx.log";shutil.copyfile(entry["local_orx_log"],log)
        if '"event": "bounded_retained_evidence_verified"' not in log.read_text():
            raise ValueError("Actual ORX log lacks post-run retained-evidence verification")
        receipt={"status":"VERIFIED_AND_RETAINED","run_id":rid,"experiment_id":key,"commit":spec["commit"],
                 "verification_sha256":sha(out/"verification.json"),"actual_orx_log_sha256":sha(log),"indexed_files":hashed,
                 "optimizer_rng_sha256":verification["optimizer_rng_sha256"],"retention_excludes":excludes,
                 "independent_wire_audit_sha256":sha(wire_receipt),"wire_audit_source_sha256":self.capture_sha}
        atomic_write(out/"controller-retention.json",receipt)
        elapsed=max(float(terminal["wall_seconds"]),entry["terminal_observed_unix"]-entry["intent_unix"])
        entry.update(state="verified",output_directory=str(out),charged_seconds=elapsed,
                     retention_receipt_sha256=sha(out/"controller-retention.json"))
        self.save("run_verified_and_retained",experiment_id=key,run_id=rid,charged_seconds=elapsed)

    def wire_audit(self,out):
        if not self.capture_module or not self.capture_sha or sha(self.capture_module)!=self.capture_sha:
            raise ValueError("A frozen independent CPU wire auditor is required before refill")
        receipt=Path(out)/"independent-wire-audit.json"
        self.execute(["env","CUDA_VISIBLE_DEVICES=","OMP_NUM_THREADS=4","MKL_NUM_THREADS=4",str(self.audit_python),
                      str(self.capture_module),"--output",str(out),"--receipt-out",str(receipt)],timeout=900)
        result=json.loads(receipt.read_text())
        if (result.get("status")!="PASS" or result.get("capture_files_verified")!=400
                or result.get("wire_tensors_verified")!=400 or result.get("source_sha256")!=self.capture_sha):
            raise ValueError("Independent CPU wire audit did not verify all400 packets with pinned source")
        return receipt

    def pairs(self):
        if sha(self.pair_module)!=self.pair_sha:raise ValueError("Pair verifier source changed")
        dependency=Path(__file__).with_name("bounded_study.py")
        dependency_sha=next(iter(self.specs.values()))["source_files"]["bounded_study.py"]
        if sha(dependency)!=dependency_sha:raise ValueError("Pair audit selector dependency changed")
        module=None
        for seed in [11,22,33]:
            entries=[x for x in self.state["entries"].values() if x["seed"]==seed]
            if all(x["state"]=="verified" for x in entries):
                outputs={x["condition"]:Path(x["output_directory"]) for x in entries}
                receipt=self.manifest_path.parent/"pair-audits"/f"seed-{seed}.json";receipt.parent.mkdir(exist_ok=True)
                proof={"pair_audit_source_sha256":self.pair_sha,"selector_source_sha256":dependency_sha}
                for entry in entries:
                    digest=sha(Path(entry["output_directory"])/"controller-retention.json")
                    if digest!=entry["retention_receipt_sha256"]:raise ValueError("Previously retained evidence receipt changed")
                    proof[entry["condition"]+"_retention_sha256"]=digest
                cached=self.state.get("pair_audits",{}).get(str(seed),{})
                if (cached.get("proof")==proof and cached.get("path")==str(receipt) and receipt.is_file()
                        and sha(receipt)==cached.get("sha256")):
                    continue
                if module is None:
                    spec=importlib.util.spec_from_file_location("bounded_controller_pair_audit",self.pair_module)
                    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                result=module.verify_pair(outputs["mid"],outputs["none"],receipt)
                if result.get("status")!="VERIFIED":raise ValueError("Paired identity/evidence gate failed")
                self.state.setdefault("pair_audits",{})[str(seed)]={"path":str(receipt),"sha256":sha(receipt),"proof":proof}

    def tick(self):
        self.check_operator_stop()
        if self.state["stop"]:raise RuntimeError("STOP is latched; automatic resume is forbidden")
        self.budget();observations=self.reconcile()
        for key,status in observations.items():
            if self.state["entries"][key]["state"]=="verified":continue
            if status=="done":
                self.state["entries"][key].setdefault("terminal_observed_unix",self.clock())
                self.save("terminal_observed",experiment_id=key,run_id=self.state["entries"][key]["run_id"])
                self.retain_verify(key)
            else:self.progress(key)
        self.pairs();self.budget();self.save("reconciled")
        if all(x["state"]=="verified" for x in self.state["entries"].values()):
            self.save("COMPLETE_SIX_RUNS_VERIFIED");return True
        for host in ["toula","koula"]:
            if any(x["host"]==host and x["state"] in {"running","launch_intent"} for x in self.state["entries"].values()):continue
            pending=[key for key,x in self.state["entries"].items() if x["host"]==host and x["state"]=="pending"]
            if pending:self.launch(min(pending,key=lambda k:self.specs[k]["seed"]))
        return False

    def stop(self,error):
        self.state["stop"]=True;self.state.setdefault("stop_reason",str(error));self.save("STOP",error=str(error))
        for key,entry in self.state["entries"].items():
            if entry["state"] not in {"running","launch_intent"}:continue
            try:
                if not entry.get("run_id"):
                    _,historical=self.historical()
                    rows={rid:status for rid,status in self.runs(key).items() if rid not in historical}
                    if len(rows)!=1:raise RuntimeError("Unresolved launch outcome needs manual reconciliation")
                    rid=next(iter(rows));self.binding(key,rid)
                    self.execute(["ssh","-o","BatchMode=yes",entry["host"],"test","-d",f"/home/geo/.orx/runs/{rid}"])
                    entry["run_id"]=rid
                    self.save("stop_adopted_unique_submitted_run",experiment_id=key,run_id=rid)
                status=self.runs(key).get(entry["run_id"])
                if status in ACTIVE:
                    self.binding(key,entry["run_id"])
                    reply=self.execute(["orx","exp","cancel",key])
                    self.save("cancel_requested",experiment_id=key,run_id=entry["run_id"],reply=reply)
            except Exception as exc:
                self.save("cancellation_requires_attention",experiment_id=key,run_id=entry["run_id"],error=str(exc))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=DEFAULT_MANIFEST)
    parser.add_argument("--state",type=Path)
    parser.add_argument("--pair-audit-module",type=Path,default=Path(__file__).with_name("bounded_pair_audit.py"))
    parser.add_argument("--pair-audit-sha256")
    parser.add_argument("--capture-audit-module",type=Path,default=Path(__file__).with_name("bounded_capture_audit.py"))
    parser.add_argument("--capture-audit-sha256")
    parser.add_argument("--audit-python",type=Path,default=ROOT/".venv/bin/python")
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--watch",action="store_true");mode.add_argument("--once",action="store_true");mode.add_argument("--validate-only",action="store_true")
    args=parser.parse_args();manifest=json.loads(args.manifest.read_text());validate_manifest(manifest)
    if args.validate_only:
        print(json.dumps({"status":"STATIC_MANIFEST_VALID","model_or_launch_execution":False,"manifest_sha256":sha(args.manifest)}));return 0
    if not args.pair_audit_sha256 or not args.capture_audit_sha256:parser.error("Runtime requires frozen pair/capture-audit source SHA256 values")
    state=args.state or args.manifest.with_name("state.json")
    with exclusive_lock(state.with_suffix(".lock")):
        controller=Controller(args.manifest,state,args.pair_audit_module,args.pair_audit_sha256,
                              capture_module=args.capture_audit_module,capture_sha=args.capture_audit_sha256,audit_python=args.audit_python)
        def interrupted(signum,frame):raise InterruptedError(f"Controller interrupted by signal {signum}")
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        try:
            while True:
                if controller.tick() or args.once:return 0
                # A completion wake is only a signal; next tick reconciles all
                # six node statuses, including completions during retention.
                result=subprocess.run(["orx","exp","wait","--project",manifest["project_id"],"--interval","5","--timeout","45"],
                                      cwd=ROOT,capture_output=True,text=True,timeout=60)
                text=result.stdout+result.stderr
                if result.returncode and "timeout" not in text.lower() and "timed out" not in text.lower():
                    raise RuntimeError("ORX completion wait failed: "+text[-2000:])
        except BaseException as exc:
            controller.stop(exc)
            raise


if __name__=="__main__":
    raise SystemExit(main())
