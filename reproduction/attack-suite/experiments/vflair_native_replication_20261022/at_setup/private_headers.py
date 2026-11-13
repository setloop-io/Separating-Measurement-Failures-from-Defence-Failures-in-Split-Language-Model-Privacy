"""Verify and extract one official Ubuntu header package without installing it."""

import hashlib
import json
import os
from pathlib import Path
import urllib.request


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_download(spec, target, opener=urllib.request.urlopen):
    """The reviewed Ubuntu URL, exact size and SHA are all mandatory."""
    target = Path(target)
    if not spec["url"].startswith("https://ports.ubuntu.com/ubuntu-ports/pool/main/p/python3.12/"):
        raise ValueError("Python headers must use the pinned official Ubuntu package URL")
    if target.is_file() and target.stat().st_size == spec["bytes"] and sha(target) == spec["sha256"]:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    digest, total = hashlib.sha256(), 0
    try:
        with opener(spec["url"], timeout=120) as response, temporary.open("wb") as output:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                total += len(block)
                if total > spec["bytes"]:
                    raise ValueError("Ubuntu header download exceeds its pinned size")
                digest.update(block)
                output.write(block)
        if total != spec["bytes"] or digest.hexdigest() != spec["sha256"]:
            raise ValueError("Ubuntu header package size/SHA256 mismatch")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def header_environment(spec, root, environment):
    root = Path(root).resolve()
    headers = {}
    for relative, expected in spec["required_headers"].items():
        path = root / relative
        if not path.is_file() or sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"Task-private Python header differs: {relative}")
        headers[relative] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    # Python.h finds the generic pyconfig.h in its own directory. That wrapper
    # includes <aarch64-linux-gnu/python3.12/pyconfig.h>, requiring usr/include too.
    includes = [str(root / relative) for relative in spec["include_directories"]]
    env = dict(environment)
    # No inherited compiler header path may silently override the pinned package.
    for name in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH"):
        env.pop(name, None)
    env["CPATH"] = os.pathsep.join(includes)
    return env, headers


def prepare(spec, private, out, environment, run_logged, probe):
    """Only dpkg-deb's unprivileged extraction subcommand is permitted."""
    private, out = Path(private), Path(out)
    facts = probe(["/usr/bin/dpkg-query", "-W", "-f=${Version}", "python3.12-minimal"])
    if facts.strip() != spec["version"]:
        raise RuntimeError("Task-private Python headers do not match the installed Python package version")
    package_root = private / "software-prerequisites" / spec["sha256"]
    package_root.mkdir(parents=True, exist_ok=True)
    package = package_root / Path(spec["url"]).name
    downloaded = checked_download(spec, package)
    root = package_root / "extracted"
    stamp = package_root / "verified-headers.json"
    if not stamp.is_file():
        # Repeating extraction after an interrupted attempt is safe: all bytes
        # originate from the same verified package; dpkg never installs it.
        root.mkdir(parents=True, exist_ok=True)
        run_logged(["/usr/bin/dpkg-deb", "--extract", str(package), str(root)],
                   out / "python-header-extract.log", environment, 120)
    env, headers = header_environment(spec, root, environment)
    # A C preprocessor check exercises both the generic and multiarch includes.
    fixture = out / "python-header-smoke.c"
    fixture.write_text('#include <Python.h>\n#if PY_MAJOR_VERSION != 3 || PY_MINOR_VERSION != 12\n#error Wrong Python header ABI\n#endif\n')
    run_logged(["/usr/bin/cc", "-E", "-o", str(out / "python-header-smoke.i"), str(fixture)],
               out / "python-header-smoke.log", env, 60)
    receipt = {"schema": "original-at-private-python-headers-v1", "package": spec["package"],
               "version": spec["version"], "architecture": spec["architecture"], "url": spec["url"],
               "sha256": sha(package), "bytes": package.stat().st_size, "downloaded_this_run": downloaded,
               "extraction_root": str(root), "CPATH": env["CPATH"], "required_headers": headers,
               "matching_installed_python_package": facts.strip(), "preprocessor_check_passed": True,
               "apt_install_or_system_write_performed": False}
    stamp.write_text(json.dumps(receipt, indent=2) + "\n")
    (out / "python-header-admission.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return env, receipt
