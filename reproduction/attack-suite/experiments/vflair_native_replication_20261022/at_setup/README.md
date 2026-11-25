# Original AT software and asset admission

This performs software and asset admission only. It cannot start a native model, training, generation or attack. Its first recorded software attempt (`2a8327a1-dd19-4999-9ca7-28f73f029a85`) stopped while compiling the grammar because the host lacked Python development headers. The prepared repair below has not yet passed a server run. Its runtime entry requires an `ORX_RUN_ID`; the local `--validate-only` path performs only standard-library source/config checks.

The fixed command for the separate AT baseline is:

```sh
bash experiments/vflair_native_replication_20261022/at_setup/entry.sh
```

Keep that command unchanged across its children. Freeze `run.json` on each child to select its stage. This baseline has its own command without retargeting MID's existing command. The parent owns registration, source freezing and launches.

## Ordered admission stages

| Stage in committed `run.json` | Work and acceptance |
|---|---|
| `dependencies` | Create the separately owned AT venv; inherit the exact base CUDA Torch; install/record the pinned private stack; reject any pip plan that installs Torch/torchvision/torchaudio/triton; pass pip check and the 19 CPU metric checks. No model/data download. |
| `dataset-assets` | Require the matching successful dependency receipt first. Download/verify the original-schema CodeAlpaca JSON (~8 MB), retaining exact bytes/order/native fields. No split is made. |
| `model-assets` | Recheck the dependency receipt, exact installed versions/platform and inherited Torch; **offline-verify the prepared JSON before any model download**. Require 60 GiB free for asset preparation, then download the exact 12-file, 13,479,516,728-byte model allowlist. |
| `assets-verify` | Offline rehash/revalidate both prepared asset packages and emit paths/manifests for the separate native canary. No network, model load or forward pass. |

An invalid parser/metric environment stops in the first stage. Missing/corrupt data stops before the 13.5 GB model download. An asset-stage run cannot manufacture its own successful dependency receipt. The receipt binds source/configuration, requirements, package versions, Torch identity, compiled platform and the actual metric-gate output; changing the environment or pinned source requires another recorded dependency admission.

`at_setup/vendor/` contains the already reviewed 254 native implementation files and both independently Git-verified config blobs, plus the standard-library input preparer. It includes no weights or dataset records. `inputs/` contains the retained official metadata/tree/guide/provenance inputs. The dependency stage imports Torch only to identify its inherited library path/version/file hash; CUDA devices are hidden and no tensor/model kernels run. The metric check builds a tiny tokenizer fixture, not a pretrained tokenizer or model.

## Separate environment and immutable pins

* Base Python/Torch, read only: `/home/geo/llm-attacker/.venv/bin/python`.
* New AT environment: `/home/geo/.venvs/vflair-native-at-20261022`.
* The MID environment `/home/geo/.venvs/vflair-native-20261022` is explicitly rejected as an AT target. Existing unowned AT directories are also rejected.
* Host CUDA Torch is inherited with a private `.pth` path and an exact-version constraint. Its path/version/init-file hash must match before and after setup; pip dry-run reports must contain no CUDA package replacement. The bootstrap never installs upstream `requirements.txt`.
* Metric pins: CodeBLEU 0.7.0; tree-sitter 0.22.3; grammar 0.21.0. Build tools: setuptools 80.9.0; wheel 0.45.1. ARM grammar build uses official commit `0f9047c857ed0990931b1f899c7d3bf403703147`, archive SHA-256 `62cf59b7013e8d1c7e60aa181fe3216f54df18145d218699061f84376b17170c`, with build isolation disabled after installing those build tools.
* The private native stack keeps Transformers 4.39.0, tokenizers 0.15.2, PEFT 0.10.0 and Accelerate 0.28.0, with the other direct versions in `requirements-native.txt`. Actual transitive resolution and official download hashes are retained in pip reports; the full installed-version snapshot is then frozen in the admission receipt.
* The header repair downloads official Ubuntu Noble ARM64 `libpython3.12-dev=3.12.3-1ubuntu0.17` (5,538,952 bytes; SHA-256 `945ad3f651f683c11e72d19d804b1cff097308dcffa01057635cba038f875164`) into this AT venv's `software-prerequisites` directory. It checks the installed interpreter's matching package version, uses only `dpkg-deb --extract`, and never calls apt, sudo or a package-install hook. A C-preprocessor smoke check exercises `Python.h` and the multiarch configuration header before any grammar build. `CPATH` contains the extracted `usr/include/python3.12`, `usr/include/aarch64-linux-gnu/python3.12` and `usr/include`; the third is needed by Ubuntu's multiarch dispatch wrapper.
* Read-only preflight found inherited PyNaCl missing cffi and nvitop requiring an older NVML binding, plus a system SciPy/base NumPy conflict. Exact official PyPI wheels for private cffi 1.17.1, pycparser 2.22 and nvidia-ml-py 12.535.161 resolve the first two; the native NumPy/SciPy pins address the third. These are private Python packages, with no CUDA library or base/MID replacement. Final private `pip check` must pass without suppressed diagnostics. Base package versions and its pre-existing pip diagnostic are recorded before and after and must remain unchanged.
* Only the exact unadmitted owner contract `6e2732f206cfe1a158439fd226cb553810b027b96b0d975f17ec0ae6c2f583d5` from the named failed first software run may transition to this repair. The old owner and replacement contract are retained in a repair receipt; an admitted environment or a different owner cannot be silently repaired.

The earlier Mac temporary-environment gate passed 19 checks. The new portable gate is prepared from those checks; its **Linux ARM environment run remains pending**. The static preparation here does not claim those server dependencies or imports have passed.

## Exact original-task assets and later executor interface

Model: `codellama/CodeLlama-7b-hf`, revision `6c284d1468fe6c413cf56183e69b194dcfa27fe6`. Asset root is `/home/geo/.cache/llm-attack-suite/vflair-native-at-20261022/CodeLlama-7b-hf`; its manifest is `manifest.json`. Only JSON/config/tokenizer/licence/readme assets and the two safetensors shards are selected. No `.bin` weights are downloaded. Files are streamed into atomic temporaries and verified by LFS SHA-256 or non-LFS Git blob identity before final naming.

Data: `sahil2801/CodeAlpaca-20k`, revision `152bb5e9a29651266b018106053980070a0521a1`. The raw JSON is stored at `/home/geo/.cache/llm-attack-suite/vflair-native-at-20261022/CodeAlpaca-20k/code_alpaca_20k.json`, with `manifest.json` beside it. The official guide points to H4; its pinned creation script loads this original source, then strips/flattens fields and applies its own split. The original-schema precursor is therefore an explicit source choice, preserving `instruction`/`input`/`output` and leaving the native executor's seed-dependent 90/10 split intact. This is not a claim about the original paper's unspecified data revision.

After all admission stages, the emitted `asset-admission.json.next_executor_host` supplies exact paths to `codealpaca_native.check_inputs`. A separately recorded native child must use `import_compat_llama.prepare` on a fresh source copy before importing the native pipeline, then `codealpaca_native.execute`. Its original protocol is frozen backbone slices, two local head layers, sequence 256, batch 8, 50-epoch nominal limit with native stopping, VMI/WBI 100/500 iterations and 100 test targets per configured entry. The two default/explicit-seed entries per inversion class are retained. BiSR still needs its separate auxiliary-training admission; omission is an inversion-subset result.

The later canary is one native first-batch epoch, eight training rows after the native split, two test rows and one target with two attack iterations per inversion class. It is excluded from benchmark claims. The full generation/CodeBLEU path, frozen trainability, actual AT optimiser updates, native Llama cache APIs and resource limits remain canary gates. Asset setup's 60 GiB guard does not certify enough space or memory for full training/checkpoints/captures.

## Native utility and optional corrected metric

The native score remains `preserve-upstream-mapping-callback-v1`: the upstream loop makes ten Python-language calls per document, and its whole-HF-tokenizer callback supplies mapping field names to the lexical components. The prepared software gate verifies that convention rather than silently correcting it. It is labelled when interpreting native utility.

`rescore_saved.py` is an optional **separate** metric, `codebleu-default-whitespace.v1-from-saved-generations`. It validates the native observer's saved generated/target IDs, exact decoded scoring texts, all ten per-document calls and native arithmetic, then can recompute CodeBLEU using its documented default whitespace callback. It never regenerates predictions and never replaces the native number. Actual rescoring requires a separately recorded ORX postprocessing node; setup does not execute it. Utility thresholds are not selected by this helper.

## Current local checks

The source/configuration validation and 22 standard-library contract fixtures passed locally. The added fixtures check exact owner recovery, atomic header hash failure, rehashing cached downloads, official-source restrictions, multiarch include paths, early package-version rejection and wheel identities. Read-only SSH inspected host/package metadata, and an official header package plus PyPI metadata were downloaded locally for verification. No package was installed or extracted on the server by this repair preparation; no model or remote workload was launched by it. The historical `PLAN.json` records the first handoff; `at-admission/header-repair-20261022/` records this repair and its updated source hashes. Linux ARM metric admission remains a required actual server gate.
