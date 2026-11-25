# Verified duplicate staging cleanup: 15 October 2026

Removed only redundant ignored local staging copies. Paper A manuscript sources and published outputs were untouched. All canonical evidence remains under `build/hf-release-2026-09-17/`. Every removed file was SHA-256 identical to its retained canonical counterpart, with stable source/destination stat records, Git-tracked-file protection and symlink refusal.

| Removed ignored staging group | Files | Logical bytes | Allocated bytes |
| --- | ---: | ---: | ---: |
| `build/release-staging/paper-b/agentic-live-v1` | 16 | 69,671 | 102,400 |
| `build/release-staging/paper-b/clustered-gate-v1` | 2 | 45,051 | 49,152 |
| `build/release-staging/paper-b/domain-shift-v1` | 6,291 | 5,978,202,950 | 5,995,593,728 |
| `build/release-staging/paper-b/long-horizon-v1` | 10,141 | 300,676,890 | 332,648,448 |
| `build/release-staging/paper-b/positive-controls` | 1,609 | 1,750,306,511 | 1,755,451,392 |
| `build/release-staging/paper-b/protocols` | 2 | 16,136 | 20,480 |
| `build/release-staging/paper-a/v3-campaign` | 7,325 | 7,163,908,559 | 7,178,248,192 |
| `build/release-staging/paper-b/v3-campaign` | 7,325 | 7,163,908,559 | 7,178,248,192 |

Combined total: 32,711 files, 22,357,134,327 logical bytes and 22,440,361,984 allocated file bytes; 0 errors. Allocated sizes use `stat.st_blocks`; actual volume free space can differ when the filesystem shares extents.

Each `paper-{a,b}/v3-campaign` staging tree retains its unique historical `evidence-release-v3.md` and `review/verdict.md`. The audit records their hashes. Source shard directories named `validation-heldout-v3-r3-shardN` map to retained canonical directories named `shardN`; reconstruction uses the exact per-file map. Other nonidentical/unmapped staging files were retained.

The compressed [per-file audit](repository-deduplication-20261015.json.gz) incorporates the original six-subtree audit unchanged, records its previous compressed-file digest and includes every additional source/canonical mapping, SHA-256, size, inode/device/mode/link count, timestamp and removal status.

Restore every removed staging file from the repository root. Existing files must already match their recorded hash; different files are never overwritten:

```sh
python3 - <<'PY'
import gzip, hashlib, json, shutil
from pathlib import Path
audit = json.loads(gzip.decompress(Path('docs/repository-deduplication-20261015.json.gz').read_bytes()))
def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()
for item in audit['files']:
    if item['status'] != 'removed':
        continue
    source, target = Path(item['canonical']), Path(item['source'])
    assert sha(source) == item['sha256'], source
    if target.exists():
        assert sha(target) == item['sha256'], target
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert sha(target) == item['sha256'], target
PY
```
