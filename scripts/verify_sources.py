"""Detect missing or modified archived notebook source files. No GPU or data required."""
from pathlib import Path
import hashlib,json
base=Path(__file__).resolve().parent.parent
registry=json.loads((base/'docs'/'SOURCE_REGISTRY.json').read_text())
assert registry, 'Empty source registry'
for r in registry:
    p=base/r['path']
    assert p.is_file(),f'Missing source: {r["path"]}'
    got=hashlib.sha256(p.read_bytes()).hexdigest()
    assert got==r['sha256'],f'Changed source: {r["path"]}'
    import json as _j
    doc=_j.loads(p.read_text())
    n=sum(c['cell_type']=='code' for c in doc['cells'])
    assert n==r['code_cells'],f'Code cell count changed: {r["path"]}'
print(f'PASS: {len(registry)} legacy creator notebooks; SHA256 and code cell counts validated. This is NOT a training rerun.')
