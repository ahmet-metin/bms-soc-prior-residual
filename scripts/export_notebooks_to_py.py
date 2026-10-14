#!/usr/bin/env python3
"""Generate faithful, output-free, Colab-dependent .py mirrors of frozen notebook code.

Do not confuse notebook exports with independently portable experiment pipelines.
The supported standalone entry points live under soc_repro/.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "PY_EXPORT_MANIFEST.json"


def source_string(value):
    return "".join(value) if isinstance(value, list) else str(value or "")


def render(notebook: Path, rel: Path):
    raw = notebook.read_bytes()
    nb = json.loads(raw)
    source_hash = hashlib.sha256(raw).hexdigest()
    header = [
        "# -*- coding: utf-8 -*-",
        "# Extracted research code; source notebook remains authoritative.",
        "# Source: " + rel.as_posix(),
        "# Notebook SHA256: " + source_hash,
        "# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.",
        "# Google Drive mounts, canonical inputs, external data permissions and GPUs",
        "# are still needed wherever the source experiment requires them.",
        "# Leading !pip / % notebook magics are commented, not silently executed.",
        "# Use original .ipynb in Colab to reproduce the original runtime setup.",
        "",
    ]
    commands = []
    ncode = 0
    for idx, cell in enumerate(nb.get("cells", []), 1):
        ctype = cell.get("cell_type")
        src = source_string(cell.get("source"))
        if ctype == "markdown":
            title = next((l.lstrip("# ").strip() for l in src.splitlines() if l.strip().startswith("#")), None)
            if title:
                header += ["# %% [markdown] " + title.replace("\n", " ")[:140], ""]
            continue
        if ctype != "code":
            continue
        ncode += 1
        header.append(f"# %% Code cell {ncode} (notebook cell {idx})")
        for line in src.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("!", "%")) and not stripped.startswith("%% =="):
                # Comment Colab shell and line magics rather than triggering package
                # installation or hidden network calls when .py is inspected.
                commands.append(stripped)
                header.append(line[:len(line)-len(stripped)] + "# [NOTEBOOK MAGIC, MANUALLY SET UP] " + stripped)
            else:
                header.append(line)
        header.append("")
    txt = "\n".join(header).rstrip() + "\n"
    compile(txt, str(rel), "exec")
    return txt, source_hash, ncode, commands


def collect():
    entries = []
    candidates = sorted((ROOT / "legacy_sources" / "notebooks").rglob("*.ipynb"))
    candidates += sorted((ROOT / "notebooks" / "updated_colab").rglob("*.ipynb"))
    for nb in candidates:
        rel = nb.relative_to(ROOT)
        if rel.parts[0] == "legacy_sources":
            output = (Path("experiments") / "notebook_exports" / Path(*rel.parts[3:])).with_suffix(".py")
        else:
            output = (Path("experiments") / "notebook_exports" / "updated_colab" / nb.name).with_suffix(".py")
        text, source_sha, ncode, commands = render(nb, rel)
        entries.append({
            "source_notebook": rel.as_posix(),
            "source_notebook_sha256": source_sha,
            "python_export": output.as_posix(),
            "python_export_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "code_cells": ncode,
            "commented_colab_commands": commands,
            "execution_status": "notebook_source_export_only; not end-to-end standalone verified",
            "_text": text,
        })
    return entries


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="Verify exported source code and manifest without changes")
    args = p.parse_args(argv)
    entries = collect()
    if len(entries) != 30:
        raise RuntimeError(f"Expected 28 archived + 2 updated notebooks, found {len(entries)}")
    manifest = {
        "purpose": "Readable Python exports of Colab notebooks; original notebook sources remain authoritative",
        "python_executable_primary_package": "soc_repro/",
        "independent_execution_verified": False,
        "count": len(entries),
        "files": [{k:v for k,v in e.items() if k != "_text"} for e in entries],
    }
    if args.check:
        if not MANIFEST.exists() or json.loads(MANIFEST.read_text(encoding="utf-8")) != manifest:
            raise SystemExit("ERROR: PY_EXPORT_MANIFEST.json drifted from source notebooks")
        for e in entries:
            path = ROOT / e["python_export"]
            if not path.is_file() or path.read_text(encoding="utf-8") != e["_text"]:
                raise SystemExit(f"ERROR: Python export out of sync: {path}")
        print(f"OK: {len(entries)} Python exports exactly match notebook code sources")
        return
    for e in entries:
        path = ROOT / e["python_export"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(e["_text"], encoding="utf-8")
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(entries)} .py notebook mirrors (NOT independent executable pipelines)")

if __name__ == "__main__":
    main()
