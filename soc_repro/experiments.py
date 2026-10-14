"""Manuscript-to-code registry with honest independent-rerun readiness statuses."""
import argparse,json,csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REGISTRY=ROOT/'docs'/'MANUSCRIPT_CODE_REGISTRY.json'

def entries():
    rows=json.loads(REGISTRY.read_text(encoding='utf8'))
    for r in rows:
        if not (ROOT/r['source']).is_file():
            raise FileNotFoundError(f"Missing source for manuscript component {r['component']}: {r['source']}")
    return rows

def main(argv=None):
    p=argparse.ArgumentParser(description='Identify actual sources and verified execution status for every REV8 table/figure')
    p.add_argument('--component',help='Optional exact LaTex label');p.add_argument('--check',action='store_true')
    args=p.parse_args(argv)
    rows=entries()
    if args.component:rows=[r for r in rows if r['component']==args.component]
    if not rows:raise SystemExit('No matching component in manuscript registry')
    for r in rows:
        print(f"{r['component']:24} | {r['implementation']:28} | {r['verification']:24} | {r['source']}")
    if args.check:
        unverified=[r['component'] for r in rows if r['verification']!='independently_reexecuted']
        if unverified:
            print('INCOMPLETE: these outputs have not been independently rerun:',', '.join(unverified))
            return 2
    return 0
if __name__=='__main__':raise SystemExit(main())
