"""No-false-positive R2D/R3A benchmark entrypoint.

The source notebooks are included under legacy_sources. The full *training*
implementation still depends on frozen R2C/R1A checkpoints and validation
artifacts. Rather than silently reusing those as fresh training, this command
fails closed, with source paths and execution contract spelled out.
"""
import argparse,sys
from pathlib import Path
from .experiments import entries

def main(argv=None):
    p=argparse.ArgumentParser(description='Inspect R2D/R3A benchmark implementation readiness (never invents trained results)')
    p.add_argument('--status',action='store_true',help='Show source and prerequisites; default also only status')
    args=p.parse_args(argv)
    for r in entries():
        if any(x in r['component'] for x in ('benchmark','ablation')):
            print(f"{r['component']}: {r['source']} [{r['verification']}]")
    print('NOT YET PORTED: R2D/R3A independent benchmark training still runs through archived original notebooks.')
    print('No new trained metrics or "PASS" issued; see docs/RELEASE_GATES.md.')
    return 2
if __name__=='__main__':sys.exit(main())
