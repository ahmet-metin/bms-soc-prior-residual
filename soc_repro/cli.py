"""Automatic one-command restart-safe scheduler for the existing 350-run R2C pipeline."""
import argparse,time,os,json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description='Reproduce 350 R2C models without manual domain edits')
    p.add_argument('--root',default=os.getenv('SOC_DATA_ROOT'),help='Root containing frozen canonical data and manifests (required for training)')
    p.add_argument('--run-root',default=None,help='Override output root; default reuses 03_R2C_FULL350_FRESH_V1, keeping existing completed work')
    p.add_argument('--max-hours',type=float,default=7.0,help='Stop at next pair boundary after this many wall-clock hours')
    p.add_argument('--max-pairs',type=int,default=175,help='Upper bound of NEW model pairs this invocation')
    p.add_argument('--status',action='store_true',help='Inspect progress without GPU or training')
    p.add_argument('--verify-hashes',action='store_true',help='During status, re-hash every existing prediction and best checkpoint')
    args=p.parse_args(argv)
    if not args.root and not args.status:
        p.error('Specify --root /path/to/dataset_root or SOC_DATA_ROOT.')
    root=Path(args.root).resolve() if args.root else Path.cwd()
    run_root=Path(args.run_root).resolve() if args.run_root else root/'reproduction_runs'/'R2C_350'
    from .audit import scan_run_root,write_progress,DOMAINS
    if args.status:
        report=scan_run_root(run_root,strict_hash=args.verify_hashes)
        print(json.dumps({k:v for k,v in report.items() if k!='pending'},indent=2))
        if report['errors']:
            raise SystemExit(2)
        return
    if args.max_hours<=0 or args.max_pairs<=0:
        raise SystemExit('max-hours and max-pairs must be positive')
    from . import r2c_engine as engine
    if engine.DEVICE.type!='cuda':raise SystemExit('Select a CUDA GPU runtime in Colab.')
    deadline=time.monotonic()+args.max_hours*3600
    count=0
    print('Resume root:',run_root,flush=True)
    print('GPU:',engine.torch.cuda.get_device_name(0),flush=True)
    for domain in DOMAINS:
        if count>=args.max_pairs or time.monotonic()>=deadline:break
        info=scan_run_root(run_root,strict_hash=False)
        if info['errors']:raise RuntimeError('Invalid previous receipt, stop: '+str(info['errors'][:2]))
        if info['per_domain'][domain]['verified_models']==70:
            print(f'SKIP {domain}: 70/70 models completed',flush=True)
            continue
        engine.prepare_domain(root,domain,run_root=run_root)
        n=engine.run_domain(args.max_pairs-count,deadline,on_pair=write_progress)
        count+=n
        engine.audit_domain_results()
        write_progress(run_root)
    final=scan_run_root(run_root,strict_hash=True)
    write_progress(run_root,strict_hash=True)
    print('VERIFIED:',final['verified_models'],'/350, new pairs this invocation:',count,
          'remaining models:',350-final['verified_models'],flush=True)
    if final['errors']:raise RuntimeError('Verification errors: '+str(final['errors'][:3]))
    if final['verified_models']==350:
        print('R2C training complete. Manuscript figures, R7 postanchor and statistical verification are separate required gates.',flush=True)

if __name__=='__main__':main()
