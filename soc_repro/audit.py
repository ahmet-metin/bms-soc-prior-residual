"""Read-only progress / integrity checks, independent of GPU training."""
from pathlib import Path
from datetime import datetime, timezone
import json,hashlib,os
DOMAINS=['VED','SCUT','TUM','KU','EVE']
FRACTIONS=[0.01,0.02,0.05,0.10,0.20,0.50,1.00]
SEEDS=[42,43,44,45,46]
MODELS=['Direct+Prior','PriorResidual+Prior']

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def scan_run_root(run_root, strict_hash=True):
    root=Path(run_root)
    details={};errors=[];total=0;pending=[]
    for domain in DOMAINS:
        records=root/domain/'run_records'
        done=0
        for fraction in FRACTIONS:
            for seed in SEEDS:
                for model in MODELS:
                    tag=f'{domain}_fraction{fraction}_{model}_seed{seed}'.replace('+','Plus')
                    p=records/(tag+'.json')
                    if not p.exists():
                        pending.append({'domain':domain,'fraction':fraction,'seed':seed,'model':model})
                        continue
                    try:
                        r=json.loads(p.read_text())
                        if (r['domain'],float(r['fraction']),int(r['seed']),r['model'])!=(domain,fraction,seed,model):
                            raise ValueError('receipt identity mismatch')
                        if r.get('clean_room_run') is not True or r.get('reused_prior_endpoint') is not False:
                            raise ValueError('clean-room provenance flags invalid')
                        for item,field in [('checkpoint','checkpoint_sha256'),('prediction_file','prediction_sha256')]:
                            candidate=Path(r[item]).resolve()
                            if not candidate.is_relative_to((root/domain).resolve()) or not candidate.is_file():
                                raise ValueError(f'file missing or external: {item}')
                            if strict_hash and sha256_file(candidate)!=r[field]:
                                raise ValueError(f'hash mismatch {item}')
                        done+=1
                    except Exception as e:
                        errors.append({'receipt':str(p),'error':str(e)})
        details[domain]={'verified_models':done,'expected_models':70}
        total+=done
    return {'verified_models':total,'expected_models':350,'per_domain':details,
            'pending':pending,'errors':errors,'fresh_all_350_verified':total==350 and not errors,
            'github_ready':False,
            'qualification':'Model training uses frozen canonical inputs; full GitHub release requires additional figure/statistics/data-rights audits.'}

def write_progress(run_root, strict_hash=False):
    root=Path(run_root)
    result=scan_run_root(root,strict_hash=strict_hash)
    summary={k:v for k,v in result.items() if k!='pending'}
    summary['created_utc']=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    summary['percent']=round(100*summary['verified_models']/350,3)
    summary['completed_models']=summary['verified_models']  # Backwards-compatible with the earlier 03 notebook.
    summary['per_domain']={d:{**v,'completed_receipts':v['verified_models'],'expected':v['expected_models']} for d,v in summary['per_domain'].items()}
    summary['fresh_all_350_verified']=bool(strict_hash and result['fresh_all_350_verified'])
    out=root/'03_progress.json';out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_name(out.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(summary,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,out)
    return summary
