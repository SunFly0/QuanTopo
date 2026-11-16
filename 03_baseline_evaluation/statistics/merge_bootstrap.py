"""Merge disjoint bootstrap shards into baseline-only confidence intervals."""
import argparse,json,hashlib,csv
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--shards',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    groups={}
    for f in sorted(a.shards.glob('*/metadata.json')):
        m=json.loads(f.read_text());assert m['status']=='completed',str(f)
        groups.setdefault(m['plan']['dataset'],[]).append((f.parent,m))
    results=[];sample_files={}
    for dataset,parts in sorted(groups.items()):
        first=parts[0][1];n=first['plan']['replicates'];methods=first['methods'];metrics=first['metrics'];values=np.empty((n,len(methods),len(metrics)));seen=np.zeros(n,dtype=bool)
        for folder,m in parts:
            assert m['plan_sha256']==first['plan_sha256'] and m['methods']==methods and m['metrics']==metrics
            assert m['provenance']==first['provenance'] and m['point_estimates']==first['point_estimates']
            file=folder/'bootstrap_samples.npz';assert hashlib.sha256(file.read_bytes()).hexdigest()==m['samples_sha256']
            with np.load(file) as d:
                ids=d['replicate_ids'];assert np.array_equal(ids,np.arange(m['start'],m['stop'])) and not seen[ids].any()
                values[ids]=d['values'];seen[ids]=True
        assert seen.all() and np.isfinite(values).all()
        np.savez_compressed(a.output/f'{dataset}_bootstrap_samples.npz',replicate_ids=np.arange(n),methods=np.asarray(methods),metrics=np.asarray(metrics),values=values)
        sample_files[dataset]=dict(filename=f'{dataset}_bootstrap_samples.npz',sha256=hashlib.sha256((a.output/f'{dataset}_bootstrap_samples.npz').read_bytes()).hexdigest(),plan_sha256=first['plan_sha256'])
        for mi,method in enumerate(methods):
            row=dict(dataset=dataset,method=method,repeats=3,bootstrap_replicates=n,
                interval_type='two-level percentile bootstrap, marginal for this method; not a QuanTopo paired difference',
                plan_sha256=first['plan_sha256'],metrics={})
            for ki,k in enumerate(metrics):
                lower,upper=np.quantile(values[:,mi,ki],[.025,.975]);point=first['point_estimates'][mi][ki]
                row['metrics'][k]=dict(point_estimate=point,ci95=[float(lower),float(upper)])
            results.append(row)
    report=dict(scope='baseline-only evaluation',seed=2026,results=results,sample_files=sample_files,
        outer_sampling='draw 3 repeat indices with replacement; equally average their metrics',
        inner_sampling='within each outer draw, independently resample original exact motion_id groups; all records/captions of the drawn ID stay together',
        fid='recompute weighted mean and unbiased covariance of generated and GT embeddings in float64 for every draw; symmetric PSD square root formulation',
        retrieval='bootstrap fixed-candidate per-record hits and matching distances; do not redefine retrieval candidates',
        pairing='shared outer and inner draws across available methods',
        interval='2.5/97.5 percentiles of the 10000 two-level draws')
    (a.output/'baseline_ci.json').write_text(json.dumps(report,indent=2))
    flat=[]
    for r in results:
        for metric,stats in r['metrics'].items():flat.append(dict(dataset=r['dataset'],method=r['method'],metric=metric,mean=stats['point_estimate'],ci95_lower=stats['ci95'][0],ci95_upper=stats['ci95'][1]))
    with (a.output/'baseline_ci.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print(json.dumps(results,indent=2),flush=True)
if __name__=='__main__':main()
