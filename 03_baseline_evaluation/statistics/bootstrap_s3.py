"""Two-level repeat/motion-ID bootstrap from frozen evaluator outputs."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch

METRICS=['fid','r1','r2','r3','matching']
METHODS={'humanml3d':['momask','motionhiflow','salad'],'kit-ml':['motionhiflow']}
ALGORITHM='repeat-motion-id-paired-v1'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def weighted_cov(x,w):
    n=w.sum(1)
    mean=w@x/n[:,None]
    cross=torch.matmul(x.T[None],x[None]*w[:,:,None])
    c=(cross-n[:,None,None]*mean[:,:,None]*mean[:,None,:])/(n-1)[:,None,None]
    return (c+c.transpose(1,2))*0.5,mean

def psd_sqrt(c):
    eig,vec=torch.linalg.eigh(c)
    if float(eig.min()) < -1e-8*max(1.,float(eig.abs().max())):
        raise ValueError('Covariance has a material negative eigenvalue')
    return (vec*eig.clamp(min=0).sqrt()[:,None,:])@vec.transpose(1,2)

def fid_given_reference(x,w,cg,mg,sg):
    c,m=weighted_cov(x,w)
    middle=sg@c@sg;middle=(middle+middle.transpose(1,2))*0.5
    eig=torch.linalg.eigvalsh(middle)
    if float(eig.min()) < -1e-8*max(1.,float(eig.abs().max())):
        raise ValueError('FID product has a material negative eigenvalue')
    f=((m-mg)**2).sum(1)+c.diagonal(dim1=-2,dim2=-1).sum(1)+cg.diagonal(dim1=-2,dim2=-1).sum(1)-2*eig.clamp(min=0).sqrt().sum(1)
    if float(f.min()) < -1e-6:raise ValueError('Negative FID')
    return f.clamp(min=0)

def cluster_weights(group_index,groups,seed,replicate,slot):
    # The stream depends only on the replicate and slot, never on a method/GPU/shard.
    rng=np.random.default_rng(np.random.SeedSequence([seed,int(replicate),int(slot),104729]))
    multiplicity=np.bincount(rng.integers(0,groups,size=groups),minlength=groups)
    return multiplicity[group_index].astype(np.float64)

def load_dataset(root,dataset,methods,device):
    repeats=[];provenance=[];point=[]
    for repeat in range(3):
        arrays=[];scores=[];metrics=[];source=[]
        shared_path=root.parent/'shared_embeddings'/f'{dataset}_repeat{repeat}.npz'
        with np.load(shared_path) as shared:
            ids=shared['record_ids'].astype(str);motion_ids=shared['motion_ids'].astype(str)
            gt=shared['reference'].copy();text=shared['text'].copy()
        assert np.isfinite(gt).all() and np.isfinite(text).all()
        for method in methods:
            run=root/f'{dataset}_{method}_repeat{repeat}'
            ep=run/'evaluation/motion_embeddings.npz';pp=run/'evaluation/per_record.jsonl';mp=run/'evaluation/metrics.json'
            with np.load(ep) as e:
                assert np.isfinite(e['motion']).all()
                arrays.append(torch.tensor(e['motion'],dtype=torch.float64,device=device))
            rows=[json.loads(l) for l in pp.read_text().splitlines()]
            assert [r['record_id'] for r in rows]==ids.tolist()
            assert [r['motion_id'] for r in rows]==motion_ids.tolist()
            scores.append(torch.tensor([[r[k] for k in METRICS[1:]] for r in rows],device=device,dtype=torch.float64))
            mm=json.loads(mp.read_text());metrics.append([mm[k] for k in METRICS])
            source.append(dict(method=method,run=run.name,motion_embeddings_sha256=sha(ep),
                shared_embeddings_sha256=sha(shared_path),per_record_sha256=sha(pp),
                records_sha256=mm['records_sha256'],records=len(ids)))
        _,group_index=np.unique(motion_ids,return_inverse=True)
        group_index=group_index.astype(np.int64);groups=int(group_index.max())+1
        reference=torch.tensor(gt,device=device,dtype=torch.float64)
        one=torch.ones((1,len(ids)),device=device,dtype=torch.float64)
        c,m=weighted_cov(reference,one);s=psd_sqrt(c)
        # Validate the stable symmetric FID calculation against original SciPy results.
        for i,method in enumerate(methods):
            actual=float(fid_given_reference(arrays[i],one,c,m,s)[0])
            assert abs(actual-metrics[i][0])<2e-6,(method,actual,metrics[i][0])
        repeats.append(dict(reference=reference,motions=arrays,scores=scores,group_index=group_index,groups=groups))
        provenance.append(dict(repeat=repeat,groups=groups,records=len(ids),sources=source,
            record_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
            motion_ids_sha256=hashlib.sha256('\n'.join(motion_ids).encode()).hexdigest(),
            reference_sha256=hashlib.sha256(gt.tobytes()).hexdigest()))
        point.append(metrics)
    return repeats,provenance,np.asarray(point).mean(0)

def run(args,dataset,methods):
    device=f'cuda:{args.gpu}';repeats,provenance,point=load_dataset(args.runs,dataset,methods,device)
    plan=dict(algorithm=ALGORITHM,dataset=dataset,seed=args.seed,replicates=args.replicates,
        grouping='exact motion_id, all associated records sampled together',outer_draws=3,
        repeat_record_hashes=[p['sources'][0]['records_sha256'] for p in provenance])
    planhash=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    output=args.output/f'{dataset}_{args.start:05d}_{args.stop:05d}';output.mkdir(parents=True,exist_ok=True)
    result=np.empty((args.stop-args.start,len(methods),len(METRICS)),dtype=np.float64)
    outer=np.random.default_rng(args.seed).integers(0,3,size=(args.replicates,3))
    started=time.monotonic()
    meta=dict(plan=plan,plan_sha256=planhash,start=args.start,stop=args.stop,methods=methods,metrics=METRICS,
        provenance=provenance,point_estimates=point.tolist(),dtype='float64',gpu=args.gpu,status='running')
    (output/'metadata.json').write_text(json.dumps(meta,indent=2))
    for begin in range(args.start,args.stop,args.chunk):
        end=min(begin+args.chunk,args.stop);values=np.empty((end-begin,3,len(methods),len(METRICS)))
        for r in range(3):
            ij=np.argwhere(outer[begin:end]==r);data=repeats[r]
            for k in range(0,len(ij),args.batch):
                where=ij[k:k+args.batch]
                w=np.stack([cluster_weights(data['group_index'],data['groups'],args.seed,begin+i,s) for i,s in where])
                w=torch.tensor(w,device=device,dtype=torch.float64);n=w.sum(1)
                c,m=weighted_cov(data['reference'],w);sqrt=psd_sqrt(c)
                for mi,x in enumerate(data['motions']):
                    f=fid_given_reference(x,w,c,m,sqrt)
                    stats=(w@data['scores'][mi])/n[:,None]
                    combined=torch.cat((f[:,None],stats),1).cpu().numpy()
                    values[where[:,0],where[:,1],mi]=combined
        result[begin-args.start:end-args.start]=values.mean(1)
        if (end-args.start)%256==0 or end==args.stop:
            elapsed=time.monotonic()-started
            progress=dict(dataset=dataset,start=args.start,stop=args.stop,completed=end-args.start,elapsed_seconds=elapsed)
            (output/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
    assert np.isfinite(result).all()
    np.savez_compressed(output/'bootstrap_samples.npz',replicate_ids=np.arange(args.start,args.stop),values=result)
    meta.update(status='completed',elapsed_seconds=time.monotonic()-started,samples_sha256=sha(output/'bootstrap_samples.npz'))
    (output/'metadata.json').write_text(json.dumps(meta,indent=2));print('COMPLETE',output,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,default=Path(__file__).resolve().parents[1]/'runs')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'shards')
    p.add_argument('--datasets',nargs='+',choices=list(METHODS),default=list(METHODS))
    p.add_argument('--method-config',type=Path,help='Optional JSON mapping dataset names to methods; each method needs three complete run directories')
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--replicates',type=int,default=10000)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=10000)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--chunk',type=int,default=32)
    args=p.parse_args();assert 0<=args.start<args.stop<=args.replicates
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    mapping=json.loads(args.method_config.read_text()) if args.method_config else METHODS
    for dataset in args.datasets:run(args,dataset,mapping[dataset])
if __name__=='__main__':main()
