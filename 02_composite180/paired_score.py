import argparse,csv,json
from pathlib import Path
import numpy as np
from score import compute

def paired(root):
    _,majority=compute(root)
    dag=json.loads((root/'events_dag.json').read_text());cases=dag['records'];protocol=dag['protocol']
    by={(r['caption_id'],r['method']):r for r in majority['records']}
    groups=protocol['bootstrap_group_order'];bins=[[i for i,c in enumerate(cases) if c['source_group']==g] for g in groups]
    sizes=np.array([len(ix) for ix in bins]);G=len(groups);draws=np.random.default_rng(protocol['bootstrap_seed']).integers(0,G,(protocol['bootstrap_replicates'],G));den=sizes[draws].sum(1)
    results={}
    for method in ['momask','motionhiflow','actionplan','salad']:
        a=np.array([[by[c['caption_id'],method]['seq'],int(by[c['caption_id'],method]['fully_covered'])] for c in cases]);b=np.array([[by[c['caption_id'],'quantopo']['seq'],int(by[c['caption_id'],'quantopo']['fully_covered'])] for c in cases])
        diff=b[:,0]-a[:,0];both=(a[:,1]>0)&(b[:,1]>0);values=np.stack([diff,b[:,1]-a[:,1],diff*both,both],1);totals=values.sum(0)
        grouped=np.stack([values[ix].sum(0) for ix in bins]);sample=grouped[draws].sum(1);boot=np.stack([sample[:,0]/den,sample[:,1]/den,sample[:,2]/sample[:,3]],1);ci=np.quantile(boot,[.025,.975],axis=0).T.tolist()
        results[method]={'comparison':'quantopo_minus_'+method,'caption_count':len(cases),'delta_seq':float(totals[0]/len(cases)),'delta_seq_ci':ci[0],'delta_coverage':float(totals[1]/len(cases)),'delta_coverage_ci':ci[1],'joint_coverage':int(totals[3]),'delta_order_given_joint_coverage':float(totals[2]/totals[3]),'delta_order_ci':ci[2]}
    return {'bootstrap_replicates':protocol['bootstrap_replicates'],'bootstrap_seed':protocol['bootstrap_seed'],'source_groups':G,'comparisons':results}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path,nargs='?',default=Path(__file__).resolve().parent);root=p.parse_args().directory;out=paired(root);(root/'results').mkdir(exist_ok=True);(root/'results/paired_comparisons.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
