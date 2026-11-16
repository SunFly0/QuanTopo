import argparse,csv,json,re
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

METHODS=['momask','motionhiflow','salad','actionplan','quantopo']

def edge_outcome(first, second):
    if first['present'] is not True or second['present'] is not True:
        return 'missing'
    if first['offset_frame'] <= second['onset_frame']:
        return 'satisfied'
    if second['offset_frame'] <= first['onset_frame']:
        return 'reversed'
    return 'overlap'


def compute(root):
    dag=json.loads((root/'events_dag.json').read_text())
    annotation=json.loads((root/'annotations.json').read_text())
    cases=dag['records'];protocol=dag['protocol']
    manifest=json.loads((root/'generation_manifest.json').read_text())
    output_frames={(o['caption_id'],o['method']):o['frames'] for o in manifest['outputs']}
    assert len(cases)==180 and len({c['caption_id'] for c in cases})==180
    normalized=lambda s:' '.join(re.findall(r'[a-z0-9]+',s.lower()))
    assert len({normalized(c['caption']) for c in cases})==180
    assert len(annotation['raters'])==3
    raters=[{(r['caption_id'],r['method']):r for r in a['annotations']} for a in annotation['raters']]
    assert all(len(a)==len(cases)*len(METHODS) for a in raters)
    by=defaultdict(list)
    for i,c in enumerate(cases):by[c['source_group']].append(i)
    order=protocol['bootstrap_group_order']
    assert len(order)==len(set(order)) and set(order)==set(by)
    groups=[by[k] for k in order];sizes=np.array([len(x) for x in groups]);G=len(groups)
    draws=np.random.default_rng(protocol['bootstrap_seed']).integers(0,G,(protocol['bootstrap_replicates'],G))
    results={};majority=[]
    for method in METHODS:
        rows=[]
        for c in cases:
            ds=[]
            for rater in raters:
                row=rater[(c['caption_id'],method)]
                d={e['event_id']:e for e in row['event_labels']}
                assert set(d)=={e['event_id'] for e in c['events']}
                for e in d.values():
                    if e['present'] is True:assert 0<=e['onset_frame']<=e['offset_frame']<output_frames[(c['caption_id'],method)]
                ds.append(d)
            events=[]
            for e in c['events']:
                votes=sum(d[e['event_id']]['present'] is True for d in ds)
                events.append(dict(event_id=e['event_id'],present=votes>=2,present_votes=votes))
            full=all(e['present'] for e in events);edges=[]
            for a,b in c['precedence_edges']:
                outcomes=Counter(edge_outcome(d[a],d[b]) for d in ds)
                votes=outcomes['satisfied']
                winners=[k for k,v in outcomes.items() if v>=2]
                if len(winners)!=1:
                    raise ValueError(f"Edge outcome requires adjudication: {c['caption_id']} {method} {a}->{b}")
                outcome=winners[0]
                assert (outcome=='satisfied') == (votes>=2)
                edges.append(dict(predecessor=a,successor=b,satisfied=votes>=2,satisfied_votes=int(votes),outcome=outcome))
            seq=sum(e['satisfied'] for e in edges)/len(edges)
            causes=[sum(e['outcome']==k for e in edges)/len(edges) for k in ['missing','overlap','reversed']]
            assert abs(seq+sum(causes)-1.0)<1e-12
            rows.append([seq,int(full),seq if full else 0.,*causes])
            majority.append(dict(caption_id=c['caption_id'],method=method,events=events,edges=edges,fully_covered=full,seq=seq))
        rows=np.array(rows);totals=rows.sum(0);grouped=np.stack([rows[ix].sum(0) for ix in groups]);sample=grouped[draws].sum(1);den=sizes[draws].sum(1)
        boot=np.stack([sample[:,0]/den,sample[:,1]/den,sample[:,2]/sample[:,1]],1)
        results[method]=dict(seq=float(totals[0]/180),coverage=float(totals[1]/180),order=float(totals[2]/totals[1]),ncov=int(totals[1]),ci=np.quantile(boot,[.025,.975],axis=0).T.tolist(),missing=float(totals[3]/180),overlap=float(totals[4]/180),reversed=float(totals[5]/180))
    summary=dict(caption_count=180,video_count=len(cases)*len(METHODS),source_clusters=G,bootstrap_replicates=protocol['bootstrap_replicates'],bootstrap_seed=protocol['bootstrap_seed'],results=results)
    return summary,dict(records=majority)

def save(root):
    summary,majority=compute(root);out=root/'results';out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'majority_labels.json').write_text(json.dumps(majority,indent=2))
    with (out/'scores.csv').open('w',newline='') as f:
        fields=['method','seq','seq_ci_low','seq_ci_high','coverage','coverage_ci_low','coverage_ci_high','order_given_coverage','order_ci_low','order_ci_high','ncov','missing','overlap','reversed']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for method,r in summary['results'].items():
            values=[method,r['seq'],*r['ci'][0],r['coverage'],*r['ci'][1],r['order'],*r['ci'][2],r['ncov'],r['missing'],r['overlap'],r['reversed']]
            w.writerow(dict(zip(fields,[x if isinstance(x,(str,int)) else f'{x:.3f}' for x in values])))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path,nargs='?',default=Path(__file__).resolve().parent)
    print(json.dumps(save(p.parse_args().directory),indent=2))
