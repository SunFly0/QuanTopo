"""Compute a paired difference only when both methods have matched bootstrap draws."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--results',type=Path,required=True);p.add_argument('--dataset',required=True)
p.add_argument('--a',required=True);p.add_argument('--b',required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();summary=json.loads((a.results/'baseline_ci.json').read_text())
entry=summary['sample_files'][a.dataset];file=a.results/entry['filename']
assert hashlib.sha256(file.read_bytes()).hexdigest()==entry['sha256']
with np.load(file) as data:
 methods=data['methods'].tolist();metrics=data['metrics'].tolist()
 if a.a not in methods or a.b not in methods:raise SystemExit('Both methods need complete matched outputs; missing methods are not substituted.')
 delta=data['values'][:,methods.index(a.a)]-data['values'][:,methods.index(a.b)]
 rows={r['method']:r for r in summary['results'] if r['dataset']==a.dataset}
 result=dict(dataset=a.dataset,method_a=a.a,method_b=a.b,contrast='a minus b',plan_sha256=entry['plan_sha256'],bootstrap_replicates=len(delta),metrics={})
 for i,k in enumerate(metrics):
  point=rows[a.a]['metrics'][k]['point_estimate']-rows[a.b]['metrics'][k]['point_estimate']
  result['metrics'][k]=dict(difference=point,ci95=np.quantile(delta[:,i],[.025,.975]).tolist())
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
