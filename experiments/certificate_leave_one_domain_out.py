#!/usr/bin/env python3
"""Leave-one-domain-out threshold selection for the PP applicability certificate."""
import itertools,json
from pathlib import Path

GRID={'gain':(.001,.005,.01,.05,.1,.25),'activity':(.0001,.001,.005,.01,.05,.2),
      'disagreement':(.1,.25,.5,1.),'correlation':(0.,.1,.2,.4,.6)}
def decision(x,t):
 return x['relative_pp_mse_gain']>=t['gain'] and x['residual_activity']>=t['activity'] and x['normalized_seed_disagreement']<=t['disagreement'] and x['max_abs_feature_rul_spearman']>=t['correlation']
def select(train):
 candidates=[]
 for values in itertools.product(*GRID.values()):
  t=dict(zip(GRID,values));pred=[decision(x['validation_only'],t) for x in train];truth=[x['test_audit']['success_definition'] for x in train]
  fa=sum(a and not b for a,b in zip(pred,truth));fr=sum((not a) and b for a,b in zip(pred,truth));correct=sum(a==b for a,b in zip(pred,truth))
  # A certificate must require positive evidence. On exact training ties prefer
  # the less restrictive disagreement ceiling to avoid arbitrary abstention.
  complexity=t['gain']+t['activity']+t['correlation']
  candidates.append((fa,-correct,fr,-sum(pred),complexity,-t['disagreement'],t))
 feasible=[x for x in candidates if x[0]==0];return min(feasible or candidates)[-1]
def main():
 source=json.loads(Path('results/applicability_certificate_v1/results.json').read_text())['datasets'];rows=[]
 for held in source:
  threshold=select([v for k,v in source.items() if k!=held]);prediction=decision(source[held]['validation_only'],threshold);truth=source[held]['test_audit']['success_definition']
  rows.append({'held_out':held,'thresholds':threshold,'predicted_usable':prediction,'test_success':truth,'correct':prediction==truth})
  print(held,prediction,truth,threshold)
 payload={'experiment':'certificate_leave_one_domain_out_v1','threshold_grid':GRID,'selection':'zero training false accepts, then maximum accuracy, fewer false rejects, greater coverage','folds':rows,
          'summary':{'correct':sum(x['correct'] for x in rows),'total':len(rows),'false_accept':sum(x['predicted_usable'] and not x['test_success'] for x in rows),'false_reject':sum((not x['predicted_usable']) and x['test_success'] for x in rows)}}
 out=Path('results/certificate_lodo_v1');out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps(payload,indent=2)+'\n');print(payload['summary'])
if __name__=='__main__':main()
