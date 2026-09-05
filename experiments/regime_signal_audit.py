"""Known synthetic rate change audit; threshold/window fixed before scoring."""
import json
from pathlib import Path
import numpy as np
from pp_extrapolation.temporal import DegradationContract,causal_history

rows=[]
for scenario in ('stable','abrupt','gradual'):
    for seed in range(10):
        rng=np.random.default_rng(seed);time=np.arange(150.)
        rate=np.full(150,.002)
        if scenario=='abrupt':rate[75:]=.007
        if scenario=='gradual':rate[75:]=np.minimum(.002+.0002*np.arange(75),.007)
        q=1.-np.cumsum(rate)+rng.normal(0,.001,150)
        h=causal_history(time,q,DegradationContract(0.,'decreasing'))
        flag=h['disagreement']>.35
        # Persistence suppresses isolated spikes; warmup excludes unformed windows.
        alarm=np.array([i for i in range(26,150) if flag[i-2:i+1].all()])
        after=alarm[alarm>=75]
        rows.append(dict(scenario=scenario,seed=seed,prechange_alarm_count=int((alarm<75).sum()),
                         postchange_alarm_count=int(len(after)),first_delay=int(after[0]-75) if len(after) else None))
out=Path('results/temporal_regime_audit');out.mkdir(parents=True,exist_ok=False)
(out/'results.json').write_text(json.dumps(dict(threshold=.35,persistence=3,warmup=26,change=75,rows=rows),indent=2)+'\n')
for scenario in ('stable','abrupt','gradual'):
    r=[v for v in rows if v['scenario']==scenario];delays=[v['first_delay'] for v in r if v['first_delay'] is not None]
    print(scenario,'prechange alarms',sum(v['prechange_alarm_count'] for v in r),'postchange alarms',sum(v['postchange_alarm_count'] for v in r),'detections',len(delays),'median delay',np.median(delays) if delays else None)
