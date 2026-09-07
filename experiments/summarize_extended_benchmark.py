import json,numpy as np
from pathlib import Path
P=Path('results/extended_nn_benchmark_v1');models=['pp','pp_joint','mlp','resnet','ft_transformer','moe','single','fixed','unconstrained','no_affine','gru','tcn','temporal_transformer','ridge','spline','boosting','tabpfn'];rows=[]
for name in models:
 path=P/(name+'.json')
 if not path.exists():continue
 r=json.load(open(path));e=r['ensemble'];rows.append({'model':name,'pooled_r2':e['pooled']['r2'],'rmse':e['pooled']['rmse'],'macro_r2':e['unit_macro_r2'],'mean_r2':r.get('mean_r2'),'sd_r2':r.get('sd_r2'),'runtime_seconds':sum(v.get('seconds',0) for v in r.get('search',[]))+sum(v.get('seconds',0) for v in r.get('runs',[]))})
rows.sort(key=lambda r:r['pooled_r2'],reverse=True);lines=['# 추가 NN 벤치마크 결과','', 'MATR 2019 동일 row 분할. 사후 비교이며 모델 종류 전체에 대한 최적 성능을 의미하지 않는다. 9개 후보를 seed 42 validation으로 선택한 뒤 5개 seed 재학습. 150 epoch / patience 25. 이전 300 epoch / seed별 선택 결과와 프로토콜이 다르므로 직접적인 개선·열화 원인으로 단정하지 않는다.','', '| 모델 | pooled ensemble R² | single mean | seed SD | RMSE | macro R² |','|---|---:|---:|---:|---:|---:|']
for r in rows:
 fmt=lambda x:'—' if x is None else f'{x:.3f}'
 lines.append('| '+r['model']+' | '+' | '.join(fmt(r[k]) for k in ['pooled_r2','mean_r2','sd_r2','rmse','macro_r2'])+' |')
lines+=['','기존 확증 latent PP 0.257은 원본 프로토콜의 결과로 별도 보존한다. 이번 pp는 regularizer를 끈 구조 대조이며 pp_joint는 중간 결과 확인 이후 추가한 탐색군이다. Test에 따른 모델 선택을 새로운 확증 성공으로 보고하지 않는다.','', 'TabPFN은 CPU 1,000 train rows, 1 internal estimator를 사용한 보조 조건이다. GRU/TCN/temporal Transformer는 같은 과거 8개 관측을 직접 받고, 표형 모델은 그 관측으로 만든 6개 요약 입력을 받는다.','', 'RTDL ResNet/FT-Transformer는 rtdl-revisiting-models 0.0.2 공식 패키지. TCN은 causal dilated CNN task adaptation. https://github.com/yandex-research/rtdl-revisiting-models','', 'raw/clipped row predictions, validation 후보, refit checkpoints는 results/extended_nn_benchmark_v1 에 저장한다. 체크포인트는 state_dict이며 모델 재구성에는 코드와 train 자료가 함께 필요하다.']
Path('EXTENDED_NN_BENCHMARK_RESULTS_KO.md').write_text('\n'.join(lines)+'\n');(P/'summary.json').write_text(json.dumps(rows,indent=2));print('\n'.join(lines))
