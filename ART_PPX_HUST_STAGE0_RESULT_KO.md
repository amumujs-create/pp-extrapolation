# ART-PPX HUST Stage-0 결과

실행일: 2026-09-12  
연구자: 박진서

## 결론

ART-PPX의 첫 사전등록 조기중단 실험은 **승격 실패**다. 따라서 동결된
규칙에 따라 MICH 및 나머지 데이터셋으로 확장하지 않는다.

## 결과

- PP-X test: RMSE 30.900, R² 0.9580
- 최저 validation 후보:
  - 전체 context
  - Ridge alpha 100
  - correction bound 20%
  - authority 1.0
- validation RMSE: 34.922 → 34.869 (0.15% 개선)
- validation unit win: 10/16
- validation worst-unit RMSE ratio: 1.039
- unit bootstrap 개선량 95% CI: [-0.347, 0.541]
- 개선 확률: 0.606

요구한 최소 validation 개선 2%를 충족하지 못했고 bootstrap CI도 0을
포함하므로 correction은 승인되지 않았다. 실행 authority는 0으로
fallback되었으며, test 결과는 PP-X와 정확히 동일하다.

## 구조 검증

- support boundary에서 correction이 0이 되는 적분 구조를 구현했다.
- authority 0에서 PP-X exact replay 오차는 0이다.
- 안전성은 fallback으로 보존됐지만, derivative transport 자체의 유효한
  성능 근거는 얻지 못했다.

## 해석

HUST의 최종 PP-X는 이미 validation-selected regime transport를 사용한다.
그 위에서 validation residual derivative를 추가로 학습해 얻는 독립 신호가
거의 없었다. 따라서 이 결과를 모델 노벨티 또는 성능 개선으로 주장할 수
없다.

ART-PPX는 아이디어 기록과 음성 결과로만 보존하고 PP-X 메인 모델에는
포함하지 않는다. 다음 challenger는 같은 residual correction의 변형이
아니라, PP-X가 약한 계약에서 구조적으로 다른 학습 목표를 사용해야 한다.

재현:

```bash
PYTHONPATH=src /opt/anaconda3/bin/python experiments/art_ppx_hust_stage0.py
```

원자료:

- `protocols/ART_PPX_HUST_STAGE0_PROTOCOL.md`
- `results/art_ppx_hust_stage0/results.json`
- `results/art_ppx_hust_stage0/predictions.npz`
