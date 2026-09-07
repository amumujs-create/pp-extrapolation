# PP 및 경쟁모델 튜닝 감사

## 경쟁 NN

V-REx, GroupDRO, monotone NN은 양의 R² 외삽 설정에서 다시 탐색했다. 각 모델은 24개 width/depth/learning-rate/weight-decay 조합과, 선택된 조합의 penalty 5개를 사용해 총 29개 validation 후보를 평가했다. 선택된 후보는 seed 42--46으로 재학습했다. test label은 선택에 쓰지 않았다.

이 확장 탐색은 경쟁모델을 항상 높이지는 않았다. HUST GroupDRO는 소규 탐색의 ensemble 0.934보다 낮았고, MATR batch2 V-REx는 0.850으로 높아졌다. 보수적 성능 감사에서는 해당 모델에서 관측된 더 높은 수치를 PP의 비교 기준으로 남겼다. 이는 확증 순위가 아니라 post-hoc upper-bound stress audit이다.

## PP

- HUST: width/group-DRO/affine-anchor 탐색 후 validation group-LOO rate-conditioned transport. pooled R² 0.958.
- MATR batch2: 18개 architecture/optimizer 기본 후 15개 DRO/decay 조합, 총 33개 후보. group-LOO state/rate transport 포함 pooled R² 0.862.
- NASA battery: fold별 18개 architecture/optimizer + 11개 DRO/decay/anchor 후보는 0.506, blocked transport는 0.522로 채택하지 않음. 별도 regime-spline/Jacobian 구조 탐색에서 네 fold 모두 unconstrained ReLU-hinge spline을 validation이 선택해 pooled R² 0.571.

이 감사 후 양의 R² 8개 설정에서 개선 PP가 관측된 경쟁모델 최고치보다 높다. 독립 untouched cohort에 프로토콜을 고정하기 전까지는 개발 결과로 표기한다.
