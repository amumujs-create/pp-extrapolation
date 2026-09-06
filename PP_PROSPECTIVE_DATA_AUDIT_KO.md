# PP 기준 추가 데이터 적합성 감사

PP 논문의 새 데이터는 PAE 사용 여부가 아니라 PP/OFF-UQ 구조 선택에 test label을 사용했는지로 정의한다.

| 후보 | PP 기준 상태 | 독립 unit/EOL | 현재 판정 |
|---|---|---|---|
| IMS bearing | PAE-only | 실패 bearing 4개 | OOF 최소 12 unit 미달 |
| FEMTO/PRONOSTIA | PAE-only | 학습 bearing과 공식 test bearing 존재 | 로컬 원신호/추출 자료 부재 |
| C-MAPSS | PP-seen | 충분 | support-adaptive PP 결과를 이미 확인하여 prospective 아님 |
| PHM2010 milling | PAE-only | 약 5 tool | 독립 unit 부족 |
| N-CMAPSS | PAE/PP 관련 개발 이력 | 충분 가능 | PP-independent claim이 약하고 다운로드 불완전 |

따라서 개념에 가장 맞는 다음 후보는 FEMTO/PRONOSTIA다. 시간 순서, 실제 run-to-failure 학습 bearing, 공식 truncated test RUL이 있고 기계 도메인 외삽이라는 점에서 PP의 범위를 넓힌다. 다만 현재 로컬에는 loader만 있고 원자료가 없어 실행할 수 없다.

IMS나 milling을 억지로 사용하면 cycle을 독립 unit처럼 취급해야 하므로 pseudo-replication이 된다. 데이터 수를 늘리기 위한 본표에는 넣지 않는다.

다음 실행 순서는 FEMTO 원자료 확보, 파일 해시 기록, PP protocol commit 이후 test 봉인, train bearing 내부 OOF head 학습, 공식 truncated bearing 단일 평가다.
