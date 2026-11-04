# 관측 여부 10쌍 — 실행 전 검토표

**설계와 구조 점검만 완료했으며 모델 호출은 0회다. 사람의 의미·정답 검토는 아직 필요하다.** 원래 영어 문장과 질문은 observation_pairs10.json에 그대로 보존했다. 아래 한국어는 설계 설명이다.

각 쌍은 처음 물건 위치를 모두 본 뒤 한 번 이동한다. 물건 이동자·출발/도착 container·방·최종 실제 위치를 유지하고, 대상이 이동을 보았는지 여부를 바꾼다. 2차 질문에서는 바깥 인물이 안쪽 인물의 관측 여부를 정확히 안다는 조건도 이에 맞게 표시한다. 이는 한 정보 접근 조작의 명시이며 새로운 물리 이동을 추가하지 않는다.

| 쌍 | 차수 | 관측 조건 대상 | 질문(원문) | 관측 시 gold | 미관측 시 gold | 두 조건의 실제 위치 |
|---|---:|---|---|---|---|---|
| obs-01 | 1 | Ben | Where will Ben look for the marble? | basket | box | basket in the kitchen |
| obs-02 | 1 | Emma | Where will Emma look for the key? | cupboard | drawer | cupboard in the study |
| obs-03 | 1 | Henry | Where will Henry look for the coin? | suitcase | envelope | suitcase in the bedroom |
| obs-04 | 1 | Kevin | Where will Kevin look for the ribbon? | bucket | crate | bucket in the garage |
| obs-05 | 1 | Noah | Where will Noah look for the card? | box | basket | box in the office |
| obs-06 | 2 | Quinn | Where does Peter think Quinn will look for the button? | envelope | drawer | envelope in the hallway |
| obs-07 | 2 | Thomas | Where does Sara think Thomas will look for the ring? | cupboard | box | cupboard in the lounge |
| obs-08 | 2 | Wendy | Where does Victor think Wendy will look for the ticket? | crate | suitcase | crate in the attic |
| obs-09 | 2 | Zach | Where does Yara think Zach will look for the eraser? | basket | bucket | basket in the classroom |
| obs-10 | 2 | Chloe | Where does Brian think Chloe will look for the badge? | drawer | envelope | drawer in the workshop |

## 사람이 실행 전에 확인할 사항

- 원문의 ‘이동을 보았다/못 봤다’ 문장이 실제 정보 접근을 충분히 정의하는지 확인한다.
- 초기 위치를 모두 안다는 조건과 이후 재관측·대화·청각/행동 단서 배제가 모순되지 않는지 확인한다.
- 2차 질문의 바깥 인물이 안쪽 인물의 관측 여부를 정확히 아는지 확인한다.
- observed/unobserved의 gold가 각각 새 위치/옛 위치인지 검토하고 고정한다.
- 실행 전 순서 교차·모델·seed·채점 규칙과 별도 run 버전을 고정한다.

주요 지표는 쌍의 두 조건을 모두 맞힌 수 /10이다. 1차·2차별 공동 정답과 실제 실패 단계도 기록한다. 믿음 필드 개선안을 시험한다면 필드–최종 답변 불일치와 gold 정확도를 함께 본다. 이 쌍들은 단순한 공통 template을 공유하므로 독립적인 자연 상황 10개라고 주장하지 않는다.

문자열·해시·명시된 접근 조건에 따른 구조 점검은 의미적 타당성을 자동 입증하지 않는다. 현재 어떤 개선 방법도 실행하지 않았다.
