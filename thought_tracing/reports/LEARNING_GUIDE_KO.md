# 논문 평가를 이해하고 직접 테스트하는 순서

**2026-10-26:** 이 문서의 3사례와 기존 20문항은 구조 학습용으로 보존한다. 논문 비교를 위한 공식 코드 기본 600문항의 별도 실행은 [새 문항·조건·진행 안내](PAPER_COMPARISON_ALIGNMENT_KO.md)를 참조한다. 기존 `--phase pilot` 명령이 새 600문항을 실행하는 것은 아니다.

2026-10-21. 논문 v2와 기존 `bf16_v2`의 실제 기록에 근거한 학습 안내다. 이 안내를 만들면서 추가 모델 추론·새 실험·알고리즘 수정을 하지 않았다. 기존 결과와 영어 데이터·프롬프트는 보존했다.

**출처 구분 — 2026-10-22 보완:** 1906·3431·2097의 이야기·질문·gold는 공식 공개 데이터, 세 사례의 선택은 우리 프로젝트의 고정 seed, Base/CoT/TT 출력과 TT 중간 기록은 사용자의 Windows/A6000에서 실행한 Qwen2.5-7B-Instruct BF16 결과다. 한국어 해설은 그 기록에 대한 Codex의 분석이다. 이 세 사례는 논문이 제시한 실행 예시가 아니며, 아래 출력은 저자들의 모델 출력이 아니다.

저자 측의 같은 세 사례 예측·trace는 확인한 논문 v2와 기준 커밋 공개 파일에서 확보하지 못했다. 공식 코드 기본값의 600문항 선택을 재계산하면 3431만 포함되지만, 실제 논문 실행의 입력 manifest나 개별 예측을 확인한 것은 아니다. [사례 출처·논문 모델·TT 구조](EXAMPLE_PROVENANCE_AND_TT_STRUCTURE_KO.md)에 근거와 범위를 기록했다.

논문 평가 설정은 [v2 §3](https://arxiv.org/html/2502.11881v2#S3), 세부 생성 설정은 [Appendix B](https://arxiv.org/html/2502.11881v2#A2), 알고리즘은 [§2.2](https://arxiv.org/html/2502.11881v2#S2.SS2)를 참조한다. 논문의 최종 답변 평가와 우리가 기록한 중간 단계 점검은 구분한다. 모델이 생성한 중간 설명은 인간의 실제 사고 과정을 입증하는 자료가 아니다.

## 1. 먼저 사람이 풀기: 아래 3개 영어 원문만 읽는다

아래 이야기와 질문은 `manifests/smoke3.json`에서 그대로 옮겼다. 아직 기존 예측이나 채점을 열지 않고 다음 표를 직접 채워 본다. 사건이 진행될 때마다 별도 행을 추가하면 관측이 믿음에 미치는 영향을 보기 좋다.

| 사례 | 마지막 실제 물건 위치 | 누가 어떤 이동을 보았는가 | 질문이 묻는 믿음의 주체 | 내가 예상하는 답 |
|---|---|---|---|---|
| 1906 |  |  |  |  |
| 3431 |  |  |  |  |
| 2097 |  |  |  |  |

### 1906

```text
Emily stepped into the lounge. Jayden walked into the lounge. There is a peas in the cupboard. The cupboard is laid in the lounge. Liam showed up at the lounge. Jayden took the peas to the treasure chest. The lounge is where the treasure chest is. Emily is besotted with the sweatshirt. Emily walked out of the lounge. Jayden stepped out of the lounge. Emily slithered into the garage.

Where will Jayden look for the peas?
```

### 3431

```text
Nathan bounded into the crawlspace. Liam arrived at the crawlspace. A sweater is in the box. The box is in the crawlspace. Aria arrived at the crawlspace. Nathan walked out of the crawlspace. Liam stowed the sweater in the suitcase. The suitcase is found in the crawlspace.

Where will Nathan look for the sweater?
```

### 2097

```text
Charlotte tiptoed into the dining room. Mila bounded into the dining room. The peach is found in the bottle. The bottle is set in the dining room. Mila exited the dining room. Mila walked into the closet. Charlotte pushed the peach to the basket. The basket is put to rest in the dining room.

Where does Charlotte think that Mila searches for the peach?
```

## 2. 이미 있는 기록과 대조하기: 새 모델 실행은 필요 없다

직접 예상 답을 적은 다음 아래 명령을 실행한다. 첫 명령은 공식 gold와 세 조건의 최종 답변을 함께 출력하므로 미리 풀어 본 뒤 사용한다. 모두 저장된 파일만 읽는 명령이다.

```powershell
Set-Location C:\code\yb-thought-tracing
py -3.12 scripts\print_cases.py --run bf16_v2 --split smoke3
py -3.12 scripts\inspect_trace.py --run bf16_v2 --split smoke3 --row 3431 --compact
```

TT 기록은 다음 순서로 읽는다. 숫자가 붙었다고 참인 가설이 된 것은 아니며, 가중치는 LLM의 행동 그럴듯함 평가를 변환한 상대 점수다.

| 기록 | 확인할 내용 | 연결되는 원리 |
|---|---|---|
| `target` | 질문의 대상 인물이 맞는가? 2차 질문에서 두 인물의 역할을 보존하는가? | 추론할 마음의 주체 |
| `action_label`, `preprocessing` | 원문 문장이 누락되지 않았는가? 대상 인물의 행동과 나머지 상태가 적절히 분리됐는가? | 상태·행동 분리 |
| `perceptions` | 자리를 떠난 뒤의 이동을 관측했다고 잘못 가정하는가? 원문에 없는 주의·지식을 덧붙이는가? | 관측과 실제 사건의 구분 |
| `initialize`, `propagate` | 4개 가설이 새 관측을 반영하는가? 못 본 사건 때문에 믿음이 바뀌는가? | 여러 가능한 믿음의 유지·전파 |
| `weigh` | 평가 선택지·수치·가중치가 어떻게 연결되는가? 이전 가중치가 다음 갱신에 어떻게 쓰이는가? | 행동에 근거한 가설 평가 |
| ESS·다양성·재표집/재표현 이벤트 | 조건을 만족했는가? 이벤트가 실제 발생했는가? | 가설 분포 관리 |
| `weighted_summary`와 `result.json` | 요약이 보존한 믿음을 최종 답변도 따르는가? | 추론 기록과 답변의 연결 |

`--compact`여도 JSON 출력이 길 수 있다. 먼저 아래 한국어 해설을 읽고 해당 영어 로그를 대조하면 된다.

- [1906의 단계별 TT 해설](bf16_v2/SMOKE_TRACE_EXPLAINED_KO.md)
- [3431·2097의 단계별 오류 분석](bf16_v2/SMOKE_FAILURE_REVIEW_KO.md)

### 직접 예상한 뒤 확인할 사항

| 사례 | 공식 gold와 이야기에서 기대하는 관계 | 기존 실행에서 배울 점 |
|---|---|---|
| 1906 | Jayden이 직접 peas를 옮겼으므로 treasure chest 믿음을 갖는다. | TT는 정답이지만 관측·요약의 일부 표현에 문제가 있다. 최종 정답만으로 모든 단계가 맞았다고 판단하지 않는다. |
| 3431 | 실제 sweater는 suitcase에 있지만, 이동 전에 떠난 Nathan은 box로 예상한다. | 마지막 TT 가설·요약은 box 믿음을 보존했는데 최종 답은 suitcase다. 주요 불일치가 최종 답변 선택에 있음을 확인할 수 있다. 앞선 모든 단계가 완벽했다는 뜻은 아니다. |
| 2097 | 실제 peach는 basket에 있고, Mila의 옛 믿음은 bottle이다. Charlotte는 Mila가 이동을 못 봤다는 것을 고려해야 한다. | 2차 질문에서 Charlotte와 Mila의 역할 및 물건 위치를 구별한다. 기존 Base는 bottle을 맞히면서 방을 closet으로 잘못 붙여 자동 채점이 놓치는 문제가 있다. |

이 분류는 원문과 실제 출력에 근거한 분석이며 전체 사람 검토는 끝나지 않았다. 이번 23개 TT 실행에서 실제 재표집·재표현은 0회였다. 따라서 이 기록만으로 그 분기의 효과까지 체험했다고 볼 수 없다. 필요하면 `scripts/check_compatibility.py` 같은 기존의 작은 검증과 코드 읽기를 참고하되, 이를 모델에서 해당 분기가 실제 작동한 증거로 바꾸어 설명하지 않는다.

## 3. 직접 새로 생성하기: 먼저 3사례만 실행한다

`configs/bf16_mytest.json`은 BF16 기준 설정에서 `run_version`만 바꾼 별도 설정이다. 이전에 이 이름으로 완료한 실행이 없다면 새로 생성한다. 이미 실행한 이름이면 완료 결과/호출을 재사용한다.

```powershell
$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\python.exe -u -m repro.run --config configs/bf16_mytest.json --phase smoke
py -3.12 scripts\audit_smoke.py --run bf16_mytest
py -3.12 scripts\inspect_trace.py --run bf16_mytest --split smoke3 --row 3431 --compact
```

연결용 짧은 생성 후 3사례 × 3조건을 실행한다. 한 프로세스의 모든 조건·TT 단계가 같은 Qwen 모델 객체를 사용한다. 기존 3사례 호출 합은 약 6.2분이며 새 실행 시간은 달라질 수 있다. 이 명령은 A6000 BF16 환경용이다. 집의 3080 10GB용 4비트 경로는 [빠른 실행 안내](QUICKSTART_KO.md)와 [README](../README.md)를 참조하고 별도 조건으로 기록한다.

새 출력에서 2번의 표를 다시 채운다. 기존 결과와 다른 경우에도 우선 입력·설정·버전·실행 환경·로그를 비교하고, 정답률을 보고 프롬프트나 사례를 교체하지 않는다. 실패한 최초 단계가 불명확하면 미확정으로 둔다.

## 4. 20문항은 그다음에: 실패 유형과 계산량을 함께 읽는다

기존 20문항 보고서를 먼저 읽어도 된다. [전체 결과](bf16_v2/REPORT.md)에서 다음 묶음을 비교하면 평균 정답률보다 많은 것을 배울 수 있다.

- 1차/2차, 참/거짓 믿음의 네 묶음.
- Base와 TT가 모두 맞은 사례, TT에서 좋아진 4293, TT에서 나빠진 3756/3850.
- 공식 substring 점수와 별도 채점·사람 검토 표시의 차이.
- 조건별 호출 수와 전체 토큰 수. 동일 모델을 썼지만 계산량은 동일하지 않다.

직접 재실행하려면 새 실행 이름의 smoke 9조건이 정상 완료된 후 다음을 선택해서 실행한다.

```powershell
.\.venv\Scripts\python.exe -u -m repro.run --config configs/bf16_mytest.json --phase pilot
py -3.12 scripts\analyze_results.py --run bf16_mytest
```

이번 결과는 논문의 전체 성능 수치 재현과 구별해야 한다. 우리는 Qwen 7B 하나를 모든 단계에 사용했다. 논문은 더 큰 모델들을 평가하고 ToMi/BigToM의 행동 표시에는 GPT-4o를 사용했다. 우리의 파일럿은 사실·기억 질문을 제외한 믿음 질문 20개다.

## 5. 관측 여부 10쌍은 현재 설계만 검토한다

[관측 조건 10쌍 검토표](../design/OBSERVATION_PAIRS_REVIEW_KO.md)를 보며 물리적 사건과 물건 위치를 고정한 채 관측 조건만 바꾸면 왜 정답 믿음이 달라져야 하는지 설명해 본다. 재관측·대화로 정보가 다시 전달되지 않는지도 확인한다. 현재 범위에서는 새 모델 실행을 하지 않는다.

학습 결과로는 “TT가 몇 개 더 맞혔다”보다 다음 형식의 메모 한 개가 유용하다: `원문 근거 → 기대 관측/믿음 → 실제 중간 출력 → 최초 불일치 단계 → 대안 설명/미확정`. 예를 들어 3431은 마지막 이동의 관측 판별과 마지막 믿음 요약이 적절해도 최종 답변이 이를 따르지 않을 수 있음을 보여준다. 이것은 관측 추론 제거 같은 논문 기존 ablation을 새로운 기여로 주장하는 것과 구별한다.
