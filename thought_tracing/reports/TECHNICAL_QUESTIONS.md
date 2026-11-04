# 김현우 교수에게 확인할 기술 질문 후보 (최대 3개)

2026-10-21 **실제 Qwen2.5 7B BF16 점검 9조건과 파일럿 60조건을 완료**했다. 공식 점수는 Base 12/20, CoT 14/20, TT 11/20이며 작은 모델/표본의 결과다. 질문 1/2의 근거는 정적 독해와 최소 인터페이스 실행 재현이며, 질문 3에는 실제 점검 로그도 더했다. 실제 재표집/재표현은 모두 0회라 해당 분기의 모델 효과는 관측하지 못했다. 성능 문제 또는 논문의 잘못으로 확정하지 않는다. 연락·발송하지 않았다. 첫 연락은 아래 재현 질문 중심으로 하고 인턴 문의와 구분한다.

## 1. 공개 평가에서 호출별 생성 설정을 적용한 경로는 무엇인가?

- **연관:** 논문 v2 Appendix B의 greedy 설정, §2.2 Rejuvenation; 원본 `agents/base.py:74–88`, `agents/together_ai.py:173–200`, `tracer.py:611/644/659`, `revised_tomi/test_tomi.py:199/209`.
- **기대:** 평가 코드가 CoT에 `max_tokens=512`, 최종 답변에 256을 주고 재표현에 `temperature=1`을 주면 실제 backend에도 그 값이 전달된다.
- **확인한 동작:** `AsyncBaseAgent.batch_interact`는 `batch_generate(..., temperature=None, max_tokens=None)`를 호출한다. 실제 상속 함수에 stub `generate`를 연결했을 때 요청 `(0.37,17)` 대신 `(None,None)`이 도착했다. Qwen remote 클래스도 이 상속 경로를 쓴다. 외부 API 서버의 결과까지 시험한 것은 아니다.
- **최소 근거:** `scripts/check_compatibility.py`, `records/compatibility_checks.json`의 `original_async_batch_parameter_drop`. 실제 Qwen 기본값은 loader에 0/1024로 지정되어 있다. 공개 코드만 보면 CoT·최종 상한과 재표현 온도가 호출부와 다를 수 있다.
- **대안 설명:** 실제 논문 실험은 다른 wrapper/비공개 설정/이전 커밋을 썼을 수 있다. 특히 Appendix B의 greedy 설명은 이 전달 손실로 재표현 온도도 0이 되는 실제 경로와 일치할 가능성이 있다. 따라서 단순 문서 오류라고 결론내릴 수 없다.
- **질문:** “논문 결과를 산출한 backend에서 batch 호출의 temperature/max_tokens는 호출별 설정과 agent 기본값 중 어느 것이 적용되었나요? 해당 실행 config나 재현 명령을 확인할 수 있을까요?”
- **추가 확인:** 공식 실행 설정과 backend별 effective generation 설정. 우리의 local adapter는 명시적 인수를 적용하므로 원본 경로와의 차이를 먼저 합의해야 공정한 비교가 가능하다.

## 2. 재표집과 재표현의 분기 및 누락 옵션의 의도는 무엇인가?

- **연관:** v2 Algorithm 1의 별도 resample/rejuvenate if, §2.2, Appendix B; 원본 `tracer.py:28–40`, `649–665`, `744–757`, `hypothesis.py:134–159`.
- **기대:** 문서화된 재표현이 저다양성 조건에서 실행 가능하고, resampling 이후에도 필요한 rejuvenation을 어떻게 처리하는지 명확하다.
- **확인한 동작:** 실제 코드는 action 단계에서 `if ESS<2: resample; elif diversity<0.25: rejuvenate`이므로 같은 단계에서 둘 다 하지 않는다. no-action 단계에서는 검사하지 않는다. 재표현 함수가 읽는 `args.use_perception_only`는 원본 parser에 없다.
- **최소 근거:** 원본 parser로 만든 args와 4개의 동일 텍스트를 재표현 함수에 넘기면 `'Namespace' object has no attribute 'use_perception_only'`가 발생한다. `records/compatibility_checks.json`의 해당 항목에 기록했다. 이는 해당 함수 분기의 실행 오류 재현이며, 실제 benchmark 모델 출력이 그 분기에 도달했다고 주장하지 않는다.
- **대안 설명:** 논문 실행에는 추가 args/default 설정이 있었거나, 당시 출력의 다양성이 threshold를 넘어서 해당 분기에 도달하지 않았을 수 있다. Algorithm 1이 의도적인 개략 표현일 수 있다.
- **질문:** “공개 실행에서 use_perception_only의 정상 기본값과 설정 경로는 무엇인가요? Algorithm 1의 독립 if와 구현의 if/elif, action 단계 한정 검사는 의도된 스케줄인가요?”
- **추가 확인:** 공식 args dump 또는 해당 분기를 지난 공개 로그. 이번 재현은 기본 False만 명시했고 분기 스케줄은 바꾸지 않았다. 이 수정으로 성능이 좋아졌다는 근거는 없다.

## 3. Jaccard 기반 재표현이 보존하려는 다양성의 의미는 무엇인가?

- **연관:** v2 §2.2 Resampling & Rejuvenation, Appendix B/C; 원본 `utils.py:230–251`, `tracer.py:649–665/752–757`.
- **기대/문제의식:** 자연어 입자의 다양성과 어휘의 차이는 다르므로 threshold와 의미 보존 paraphrase가 어떤 역할을 목표로 하는지 알고 싶다.
- **확인한 동작:** Lancaster stem 토큰 집합 Jaccard를 쓰고, 재표현 prompt는 “keep the meaning ... intact”를 요구한다. 직접 작성한 두 문장 `Alex believes the marble is in the box.`와 `Alex believes the marble is not in the box.`의 실제 함수 출력은 유사도 0.875, diversity 0.125다. 부정어로 믿음이 달라져도 저다양성 조건에 든다.
- **최소 근거:** `records/diversity_sanity_check.json`. 이 문장은 설명을 위한 수작업 입력이며 모델 출력이나 추가 benchmark 실험이 아니다.
- **실제 점검 근거:** `runs/bf16_v2/smoke3/row-1906/TT/trace_events.jsonl`의 step 1에서는 네 전파 가설의 핵심이 모두 cupboard의 pea를 treasure chest로 옮겼다는 내용이다. 문구 차이로 실제 diversity=0.332459가 나와 0.25 임계값에서 재표현하지 않았다. 해당 사례는 최종 정답이므로 이 현상을 오답 원인이라고 주장하지 않는다. 상세는 `reports/bf16_v2/SMOKE_TRACE_EXPLAINED_KO.md`다.
- **대안 설명:** 저자 의도는 의미적으로 새로운 입자를 제안하는 것이 아니라 문구 반복을 줄여 이후 전파의 경직성을 완화하는 것일 수 있다. 의미 보존 요구는 의도적인 설계이고, 그러므로 이를 의미 다양성 버그로 부르지 않는다.
- **질문:** “재표현은 의미적 posterior 다양성보다 어휘 반복 감소를 위한 단계로 이해하는 것이 맞나요? similarity/diversity 방향과 0.25 threshold는 어떻게 정했으며 의미적 다양성을 별도로 점검했나요?”
- **추가 확인:** 실제 재표현 전후 로그와 후속 전파 차이. 위 실제 로그만으로도 이 현상이 파일럿 오답을 일으킨다고 주장하거나 개선 성능을 예상할 수 없다.

## 개선안과 후속 검증의 현재 상태

실제 점검 row 3431에서는 마지막 가중 요약이 올바른 box 믿음을 남겼는데도 최종 답변이 suitcase로 바뀌었다(`reports/bf16_v2/SMOKE_FAILURE_REVIEW_KO.md`). **제안은 하나만 둔다:** 요약에서 대상의 믿음 위치를 명시적인 필드로 넘기고 최종 답변이 그 필드를 표현하도록 연결하는 C 변경을 검토한다. 2차 질문은 바깥/안쪽 믿음 주체도 구분해야 한다. 필드 자체가 틀릴 수 있으므로 자기 일관성만으로 성공 판정하지 않는다. 이번 실행에는 적용하지 않았다.

검증에는 수정에 사용한 점검/파일럿과 다른 **미실행 관측 10쌍**을 쓸 수 있다. 먼저 사람이 두 조건의 gold·관측 접근·물리 사건 동일성을 검토한 뒤 고정한다. 같은 모델·원 설정의 TT와 새 버전을 분리해 쌍의 두 조건 모두 정답인 비율, 실패 단계, 믿음 필드–최종 답변 불일치와 전체 호출 비용을 함께 비교한다. 추가 추출 호출이 필요하면 같은 로컬 모델을 쓰고 그 비용도 센다. 이 후속 실행은 자동으로 하지 않는다.

논문 §3.3은 이미 perception 제거, weight update 제거, self-correction을 실험했다. 위 제안의 기존 연구와의 차별성도 아직 확인되지 않았으며 새 기여나 성능 개선으로 주장하지 않는다. 이번 A/B 실행·기록 수정을 성능 개선으로 부르지 않는다. 원래 코드·출력·자동 점수는 보존한다.
