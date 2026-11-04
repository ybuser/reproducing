# ThoughtTracing fixed-4 / adaptive-A 로컬 실험 인계 및 검토 요청

나는 ThoughtTracing 논문과 구현을 이해하고, 로컬 재현 결과를 근거로 저자에게 기술 질문을 드리려 한다. 아래는 이미 실행한 실험의 기록이다. 실행 성공, 규칙 구현의 적합성, 추론 내용의 의미적 정확성, 성능 향상을 서로 구분해서 검토해 줘. 확인된 사실과 가능한 설명을 섞지 말고, 근거가 부족한 부분은 미확정으로 남겨 줘. 영어 원문 데이터·프롬프트·모델 출력을 임의로 번역하거나 바꾸지 마.

2026-11-04 작성. fixed-4는 2026-10-27, adaptive-A v2는 2026-10-30에 실행·자동 채점이 끝났다. 이번 문서 작성 중에는 새 모델 추론, 설정 튜닝, 추가 벤치마크를 실행하지 않았다. fixed-4의 Base/CoT/TT 각 600개, adaptive-A 600개 시도는 모두 종료 상태이며, adaptive-A의 오류 5개도 결과에 포함한다. 사람의 의미 검토는 남아 있다.

## 1. 참조 자료와 코드 버전

- 논문: Hypothesis-Driven Theory-of-Mind Reasoning for Large Language Models, [arXiv v2](https://arxiv.org/abs/2502.11881v2).
- 공식 코드 기준: [skywalker023/thought-tracing, 966eb2712714aa29b4d5d2395e39af94a2be90d7](https://github.com/skywalker023/thought-tracing/tree/966eb2712714aa29b4d5d2395e39af94a2be90d7).
- 결과와 재현 자료: [ybuser/reproducing의 thought_tracing 폴더](https://github.com/ybuser/reproducing/tree/master/thought_tracing).
- 로컬 공식 소스의 Git HEAD `98a5ca83be700cf548dd1149f2aec12e0e96b073`은 검증한 공식 ZIP을 가져온 로컬 import commit이다. 기준 upstream commit을 임의로 최신 버전으로 바꾼 것이 아니다. 기준 commit 객체 자체는 로컬 Git에 없으므로 존재하지 않는 객체와 Git diff를 했다고 주장하지 않는다. ZIP의 29개 경로에 대한 Git tree/blob 검증과 `records/source_provenance.json`을 함께 보존했다.
- fixed-4 공식 소스의 호환성 차이는 `records/official_compatibility.patch`에 있고, `agents/load_model.py`, `tracer.py` 수정 및 별도의 로컬 adapter/runner를 구분한다. 최종 답변 지시 통일과 호출별 생성 설정 반영도 논문 실행과의 차이이다. 공식 TT의 가중치·재표집 알고리즘 자체를 개선해서 성능을 올렸다는 주장은 하지 않는다.
- adaptive-A는 fixed-4가 실제 사용한 소스와 Qwen adapter의 독립 사본에서 구현했다. 원본 소스·프롬프트·설정 91개 파일의 SHA256이 실험 후와 이번 공개 작업 후에도 일치했다. 원래 환경에 패키지를 추가하거나 변경하지 않았고, 결과·중간 로그·쓰기 가능한 캐시는 분리했다.
- 최종 adaptive inference identity: `a54bdf6bd0e3d39f4e3c219253814f8b979e5293a5d3f68674e86ab94bdd5f01`. adaptive prompt SHA256: `670080ceccca488a15877acb126c2dacbd4e7886503444ef389f7610232fa6c0`.
- adaptive_a_v1은 운영 중 PID 재사용 보호 문제로 퇴역했다. 최종 비교는 v2만 사용한다. WDDM 메모리 계측·PID·NVML timeout 관련 운영 guard 수정은 별도 이력이며, 동결한 모델·프롬프트·알고리즘 identity와 구분한다.

## 2. 실제 환경과 모델

- Windows 11 네이티브, RTX A6000 48GB, Intel Xeon w7-3565X, RAM 128GB.
- Python 3.11.15, PyTorch 2.8.0+cu128, Transformers 4.57.1, NumPy 2.2.6, pandas 2.3.3. 설치 버전은 `requirements.lock.txt`에 고정했다.
- 모든 조건과 내부 단계의 모델: **Qwen/Qwen2.5-7B-Instruct**. Qwen2.5이며 Qwen3가 아니다.
- 모델과 tokenizer revision 모두 `a09a35458c702b33eeacc393d103063234e8bc28`.
- 로컬 모델 경로: `models/Qwen2.5-7B-Instruct/a09a35458c702b33eeacc393d103063234e8bc28/`.
- BF16, SDPA, batch size 1, 양자화 없음. 모델/tokenizer를 프로세스 안에서 한 번 로드해 재사용했다. `local_files_only=True`, `trust_remote_code=False`, 문맥 한도 32768. 입력 초과를 조용히 잘라내지 않고 기록/오류 처리한다.
- tokenizer chat template SHA256: `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`.
- 최종 답변은 모든 조건에서 temperature 0, 최대 출력 256토큰. CoT 중간 추론은 512토큰. TT/adaptive 내부는 해당 호출의 512/1024토큰 제한을 유지한다. adapter는 system prompt/history와 호출별 temperature/max_tokens를 실제 generate에 전달한다. decoding top_p=1.0, top_k=0, repetition_penalty=1.0이며, 아래의 **요약 top-p=0.9와는 다른 설정**이다.
- Python/NumPy/Torch/CUDA seed 정책은 global seed 20261020과 사례별 SHA256 기반이다. adaptive는 fixed TT와 같은 `global_seed|official600|row_id|TT`에서 사례 seed를 얻되 method/config/prompt/model revision을 캐시 identity에 별도로 포함한다. 호출 수와 RNG 소비가 달라져 같은 seed가 같은 추론 출력이나 하드웨어 간 완전한 재현을 보장하지 않는다.
- 외부 API/helper 모델로 자동 전환하지 않았다. W&B 및 외부 실험 기록 전송을 껐다. 로컬 추론이 Codex 자체의 모든 외부 통신을 없앤다는 뜻은 아니다.
- 모델 가중치·가상환경·인증정보는 GitHub에 올리지 않았다. 집 RTX 3080 10GB의 4비트 실행은 별도 조건 초안이며 이번 결과에 포함하지 않는다.

## 3. 데이터: 무엇을 600개 실행했는가

공개 `data/paraphrased_tomi/test`를 사용했다. 공식 loader가 읽은 전체 6004개 row에서 공식 코드의 `df.sample(n=2000, random_state=123)` 및 질문 유형별 quota 선택 루프를 재구성해 600개를 고정했다. 선택 seed 123과 추론 seed 20261020은 다르다.

구성은 1차 참 믿음 100, 1차 거짓 믿음 100, 2차 참 믿음 100, 2차 거짓 믿음 100, reality 100, memory 100이다. 원래 qtype의 `_tom`은 거짓 믿음, `_no_tom`은 참 믿음이며 0/1 세부 suffix와 원본 row ID를 보존했다. scenario_type과 qtype을 혼동하지 않는다.

- fixed manifest: `manifests/tomi_official_defaults600_v1.json`.
- manifest SHA256: `19b41683ef039b5a99e8684666d0dab1a49dd5d54ac2a737073a490c07d5594d`.
- 모델 입력 JSONL SHA256: `86e1548033a4e930b01700477049b53df164f982714becc54501332c9fa52d5d`.
- 원본 test.txt SHA256: `2c741e940736af097393245348cc1362102ec488450b63cf2c8190ac5a263c6c`.
- adaptive manifest는 추가 metadata 때문에 파일 해시가 `9a0d5077398d8affed75ed4993efdb436e538e2c3089357363258427526ac2c1`로 다르지만, 실제 600개 ID·순서·이야기·질문·모델 입력은 fixed와 동일하다. 새 600개를 다시 뽑지 않았다.
- 정답·qtype·row ID·채점용 위치 후보는 모델에 전달하는 prompt string에 넣지 않았다. 정답은 최종 출력 이후 채점에서 사용한다. 초기 개수·threshold·프롬프트를 fixed의 정오답으로 고르지 않았다.
- 초기 3개 smoke와 20개 파일럿 이후 사용자가 실행 범위를 600개로 늘렸다. 공식 기본 600개에는 과거 smoke row 3431과 pilot row 3730,337이 겹친다. 따라서 모든 600개가 이전에 전혀 보지 않은 검증 사례라고 주장하지 않는다. 새 실행/캐시로 진행했고 결과를 보고 문항을 교체하지 않았다. adaptive 전용 실제 모델 smoke는 본 600개 밖에서 별도로 수행했다.
- **저자가 실제 실행한 row ID manifest와 개별 prediction은 확보하지 못했다.** 공식 기본 선택을 복원한 600개라는 사실과 저자의 실제 표본과 동일하다는 주장을 구분한다. 이야기·템플릿 계열의 상관성을 무시하고 독립 표본이라고 하지 않는다.
- 공식 분기대로 belief 400개에서만 TT/adaptive 추적을 수행한다. reality/memory 200개는 추적을 생략하고 원래 문맥·질문으로 최종 답변한다. adaptive가 600개 모두에서 pruning을 수행한 것은 아니다.

## 4. 코드와 실행 구조

```text
official/                    공식 알고리즘·프롬프트·공개 데이터
repro/local_model.py         로컬 Transformers adapter, 호출/RNG/토큰/시간 기록
repro/tracing.py              공식 Tracer를 감싼 RecordingTracer
repro/run.py                  Base / CoT / TT 공통 최종 답변·채점 경로
scripts/run_official_tomi600.py  공식600 프로토콜 및 reality/memory 우회
configs/, manifests/         고정 설정·입력·평가 규칙
experiments/adaptive_a_v2/
  official/, repro/          실제 fixed 실행 소스/adapter의 독립 사본
  adaptive/kernel.py         목록 파서, 6단계 점수, softmax, pruning, 요약 선택
  adaptive/tracer.py         AdaptiveTracer(RecordingTracer), 단계별 분기와 ID 기록
  adaptive/prompts/          가변 초기 목록/state-only 평가 프롬프트
  adaptive/runner.py         사례 실행, 즉시 저장, 체크포인트·재개, 첫30개 보고서
  operations/               adaptive 전용 자원 admission/재개 supervisor
  records/, runs/, reports/  identity·검증·원시 로그·결과
```

위 구조는 **완전 백업을 복원한 뒤** 기준이다. GitHub 최상위 `thought_tracing/`에서는 핵심 소스·결과를 바로 읽을 수 있고, 독립 source snapshot과 전체 raw 로그·일부 스크립트는 분할 백업 안에 있다.

Base: 원래 이야기 + 질문 → 최종 답변.

CoT: 원래 이야기 + 질문 + 공식 중간 추론 지시 → 중간 추론 생성 → 같은 최종 답변 지시 → 최종 답변.

TT fixed-4: 대상 인물 식별 → 해당 인물 행동/발화 표시 및 state/action trajectory → perception 추론 → 초기 가설 4개 → 단계별 propagation → 행동이 있는 단계의 likelihood 평가/가중치 → 조건부 재표집 또는 재표현 → 가중치 기반 단계별 요약 → 원래 이야기·질문 + trace → 최종 답변.

**행동 표시 전처리는 TT/adaptive에만 있다. Base/CoT는 행동 태그를 주입받지 않는다.** 전처리는 전체 이야기/질문을 읽을 수 있으므로 전체 시스템을 완전한 온라인 추론이라고 부르지 않는다. 내부 perception 문장도 모델 추론 결과이지 실제 관측에 대한 정답 annotation이 아니다.

fixed의 가중치는 현재 평가 점수의 softmax로 **교체**된다. 이전 가중치를 likelihood와 곱하는 누적 Bayesian 업데이트는 실제 실행 경로에 없다. 행동이 없는 fixed 단계에서는 평가 갱신 없이 전달된다. `ESS=1/sum(w_i^2)`가 4/2=2보다 작으면 복원추출하고 균등 가중치로 재설정한다. 그 분기가 아닌 경우 lexical Jaccard 기반 다양성이 0.25 미만이면 재표현한다(`if` / `elif`). 텍스트 유사도가 의미적 다양성을 보장하지 않는다. 본 fixed 실행에서는 재표집 17회, 재표현 사건 2회가 기록됐다.

## 5. adaptive-A는 정확히 무엇을 바꿨는가

이번 방법은 **가변 초기 개수 + 단계별 개별 가중치 pruning + 재표집/재표현 제거 + state-only 평가 추가**라는 묶음이다. pruning 하나만의 ablation이나 원 논문 그대로의 TT라고 해석하지 않는다.

고정값:

```text
method = adaptive-A
min_hypotheses = 1
max_initial_hypotheses = 5
prune_threshold = 0.10
prune_comparison = strictly_less_than
resampling = disabled
rejuvenation = disabled
summary_top_p = 0.9
```

단계 순서:

```text
첫 trajectory 단계:
  기존 초기화 정보 범위 → 한 번의 목록 생성으로 1~5개 선택 → 유효 N개 채택 → 1/N
후속 trajectory 단계:
  살아 있는 각 ID의 가설을 기존 propagation으로 하나씩 갱신
모든 단계:
  현재 가설 평가 → 현재 점수 softmax → w<0.10을 한 번에 삭제
  → 생존 가중치 재정규화 → 생존 가설만 다음 단계
추적 완료:
  단계별 생존 가설의 top-p 0.9 요약 → 공통 최종 질문 답변
```

초기 목록 지시는 행동 유무에 따른 기존 질문 차이를 유지하면서 다음 의도를 추가했다. 개수를 결정하는 별도 LLM 호출은 없다.

> Generate a numbered list of between 1 and 5 plausible hypotheses about the target agent's thoughts. Choose the number based on the available evidence and uncertainty. Do not pad the list to reach five or invent unsupported details merely to create more hypotheses. Do not add any additional comments.

목록 파서는 실제 출력된 1~5개를 받도록 구현했다. 빈 목록·초과·감지된 형식 오류는 목록 형식 수정 재시도 최대 1회 후에도 실패하면 해당 사례 오류로 종료한다. 임의 채우기/중복 생성/다섯 개로 자르기는 하지 않는다. 이번 실제 400개 초기화에서는 목록 수정 재시도가 0회였고, 재시도/오류 종료 분기는 CPU mock으로 검증했다. 다만 **빈 항목 검출의 추가 경계 반례가 이번 감사에서 확인됐으므로**, 파서가 모든 비정상 입력을 거부한다고 주장할 수 없다. 아래6절을 참고해 줘.

가설 ID는 `row_id:h01` 같은 형식으로 고정한다. propagation은 ID를 유지하고 텍스트 버전만 바꾼다. 추가·분기·복제·통합·삭제 가설 복원은 하지 않는다. 과거 객체와 snapshot은 새 단계가 수정하지 않도록 분리한다. 삭제된 ID의 기록은 보존하지만 후속 추론 후보에 다시 넣지 않는다. 하나만 남아도 propagation과 평가를 계속하고 상대 가중치는 1이다.

평가와 점수:

- 행동/발화가 있는 단계: fixed의 likelihood 프롬프트·응답 option parser·점수 매핑을 재사용한다.
- 행동/발화가 없는 단계: 새 state-only plausibility 프롬프트를 사용한다. 이전 가설, 갱신 가설, 현재 단계까지의 인물 관점 정보와 perception을 제공한다. 실제 세계 상태와 믿음을 구분하고, 접근하지 못한 정보의 앎을 가정하지 않으며, 세계와 다르다는 이유만으로 정당한 false belief를 벌점 주지 않도록 명시한다. 새 정보가 없으면 기존 믿음/불확실성을 유지할 수 있게 한다. 정답이나 별도 후속 사건을 추가하지 않는다. 다만 기존 전처리/propagation 결과에 잘못된 정보가 들어오는 가능성은 의미 검토 대상이다.
- 두 evaluator의 형식은 a,b,c,d,e,f이고 raw score는 각각 **3,2.5,2,1,0.5,0.001**이다. 실제로 90%/70%를 확률값으로 직접 쓰는 것이 아니다.
- `u_i = exp(s_i) / sum_j exp(s_j)`; 이전 `w_i`와 곱하지 않는다. 이전 가설 내용은 propagation과 state-only 평가에 들어가지만, 이전 숫자 가중치는 현재 softmax를 누적 보정하지 않는다.
- `K={i:u_i >= 0.10}`를 한 번 정한다. 정확히 0.10이면 유지한다. 생존 가중치는 `w_i=u_i/sum_{j in K}u_j`. 재정규화 후 다시 pruning하지 않는다.
- 일반 threshold에서 K가 비면 삭제 전 최대 가중치 하나를 남긴다. 동률은 기존 순서 우선. NaN/길이 불일치 등은 낮은 점수로 숨기지 않고 오류로 처리한다.
- 요약 top-p는 정렬한 생존 가설의 누적 가중치 0.9에 대한 선택이다. **개별 가중치 threshold pruning과 다르며**, 요약에 선택되지 않은 가설도 생존할 수 있다.
- 행동 likelihood와 state-only plausibility는 같은 매핑을 사용해도 같은 의미로 보정된 확률이 아니다. 가중치는 후보 집합 내 상대값이며 가설의 참일 절대 확률이 아니다.

관련 코드: `adaptive/kernel.py`의 `parse_initial_list:18`, `parse_option:36`, `softmax_scores:45`, `prune_once:56`, `summary_indices:73`; `adaptive/tracer.py`의 `initialize:33`, `state_only_evaluation:69`, `trace:85`. 이는 현재 파일의 위치이며 변경 후에는 함수명을 우선 확인해 줘.

## 6. 제대로 구현했다고 말할 수 있는 근거와 한계

실행 전 CPU mock **19개 통과**, 실패/오류 0, LLM 호출 0, CUDA 초기화 없음(`experiments/adaptive_a_v2/records/cpu_validation.json`). 검증 내용은 초기 1~5 파싱/최대 1회 수정/오류 종료, threshold 경계와 최소 1개·동률, softmax의 이전 가중치 교체, 단조 감소와 삭제 ID 재등장 금지, 텍스트·ID·점수 대응, 과거 객체 불변성, evaluator 분기, resampling/rejuvenation 금지, 완료 사례 재개 중복 방지, 첫30개 보고서 저장 후 계속 진행 등이다.

본 600개 밖의 실제 모델 smoke로 연결·형식·false belief와 state-only 단계를 확인했다. 첫 30개 보고서 초안도 저장한 뒤 나머지로 진행했다. smoke나 정답률을 보고 threshold를 바꾸지 않았다.

2026-11-04에는 모델을 다시 돌리지 않고 최신 `case_attempt_id`의 저장 로그를 독립적인 표준 라이브러리 검사기로 재검사했다. 결과:

- 600개 종료 결과: 정상 595, 오류 5. 추적 초기화 400개, 공식 우회 200개.
- **완료된 단계 기록 908개**: 행동 평가 단계 573, state-only 단계 335. 이 수치는 오류 전까지 기록된 부분 단계도 포함하며 모든 400개 추적이 완주했다는 뜻은 아니다.
- 각 기록에서 ID/텍스트/점수 대응, 이전→현재 ID 집합 및 가중치 전달, 현재 score softmax, `<0.10` 한 번 삭제, 생존 재정규화, 최소 1개, 과거 삭제 ID의 미복원을 확인했다. 요약 선택 기록 900개에서도 별도 top-p 목록을 확인했다. 최신 trace에는 resample/rejuvenate 사건이 없었다.
- 실제 400개 초기 목록과 908개 완료 단계 기록에 빈 가설 텍스트는 없었다.
- 원래 fixed 보호 파일 91개 해시가 여전히 일치했다.
- 감사 원본은 로컬 `records/publication_20261104_chunked_v1/adaptive_log_audit.json`에 저장했다. 이번 인계문에는 그 수치와 범위를 반영했다. 과거 보고서를 덮어써서 결과를 바꾸지 않았다.

추가 최소 CPU 검증에서 다음 **실제 파서 경계 문제**를 재현했다. `raw = "1.   " + chr(10) + "2. Valid hypothesis."`를 만들어 `parse_initial_list(raw)`를 호출하면 반환값은 `["", "Valid hypothesis."]`가 된다. 마지막 항목/단독 항목의 공백은 거부하지만, 목록 중간 항목은 정규식이 공백을 포착한 뒤 strip으로 빈 문자열이 되어도 추가 검사가 없어 통과할 수 있다. 이는 정적 의심만이 아니라 동결된 순수 CPU 함수에 입력해 확인한 반례다. 기존19개 test의 통과만으로 이 경우까지 검증됐다고 할 수 없다. 이번 실제 실행의 저장된 가설에는 이 빈 항목이 없어서 해당 반례가 600개 점수에 영향을 주었다는 증거는 없다. 기록은 로컬 `records/publication_20261104_chunked_v1/parser_edge_checks.json`에 남겼고, 동결 코드와 원래 점수는 수정하지 않았다.

따라서 **관측된 실제 실행의 핵심 제어 규칙이 소스·mock·저장 로그와 일치한다는 근거는 있지만, 모든 입력에 대한 구현 완전성을 주장할 수 없다.** 빈 항목 검출은 확인된 구현의 경계 결함으로 남긴다. 모델이 적절한 N을 골랐는지, perception이 옳았는지, propagation이 대상 물건과 중첩 믿음을 보존했는지, evaluator가 올바르게 평가했는지도 이 검사로 증명되지 않는다. 저장 로그 검사는 메모리 객체의 불변성을 직접 증명하지 않으며 그 속성은 mock 검증 근거로 구분한다. 사람의 실제 사고 과정을 재현/입증했다고도 주장하지 않는다.

## 7. 채점 규칙과 전체 결과

공식 점수는 **최종 prediction에서 정답 문자열 substring**을 대소문자 무시로 찾는다. 중간 trace/CoT 출력은 직접 채점하지 않는다. 별도 final-only 규칙은 파일럿 출력 확인 전에 고정했으며, 원문에 있는 위치 후보를 검사해 명확한 단일 답변만 자동 판정한다. 복수 위치·상충·모호함·복수 문장 등은 사람 검토(null)로 남기고 자동 정답에 포함하지 않는다. 단순 어휘 규칙이므로 의미 판정의 완전한 정답 oracle은 아니다. 별도 점수를 이미 사람이 모두 확정한 정확도라고 부르지 마.

| 조건 | 공식 정답 /600 | 공식 정확도 | 별도 자동 정답 /600 | 실행 실패 | 미완료 |
|---|---:|---:|---:|---:|---:|
| Base |378|63.00%|344 (57.33%)|0|0|
| CoT |434|72.33%|379 (63.17%)|0|0|
| TT fixed-4 |374|62.33%|347 (57.83%)|0|0|
| adaptive-A v2 |374|62.33%|342 (57.00%)|5|0|

실패도 분모 600에 포함했다. CoT 내부 호출 잘림 1건은 최종 실행 실패와 구분한다. adaptive는 잘림 0건이다. 공식 점수와 별도 자동 정답 수가 다른 것은 채점 정책 차이도 포함하므로 그 차이 전부를 추론 오류 수라고 해석하지 않는다.

| 유형: 각 100문항, 공식 정답 수 | Base | CoT | fixed-4 | adaptive-A |
|---|---:|---:|---:|---:|
| 1차 참 믿음 |73|87|86|84|
| 1차 거짓 믿음 |38|35|35|38|
| 2차 참 믿음 |77|70|80|81|
| 2차 거짓 믿음 |31|45|14|12|
| reality |62|98|62|62|
| memory |97|99|97|97|

**믿음 400개만** 보면 공식 정답은 Base 219/400(54.75%), CoT 237/400(59.25%), fixed-4 215/400(53.75%), adaptive-A 215/400(53.75%)이다. 별도 자동 정답 수는 각각 196,208,199,194다.

양쪽 모두 정상 완료한 **595개 paired 비교**에서는 fixed-4 371/595(62.35%), adaptive-A 374/595(62.86%)다. fixed만 정답 26개, adaptive만 정답 29개다. 별도 자동 정답은 paired fixed 344, adaptive 342다. 전체 시도 600과 정상 paired 595를 혼용하지 않는다. 같은 전체 정답 수라도 맞힌 사례 집합은 다르다. 이 결과로 adaptive의 통계적 우월성이나 일반적인 향상을 주장하지 않는다.

## 8. adaptive-A 형식 오류 5건: 확인한 원인

row ID **5416,5788,2299,5133,218**은 모두 `FormatError: Unparseable six-option evaluation`이다. 이번에 원시 호출을 확인하니 **모두 state-only 평가 응답에서 option 문자를 파싱하지 못한 경우**였다. 초기 가설 목록 형식 오류, NaN 또는 threshold 오류라고 기록하지 마.

예를 들어 row 5416의 실제 출력 시작은 다음과 같다.

```text
Answer: Somewhat Likely (Around 60%)
```

여기에 설명은 있지만 `(c)` 같은 option 문자가 없다. 기존 문자 기반 parser에서는 이것을 c로 자동 추정하지 않는다. row 5133과 5788에도 같은 첫 줄이 확인됐고, row 218과 2299도 해당 평가 batch에 파싱 불가능한 응답이 있었다. 이 호출들은 잘림 없이 EOS로 종료됐다. 따라서 해당 5건은 우선 **새 evaluator의 출력 형식 준수/파싱 실패**로 분류할 수 있다. 이것만으로 내용 평가가 틀렸다고 단정할 수 없다. 실패 전 propagation에 잘못된 사실/대상 물건 혼동이 있는지는 별도 검토가 필요하다.

원래 fixed likelihood 코드에는 파싱 실패를 낮은 기본 점수로 바꾸는 경로가 있지만 adaptive는 option을 명시적으로 재검사해 실패 처리한다. 이번 5건은 fixed에는 없는 state-only 경로에서 발생했다. 이 차이도 비교 조건에 드러내야 하며, 보고서 작성 중 parser를 바꿔 원래 run의 실패를 성공으로 덮어쓰지 않았다.

## 9. 가설 개수와 pruning 관측

- 초기 N 분포(400개): N=1:49개, N=2:103개, N=3:224개, N=4:19개, N=5:5개. 초기 평균 **2.57**.
- 실제 pruning 발생 **96/400=24%**; 전체 시도 기준 96/600=16%.
- 처음부터 또는 이후 1개가 된 사례 **72/400=18%**. 처음부터 1개였던 49개와 구분한다.
- 첫 pruning 단계(0부터): 0단계59개, 1단계27개, 2단계8개, 3단계2개.
- 삭제된 가설: 행동 평가에서102개, state-only에서9개. 총111개.
- 기록된 전체 단계의 삭제 전 평균2.5154, 삭제 후 평균2.3932. 요약에 사용한 평균2.39개.
- 기존 자동 보고서의 `mean_final_count=2.2925`는 400개에서 **오류 사례의 마지막 부분 상태도 포함한 마지막 기록 개수 평균**이다. 정상 완료한 추적395개만의 실제 최종 평균은 이번 재검사에서 **2.2886075949**였다. 두 분모를 구분해 줘.
- 단계0/1/2/3/4에 기록이 있는 사례 수는400/295/172/38/3이다. 단계별 평균은 도달한 서로 다른 부분집합 기준이므로 전체의 시간적 감소 곡선으로 단순 해석하지 않는다.
- normalized weight는 후보 수에 의존한다. 한 번 삭제하면 복구할 수 없고 N=1이면 항상1이다. 모든 후보가 좋지 않은 상황도 상대 가중치만으로 판별할 수 없다. 초기 N 선택의 합리성은 분포만으로 입증되지 않는다.

## 10. 계산량·실행 시간·GPU

본 600개에 대한 실제 모델 요청(모든 내부 단계 포함):

| 조건 | 호출 수 | 입력 토큰 | 출력 토큰 | 토큰 합 | 호출 wall time 합 |
|---|---:|---:|---:|---:|---:|
| Base |600|88,629|4,629|93,258|319.661초|
| CoT |1,200|326,529|168,235|494,764|10,284.969초|
| fixed-4 |8,813|3,113,752|777,841|3,891,593|45,529.266초|
| adaptive-A |8,011|2,926,875|636,593|3,563,468|30,099.930초|

- fixed 전체 Base/CoT/TT 및 connectivity 포함 10,614호출, 4,479,653토큰, 실행 경과 약16.10시간.
- adaptive smoke/connectivity는 별도55호출/24,042토큰. worker 활성 시간 약9.02시간은 로딩/smoke를 포함하고 queue 대기는 제외한다. 달력상 중단·재개에 걸친 며칠을 순수 계산 시간이라고 하지 않는다.
- adaptive의 완료 단계 기록상 가설 평가2,284회와 실제 시도 중 평가 생성 요청2,297회는 다르다. 평가 요청은 행동1,503회 + state-only794회다. 실제 전체 모델 호출8,011회와 adapter batch 인터페이스 호출1,833회도 서로 다른 단위다. adapter는 batch 인터페이스를 순차 생성으로 처리한다.
- 자원 중단으로 실패한 모델 호출2회는 최종 사례 형식 오류5건과 다른 지표이다. 중단 직전 부분 출력17토큰은 집계했고, 복구 불가능한 실패 생성 토큰은 미상일 수 있다.
- fixed와 adaptive GPU 실행의 겹침은 기록에서 없었다. 두 모델을 같은 GPU에서 동시에 돌려 fixed를 방해하지 않았다.
- Windows WDDM의 장치 전체 NVML peak44,108MiB와 CUDA mem_get_info 기준18,015MiB는 다른 측정이다. PyTorch allocator peak는 fixed16,358,994,432바이트, adaptive16,171,606,528바이트로 별도 기록했다. 다른 프로그램/GPU paging/운영 재개 차이도 있을 수 있다. `reports/RESOURCE_SUMMARY_SEPARATE_20261030.json`을 우선하며 혼합 peak 하나로 알고리즘 메모리나 속도 향상을 단정하지 않는다.
- 같은 모델을 사용했어도 Base/CoT/TT/adaptive의 계산량은 같지 않다. wall time 감소를 pruning 단독의 인과 효과라고 하지 않는다.

## 11. 논문 결과와 비교할 때의 범위

기존에 논문 v2 Table2/6 및 Appendix B와 대조해 정리한 기록 기준이다. 논문 Qwen 열은 **Qwen2.5-72B**, ToMi 행동 표시는 **GPT-4o-2024-08-06**이다. 로컬은 그 두 역할 모두 Qwen2.5-7B로 대체했다. 논문에는 GPT-4o, Llama3.3-70B, Gemini1.5-Pro도 있다. 아래 논문 값은 공개된 반올림 수치이며 저자 원시 prediction을 다시 채점한 값이 아니다.

| 방법 | 논문 Qwen72B Table2 Avg | 로컬7B 믿음400 공식 |
|---|---:|---:|
| Base |57.3%|54.75%|
| CoT |67.8%|59.25%|
| TT |74.3%|53.75%|

Table2 Avg가 참/거짓 믿음 평균과 일치하고 공식 코드도 belief 집계를 저장하므로, 기존 비교는 이를 **믿음400에 대응시키는 해석**을 사용했다. 논문 본문의600문항 설명과 표 분모의 관계는 저자에게 확인받지 못했다. 전체600 점수를 논문 Avg와 곧바로 비교하지 마.

로컬 TT는 전체600 또는 믿음400의 공식 점수에서 Base보다4개 낮을 뿐이며, 별도 자동 채점에서는 Base보다3개 높다. 따라서 “Base와 CoT보다 모두 크게 낮다”는 표현은 정확하지 않다. CoT와는 전체600에서60개 차이인데 그중38개는 추적하지 않는 reality/memory의 차이이고, 믿음400에서22개 차이다. 특히 2차 거짓 믿음은 CoT45/100 대 TT14/100(adaptive12/100)이다. 논문 TT의 같은 유형 공개 수치는61%다.

모델 차이만으로 TT가 더 어려울 수 있다는 **가능한 설명**은 있다: 대상 식별·행동 표시·perception·후보 생성/전파·평가·요약의 여러 생성 결과를 다시 입력으로 쓰므로 작은 모델의 초기 오류가 누적될 수 있고, 자기가 만든 가설의 자기 평가가 틀린 가설을 잘 구별하지 못할 수 있다. 2차 믿음이 단일 인물/관심사로 축약될 수도 있다. 그러나 모델 크기, 행동 표시 모델, 최종 지시, backend, 표본 동일성 등 여러 조건이 함께 달라져 원인을 모델 크기 하나로 확정할 수 없다. 새 controlled ablation을 실행하지 않았다.

이미 검토 후보로 기록한 row5979의 행동/문맥 분리, row4101의 2차 믿음 보존, 정답이어도 중간 내용이 의심되는 row976을 원시 로그로 확인할 수 있다. annotation의 단순 문자열 불일치나 같은 평가 label 반복을 곧바로 의미 오류율로 바꾸지 않는다. 정적 코드 의심도 실행 확인 없이 확정 버그라고 부르지 않는다.

## 12. 저장 자료를 읽고 다른 PC에서 이어가는 법

GitHub `thought_tracing/`의 핵심 자료:

- `reports/FINAL_600_COMPARISON_KO.md`, `FINAL_600_SUMMARY.json`: 최종 전체 비교. 과거 보고서의 모호한 평균 설명은 이 문서9절의 추가 확인을 함께 읽어 줘.
- `reports/bf16_official600_v1/{summary.json,predictions.jsonl,REPORT.md}`.
- `experiments/adaptive_a_v2/reports/{comparison.json,predictions.jsonl,HUMAN_REVIEW.json}`.
- `reports/LOCAL7B_VS_PAPER_QWEN72B_20261027_KO.md`, `WEIGHT_UPDATE_AND_PAPER_MATCH_KO.md`.
- `handoff/CHATGPT_HANDOFF_FIXED4_20261027_KO.md`: 이전 fixed 중심 인계문. 당시 진행 중 표기보다 이번 최종 상태를 우선한다.
- 전체 원시 호출·prompt·output·RNG·중간 trace·체크포인트·독립 소스 사본은 **20조각 백업** 안에 있다. 총72,250,742바이트의 원래 ZIP SHA256은 `d2e157b3b7e53ca234aaeb9622481780d09fc4bdd7877d4e8a0c7088c992893a`이다.

저장소를 clone하거나 Download ZIP으로 받은 뒤 `thought_tracing/`에서, Python3.11 이상만으로 검증·복원한다. 모델/GPU/유료 서비스는 필요 없다.

```powershell
py -3.11 tools/restore_snapshot.py verify
py -3.11 tools/restore_snapshot.py restore
Set-Location .restored/full_snapshot/thought_tracing
py -3.11 tools/backup.py verify
py -3.11 tools/backup.py restore-logs
```

원시 사례 경로는 복원본의 `runs/bf16_official600_v1/official600/row-XXXX/{Base,CoT,TT}/` 및 `experiments/adaptive_a_v2/runs/adaptive_a_v2/cohort/row-XXXX/`이다. `calls.jsonl`, `trace_events.jsonl`, `result.json`, `checkpoint.json`에서 확인한다. 기존 결과는 완료됐으므로 옛 queue/supervisor를 다시 시작할 필요가 없다. 환경 설치/모델 고정 다운로드/새 run 절차는 복원본 `CONTINUE_ON_ANOTHER_PC_KO.md`에 있다.

공개 사본에서는 사용하지 않는 upstream MMToM 스크립트의 인증정보 형태 문자열만 제거했고 위치와 전후 해시를 `records/publication_redactions.json`에 남겼다. 이 파일의 과거 frozen 해시와 공개 사본이 다른 것을 TT 코드 변경으로 오해하지 마. 모델·환경·인증정보·사내 보안 설정·논문 PDF와 자동 종료 예약은 저장소에 포함하지 않았다.

## 13. 남은 검토와 요청하는 답변

fixed TT/adaptive 비교에서 사람 검토 대기는 **109개 method-example 항목**(fixed49 + adaptive60, 오류5 포함)이다. 109개의 고유 질문이라는 뜻은 아니다. fixed Base/CoT/TT 전체의 검토172개와 범위도 다르다. 중간 추론의 전수 의미 검토는 아직 끝나지 않았다. 원시 파일을 실제로 열지 못했다면 열어 검증한 것처럼 말하지 마.

다음 순서로 답변해 줘.

1. 내가 이해하기 쉬운 말로 현재 재현 구조와 fixed/adaptive 차이를 설명해 줘. 구조와 데이터 흐름을 먼저 정리해 줘.
2. adaptive-A를 요청대로 구현했다고 말할 수 있는 부분과 아직 검증하지 못한 부분을 분리해 줘. 코드/검증/로그의 증거 수준을 명확히 해 줘. 실행 성공이나 정답률만으로 의미적 타당성을 보장하지 마.
3. 결과를 전체600/믿음400/paired595와 공식/별도 채점으로 나눠 해석해 줘. 실패5건과 2차 거짓 믿음의 낮은 점수, pipeline의 오류 누적 가능성을 근거와 가설로 구분해 줘.
4. 저자인 김현우 교수에게 드릴 기술 질문을 **최대3개**로 좁혀 줘. 각 질문에 논문 절·코드 함수·기대/관측·최소 로그·대안 설명·추가 확인을 붙여 줘. 후보는 (a) ToMi 행동 표시/관측 전처리를 작은 모델로 바꿀 때 필요한 형식·의미 검증(Appendix B, `label_action`/`preprocess_input`), (b) 2차 믿음을 trace/요약에서 유지하는 범위(방법 절의 hypothesis propagation/aggregation, `propagate`/`weighted_average_hypotheses`), (c) §3 Table2 Avg의 분모 및 실제 실행600 row manifest다. 논문의 기존 perception/weight/self-correction ablation을 새 기여로 제안하지 마. 교수에게 직접 발송하거나 인턴 문의와 섞지 마.
5. 다음 작업 **하나**만 추천해 줘. 우선 row5416의 state-only 형식 실패와 정상 종료한 2차 거짓 믿음 사례 하나를 원시 로그로 대조해 단계별 사람 검토를 남기는 작업이 적합한지 판단해 줘. 새 실험이나 프롬프트/threshold 변경은 자동 실행하지 마. 개선안을 꼭 제시한다면 관측된 문제에 대응하는 하나만 고르고, 수정에 쓰지 않은 새 사례로 검증하는 방법을 붙여 줘.

관측 여부만 바꾼10쌍의 추가 실험은 `design/observation_pairs10.json`에 **설계만** 있다. 물리 사건/실제 위치를 유지하고 관측에 따라 정답 믿음이 달라지는지, 후속 대화/재관측과 모순이 없는지 실행 전에 사람이 검토해야 한다. 두 조건을 모두 맞힌 쌍 비율과 실패 단계가 제안 지표다. 이번 결과에 해당 추가 실험을 실행한 것처럼 포함하지 마.
