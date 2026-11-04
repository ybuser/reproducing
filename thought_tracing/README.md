# ThoughtTracing 재현 기록

fixed-4의 Base/CoT/TT 각600문항과 adaptive-A 600문항을 로컬 Qwen2.5-7B-Instruct로 실행한 기록이다. GPU 미완료 문항은 없고 사람의 의미 검토는 남았다.

| 조건 | 공식 정답 /600 | 최종 답변 별도 채점 /600 | 실행 실패 |
|---|---:|---:|---:|
| Base |378|344|0|
| CoT |434|379|0|
| TT fixed-4 |374|347|0|
| adaptive-A |374|342|5|

논문 전체 모델/벤치마크 재현이나 pruning 단독 효과를 주장하지 않는다. 공식 기본 선택 절차로 재구성한600개이며 저자 실제 실행 row ID와의 동일성은 미확정이다. 원래 영어 데이터·프롬프트·출력은 번역하지 않았다.

## 읽을 자료

- [최종 전체 비교](reports/FINAL_600_COMPARISON_KO.md), [기계 판독 요약](reports/FINAL_600_SUMMARY.json)
- [재현 구조·오류 근거·교수 질문 초안](handoff/CHATGPT_HANDOFF_FIXED4_20261027_KO.md)
- [학습용3사례 안내](reports/LEARNING_GUIDE_KO.md), [사람 검토 목록](experiments/adaptive_a_v2/reports/HUMAN_REVIEW.json)

이 폴더에서는 핵심 코드·설정·manifest·결과를 바로 볼 수 있다. **완전한 원시 로그/공개 데이터/과거 버전/재개 안내는 분할 백업을 복원해 사용한다.** 현재 업로드 완료 여부는 [UPLOAD_STATUS.json](UPLOAD_STATUS.json)을 확인한다.

## 완전한 백업 복원 (Python3.11 이상, GPU/추가 패키지 불필요)

약72.3MB의 검증된 ZIP을20조각(각약3.62MB)으로 나눴다. 두 조각씩10개 데이터 커밋, 핵심 자료1개, 추가 요청한 결과 전달 프롬프트1개로 총12개 커밋을 사용했다. 각 데이터 커밋은 base64 인코딩까지 합쳐10MB 미만이다.

```powershell
py -3.11 tools/restore_snapshot.py verify
py -3.11 tools/restore_snapshot.py restore
Set-Location .restored/full_snapshot/thought_tracing
py -3.11 tools/backup.py verify
py -3.11 tools/backup.py restore-logs
```

기존 작업을 덮어쓰지 않는다. 복원본의 CONTINUE_ON_ANOTHER_PC_KO.md에 설치·모델 다운로드·새 실행 절차가 있다. 모델 가중치/가상환경/인증정보는 포함하지 않는다. 모델이 필요하면 공식Hugging Face의 고정revision을 별도로 받는다. 새PC의 추론은 새run ID를 사용한다. 3080 10GB의4비트는 별도 조건이다.

## 버전·차이

논문: https://arxiv.org/abs/2502.11881v2

공식기준: 966eb2712714aa29b4d5d2395e39af94a2be90d7. 원래 로컬import commit은98a5ca83be700cf548dd1149f2aec12e0e96b073이며 기준을 바꾼 것이 아니다. 모델/토크나이저: Qwen/Qwen2.5-7B-Instruct revision a09a35458c702b33eeacc393d103063234e8bc28. Windows11/RTX A6000/BF16/SDPA/batch1/비양자화, Python3.11.15/Torch2.8.0+cu128/Transformers4.57.1.

논문의72B와 GPT-4o 행동 표시 대신7B가 모든 단계를 맡았다. TT/adaptive만 행동 표시·관측 전처리를 거치며 reality/memory200개는 추적을 생략한다. adaptive-A는 가변초기1~5개+매단계<0.10 삭제+재표집/재표현 제거+state-only 평가를 함께 포함한다. 가중치는 현재 점수softmax로 교체하는 후보 집합 내부 상대값이다.

사용하지 않는 upstream MMToM 스크립트의 인증정보 형태 문자열은 출판 사본에서 제거했다. 원래 실험 파일은 그대로 보존했고 위치/전후hash만 records/publication_redactions.json에 있다. 공식 MIT LICENSE와 출처를 보존했다. PDF/전체논문 본문·보안 설정·자동 종료 작업은 저장소에 넣지 않았다.

ChatGPT 메인 채팅에 전달할 [실험 결과 프롬프트](handoff/CHATGPT_MAIN_HANDOFF_20261104_KO.md)도 마지막 커밋에 포함한다.
