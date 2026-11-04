# 고정 manifest 메타데이터 설명 정정

선택된 사례·정답·4층 분류·선택 seed·이야기/원본 파일 hash·manifest 자체 hash는 유지한다. 실제 모델 출력은 아직 없다.

Windows text write의 LF→CRLF 변환 전에 메모리 문자열로 계산한 companion `.sha256` 값이 실제 파일 바이트와 달랐다. JSON 파일 바이트를 바꾸지 않고 companion 값만 실제 파일 SHA256으로 정정했다. 이전·수정 값과 원인은 `records/manifest_hash_correction.json`, 전체 검증은 `records/manifest_validation.json`에 저장한다. 실제 모델 실행 전에 발견·수정했으며 표본 교체가 아니다.

초기 JSON의 `question_type_definition`에 쓴 **“0/1 are agent indices”**는 trace event의 agent_0/agent_1과 같은 ID를 뜻한다고 읽으면 부정확하다. 예를 들어 점검 row 1906은 `first_order_1_no_tom`이지만 질문 대상 Jayden은 실제 이동자이고 trace에서는 `agent_0_moves_obj` 역할에 해당한다. row 3431은 `first_order_0_tom`이지만 Nathan의 퇴장은 trace에서 `agent_1_exits`다.

확인된 사실은 공식 scorer가 `_tom`과 `_no_tom` subtype을 각각 false/true-belief 그룹으로 합친다는 것이다. 0/1은 원래 subtype으로 보존하고, 실제 대상 인물은 질문으로 정한다. generator의 index 정의를 검증한 것으로 주장하지 않는다. 생성기 원문 호스트가 정책 차단되어 상세 의미는 후속 확인 항목으로 남겼다. 현재 선택·실행 로직은 suffix 숫자로 대상을 추론하지 않으므로 이 설명 정정으로 모델 입력이나 4층 구성이 달라지지 않는다.

두 번째 유의점: `test.trace` 마지막 필드의 scenario_type과 끝에서 두 번째 질문 qtype은 다른 층위다. row 2097은 scenario_type=`true_belief`이면서 질문은 `second_order_0_tom`이다. Charlotte 본인의 물건 위치 믿음이 맞더라도, Mila가 가진 옛 위치 믿음을 묻는 2차 질문은 공식 false-belief 질문 그룹에 들어간다. 바깥 인물이 다른 사람의 믿음을 잘못 추정한다는 뜻으로 일괄 해석하지 않는다.
