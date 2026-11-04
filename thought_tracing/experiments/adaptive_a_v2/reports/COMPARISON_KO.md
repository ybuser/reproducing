# fixed-4 / adaptive-A 비교

상태: complete. 현재 로컬 Qwen 7B의 고정 600문항이며 논문 전체 평가를 재현한 것이 아니다.

표본: User-confirmed extension to all 600 rows of the existing fixed600 manifest

| 조건 | 시도/전체 | 공식 정답 | 공식 정확도 | 별도 정답 | 실행 실패 | 형식 실패 | 미완료 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed-4 | 600/600 | 374/600 | 62.33% | 347/600 | 0 | 0 | 0 |
| adaptive-A | 600/600 | 374/600 | 62.33% | 342/600 | 5 | 5 | 0 |

| 하위 유형 | fixed-4 공식/별도 | adaptive-A 공식/별도 |
|---|---|---|
| memory | 97/100 · 93/100 | 97/100 · 93/100 |
| order1_false | 35/100 · 32/100 | 38/100 · 35/100 |
| order1_true | 86/100 · 80/100 | 84/100 · 77/100 |
| order2_false | 14/100 · 11/100 | 12/100 · 9/100 |
| order2_true | 80/100 · 76/100 | 81/100 · 73/100 |
| reality | 62/100 · 55/100 | 62/100 · 55/100 |
| belief | 215/400 · 199/400 | 215/400 · 194/400 |
| false_belief | 49/200 · 43/200 | 50/200 · 44/200 |
| true_belief | 166/200 · 156/200 | 165/200 · 150/200 |
| order1 | 121/200 · 112/200 | 122/200 · 112/200 |
| order2 | 94/200 · 87/200 | 93/200 · 82/200 |
| qtype:first_order_0_no_tom | 42/50 · 40/50 | 41/50 · 37/50 |
| qtype:first_order_0_tom | 17/50 · 16/50 | 18/50 · 17/50 |
| qtype:first_order_1_no_tom | 44/50 · 40/50 | 43/50 · 40/50 |
| qtype:first_order_1_tom | 18/50 · 16/50 | 20/50 · 18/50 |
| qtype:memory | 97/100 · 93/100 | 97/100 · 93/100 |
| qtype:reality | 62/100 · 55/100 | 62/100 · 55/100 |
| qtype:second_order_0_no_tom | 45/50 · 42/50 | 43/50 · 38/50 |
| qtype:second_order_0_tom | 6/50 · 5/50 | 7/50 · 6/50 |
| qtype:second_order_1_no_tom | 35/50 · 34/50 | 38/50 · 35/50 |
| qtype:second_order_1_tom | 8/50 · 6/50 | 5/50 · 3/50 |

양쪽 정상 완료 paired: 595개. fixed-4만 정답 26개 [2760, 1280, 4230, 4735, 3569, 5483, 1244, 2720, 1882, 1161, 61, 3641, 4168, 3613, 3094, 5028, 905, 3209, 3932, 1974, 319, 4012, 5357, 3385, 2109, 207]; adaptive-A만 정답 29개 [2696, 5064, 477, 5363, 3557, 1661, 540, 5539, 5138, 5998, 45, 5979, 3821, 3145, 5059, 2401, 3338, 2754, 3716, 2073, 3694, 2217, 5566, 1633, 2484, 5762, 192, 3431, 1074].

가설 통계(초기화가 실제 수행된 사례 기준; 사실/기억 우회는 제외):

```json
{
  "initialized_examples": 400,
  "initial_count_distribution": {
    "4": 19,
    "3": 224,
    "2": 103,
    "1": 49,
    "5": 5
  },
  "mean_initial_count": 2.57,
  "mean_final_count": 2.2925,
  "mean_pre_prune_count_over_steps": 2.515418502202643,
  "mean_survivors_over_steps": 2.393171806167401,
  "mean_summary_hypothesis_count": 2.39,
  "pruned_examples": 96,
  "pruned_example_ids": [
    2760,
    1280,
    1001,
    2087,
    799,
    5770,
    5496,
    2083,
    382,
    4760,
    5666,
    2095,
    2696,
    477,
    4896,
    5363,
    558,
    4300,
    2382,
    4330,
    3388,
    1567,
    2968,
    5416,
    599,
    3056,
    540,
    5138,
    2808,
    5018,
    5037,
    4441,
    4457,
    724,
    1581,
    2711,
    1161,
    4998,
    4066,
    61,
    2942,
    4743,
    2663,
    872,
    5215,
    614,
    2910,
    4356,
    2401,
    3479,
    3613,
    4744,
    3554,
    5859,
    4894,
    2906,
    2850,
    2792,
    5903,
    774,
    2758,
    3209,
    3716,
    188,
    984,
    2168,
    1084,
    337,
    2073,
    2881,
    1459,
    1607,
    4372,
    218,
    1917,
    5370,
    2718,
    5535,
    2865,
    4523,
    5734,
    1340,
    5357,
    537,
    3125,
    2291,
    2482,
    245,
    3912,
    192,
    5490,
    3468,
    207,
    4371,
    3431,
    5330
  ],
  "pruned_fraction_of_initialized": 0.24,
  "pruned_fraction_of_all_attempted": 0.16,
  "first_prune_step_distribution_zero_based": {
    "1": 27,
    "2": 8,
    "0": 59,
    "3": 2
  },
  "became_or_started_single_examples": 72,
  "became_or_started_single_fraction_of_initialized": 0.18,
  "single_example_ids": [
    2204,
    2087,
    799,
    5496,
    4786,
    4760,
    9,
    5823,
    2704,
    380,
    4735,
    4300,
    2968,
    5416,
    5939,
    5018,
    5922,
    4675,
    4559,
    1882,
    2570,
    1777,
    1446,
    4466,
    3641,
    3581,
    2946,
    1807,
    5215,
    614,
    4168,
    2576,
    2271,
    3554,
    4903,
    3801,
    4894,
    3094,
    723,
    2792,
    4680,
    1113,
    95,
    890,
    2168,
    4460,
    337,
    2822,
    3333,
    2186,
    5759,
    4372,
    989,
    5370,
    1055,
    4292,
    5535,
    2592,
    2217,
    5357,
    537,
    2484,
    2631,
    2607,
    824,
    3510,
    3336,
    3301,
    5490,
    3468,
    1642,
    4371
  ],
  "deleted_hypotheses_by_evaluator_type": {
    "action_likelihood": 102,
    "state_only_plausibility": 9
  },
  "logical_completed_step_hypothesis_evaluations": 2284,
  "logical_completed_step_batch_interface_calls": 908,
  "trajectory_step_count": 908,
  "bypassed_examples": 200,
  "summary_top_p_distinct_from_pruning": true,
  "mean_counts_by_trajectory_step": {
    "0": {
      "examples_reaching_step": 400,
      "mean_before_evaluation": 2.57,
      "mean_after_pruning": 2.405
    },
    "1": {
      "examples_reaching_step": 295,
      "mean_before_evaluation": 2.5559322033898306,
      "mean_after_pruning": 2.454237288135593
    },
    "2": {
      "examples_reaching_step": 172,
      "mean_before_evaluation": 2.383720930232558,
      "mean_after_pruning": 2.3197674418604652
    },
    "3": {
      "examples_reaching_step": 38,
      "mean_before_evaluation": 2.3157894736842106,
      "mean_after_pruning": 2.210526315789474
    },
    "4": {
      "examples_reaching_step": 3,
      "mean_before_evaluation": 1.3333333333333333,
      "mean_after_pruning": 1.3333333333333333
    }
  }
}
```

계산량(재시도 포함; smoke/연결 확인은 JSON에서 별도 집계):

```json
{
  "fixed-4": {
    "calls": 8813,
    "failed_calls": 0,
    "input_tokens": 3113752,
    "output_tokens": 777841,
    "total_tokens": 3891593,
    "call_wall_seconds": 45529.26599999989,
    "truncated_calls": 0,
    "partial_output_tokens_resource_interruptions": 0,
    "actual_inference_requests_by_stage_all_attempts": {
      "tt.target": 400,
      "tt.action_label": 400,
      "tt.perception": 1402,
      "tt.initialize": 400,
      "tt.likelihood": 2292,
      "tt.propagate": 2052,
      "tt.summary": 913,
      "final": 600,
      "tt.initial_agent_state": 346,
      "tt.rejuvenate": 8
    },
    "actual_evaluation_inference_requests_all_attempts": 2292,
    "batch_interface_calls_all_attempts": 0,
    "failed_generation_tokens_may_be_unknown": false
  },
  "adaptive-A": {
    "calls": 8011,
    "failed_calls": 2,
    "input_tokens": 2926875,
    "output_tokens": 636593,
    "total_tokens": 3563468,
    "call_wall_seconds": 30099.929999999746,
    "truncated_calls": 0,
    "partial_output_tokens_resource_interruptions": 17,
    "actual_inference_requests_by_stage_all_attempts": {
      "tt.target": 400,
      "tt.action_label": 400,
      "tt.perception": 1402,
      "tt.initialize": 400,
      "tt.likelihood": 1503,
      "tt.propagate": 1270,
      "adaptive.state_only_plausibility": 794,
      "tt.summary": 901,
      "final": 595,
      "tt.initial_agent_state": 346
    },
    "actual_evaluation_inference_requests_all_attempts": 2297,
    "batch_interface_calls_all_attempts": 1833,
    "failed_generation_tokens_may_be_unknown": true
  }
}
```

fixed GPU 겹침 관측: False. 총 GPU 메모리 표본은 다른 앱을 포함할 수 있다. 호출 시간은 논리적 가설 평가 횟수와 구분한다.

최종 답변 검토 항목 109개. TT 중간 의미 검토는 미완료다. 정확도만으로 중간 추론 타당성을 보증하지 않는다.

이 비교는 가변 초기 개수·pruning·재표집/재표현 제거·state-only 평가 추가의 묶음 차이다. pruning 하나의 인과 효과나 통계적 우월성을 주장하지 않는다. 삭제 후보는 복구할 수 없고, 후보 1개이면 상대 가중치는 항상 1이며, 모든 후보가 부적절해도 상대 가중치만으로 알아낼 수 없다.

Adaptive worker 경과 시간: 32479.5초(보고서 생성 시점 기준, 대기 제외, 로딩·smoke 포함). 호출 시간·토큰 및 모델 로딩 합은 comparison.json에 따로 보존한다.
