# 2026-10-30 재개 이후 자원 측정 주의

운영 세션 `adaptive_a_v2.resume_ops_20261030_v2`부터 WDDM GPU의 admission/실행 중 guard는 CUDA가 보고하는 현재 할당 가능 메모리를 사용한다. 원래 NVML(`nvidia-smi`) 수치는 각 GPU 기록의 `nvml_reported`에 함께 저장한다. 모델·BF16·SDPA·프롬프트·가설·seed·채점은 유지했다. 운영 코드와 실제20GiB 할당/쓰기 검증의 해시는 `records/cuda_admission_validation_20261030.json` 및 `records/cuda_allocation_validation_20261030.json`에 있다.

CUDA 할당 성공은 모든 페이지의 전용VRAM 상주나 Windows paging 부재를 직접 증명하지 않는다. 실제 진단 중 시스템 RAM 가용량 감소도 관측했다. OS/driver 설정과 양자화·모델 offload 설정은 바꾸지 않았다. [NVIDIA NVML의 WDDM 설명](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)을 참고하되, 이번 불일치의 정확한 원인이 규명됐다고 표현하지 않는다.

동결된 기존 자동 보고기의 `gpu_used_peak_mib_whole_device`는 재개 전NVML과 재개 후CUDA 수치를 함께 최대 집계하므로 **동일 기준의 장치 전체 메모리 peak로 사용하지 않는다.** 비교 시 각 표본에서 `nvml_reported`가 있으면 그 안의 `used_mib`, 없으면 원래GPU 행의 `used_mib`를 사용해 NVML peak를 따로 계산해야 한다. CUDA 기준과 PyTorch allocator 기준 peak도 각각 구분한다. raw 측정·오류·시점이 원시 로그에 남아 있으므로 전체600개 완료 후 재집계할 수 있다.

재개와 자원 집계 방식 차이, WDDM paging 가능성, 다른 GPU 부하 때문에 wall-clock 시간만으로 알고리즘 속도 향상을 주장하지 않는다. 논문의 모델/환경 전체 조건 재현이나 pruning 하나의 인과 효과도 주장하지 않는다.
