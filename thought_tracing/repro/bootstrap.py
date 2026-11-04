"""Process-local runtime and telemetry settings, before ML imports."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for key, value in {
    "WANDB_MODE": "disabled", "WANDB_DISABLED": "true", "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "DO_NOT_TRACK": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TOKENIZERS_PARALLELISM": "false",
}.items():
    os.environ[key] = value

DLL_HANDLE = None
if os.name == "nt" and (ROOT / ".tools/vc_runtime/dlls").is_dir():
    DLL_HANDLE = os.add_dll_directory(str(ROOT / ".tools/vc_runtime/dlls"))
