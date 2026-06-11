from __future__ import annotations

import json
import platform


def main() -> None:
    payload = {
        "python_platform": platform.platform(),
        "machine": platform.machine(),
    }

    try:
        import torch

        payload["torch_version"] = torch.__version__
        payload["cuda_available"] = torch.cuda.is_available()
        payload["cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            payload["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        payload["torch_import_error"] = str(exc)

    try:
        import transformers

        payload["transformers_version"] = transformers.__version__
    except Exception as exc:
        payload["transformers_import_error"] = str(exc)

    try:
        import accelerate

        payload["accelerate_version"] = accelerate.__version__
    except Exception as exc:
        payload["accelerate_import_error"] = str(exc)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
