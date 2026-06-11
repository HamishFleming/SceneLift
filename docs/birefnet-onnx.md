# BiRefNet ONNX Export

This repository now ships a shared ONNX exporter framework under `src/bgremoval/models/exporters/`, with a BiRefNet exporter wired into it as `birefnet-export`.

For `BiRefNet-general-bb_swin_v1_tiny-epoch_232.pth`, the safest path is:

1. Use `birefnet-export` from this repo.
2. Point it at `BiRefNet-general-bb_swin_v1_tiny-epoch_232.pth`.
3. Keep the default `ZhengPeng7/BiRefNet` model name unless you have a local config directory.
4. Validate the exported model with ONNXRuntime before using it in this repo.

## Minimal export template

The exporter is implemented in [`src/bgremoval/models/exporters/birefnet.py`](/mnt/code/ai/background-removal/ben/src/bgremoval/models/exporters/birefnet.py) and uses the Hugging Face BiRefNet architecture plus a local checkpoint load.

```python
import torch
from pathlib import Path

from bgremoval.models.exporters.birefnet import BiRefNetExportConfig, export_to_onnx

export_to_onnx(
    BiRefNetExportConfig(
        checkpoint_path=Path("BiRefNet-general-bb_swin_v1_tiny-epoch_232.pth"),
        onnx_path=Path("BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx"),
    )
)
```

## Validation

After export, run a quick inference check with ONNXRuntime and compare the output against PyTorch on the same input. If the checkpoint uses a custom forward signature or returns multiple tensors, adjust the export wrapper in `src/bgremoval/models/exporters/birefnet.py` so the ONNX graph exposes only the mask tensor you want to consume downstream.

## Notes for this repo

- The local `birefnet` backend in `src/bgremoval/models/backends/birefnet.py` is still PyTorch/Hugging Face based.
- On first live use, `bgremoval --method birefnet` loads the Hugging Face BiRefNet model at runtime and may need to download it if it is not already cached locally.
- If you interrupt that first load with `Ctrl-C`, the traceback will usually show deep `transformers` and `torch` import frames. That indicates an interrupted startup path, not a BiRefNet inference failure.
- That backend is functional for experimentation, but it is much heavier than the TensorRT paths and is not the preferred low-latency virtual-camera backend in this repo.
- Local mirrored weights or exports can live under `src/bgremoval/models/weights/birefnet/`.
- If you want a repo-native export workflow for another model, add it under `src/bgremoval/models/exporters/` and keep the old path as a thin shim if needed.
