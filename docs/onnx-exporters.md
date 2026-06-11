# ONNX Exporters

The repository keeps all active ONNX exporter implementations in one place:

- `src/bgremoval/models/exporters/modnet.py`
- `src/bgremoval/models/exporters/birefnet.py`
- `src/bgremoval/models/exporters/common.py`

The older paths under `src/bgremoval/models/modnet/export_onnx.py` remain as compatibility shims so the existing `modnet-export` entry point keeps working during the transition.

## Current commands

- `modnet-export` exports a local MODNet checkpoint to ONNX.
- `birefnet-export` exports a local BiRefNet `.pth` checkpoint to ONNX.

## Shared behavior

- Both exporters use the same `export_torch_model_to_onnx` helper for the final `torch.onnx.export` call.
- Both exporters support a configurable opset and optional dynamic axes.
- Both exporters write into the local model weights tree by default, such as `src/bgremoval/models/weights/birefnet/onnx/` or `src/bgremoval/models/weights/modnet/onnx/`.

## Compatibility

The repo keeps the old `modnet` export module path and can add the same style of shim for any future exporter if a CLI or external script depends on the legacy location.
