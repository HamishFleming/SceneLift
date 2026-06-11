# MODNet TensorRT Scaffold

This folder is the model-specific home for the MODNet export/build/runtime pipeline.

Files:

- `export_onnx.py`: export a loaded MODNet-style PyTorch model to ONNX.
- `build_engine.py`: convert ONNX into a TensorRT engine for a specific GPU target.
- `runtime.py`: run a long-lived webcam process that can output to `v4l2loopback` or FFmpeg/SRT.

The code is intentionally split so each step can be swapped or customized per model without changing the shared runtime framework.
The ONNX export logic itself now lives in `src/bgremoval/models/exporters/modnet.py`; this package path remains as a compatibility shim.
