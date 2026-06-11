# Model Directory Layout

The project now separates model support into reusable layers:

- `src/bgremoval/models/`
  - shared model registry and metadata
  - classical and Hugging Face-backed removers
  - TensorRT helper layer
- `src/bgremoval/models/modnet/`
  - Xenova/modnet ONNX fetch helper
  - TensorRT engine build script
  - live runtime for webcam output
- `src/bgremoval/models/ben2/`
  - BEN2 ONNX fetch helper
  - TensorRT engine build script
  - live runtime wrapper for the shared TensorRT scaffold
- `src/bgremoval/models/weights/`
  - local cached weights and exported artifacts by model name
  - ignored by git except for the per-folder `README.md` files

Example model-specific artifact layout:

- `src/bgremoval/models/weights/modnet/onnx/model.onnx`
- `src/bgremoval/models/weights/modnet/modnet.engine`
- `src/bgremoval/models/weights/ben2/onnx/model_fp16.onnx`
- `src/bgremoval/models/weights/ben2/ben2.engine`
- `src/bgremoval/models/weights/u2net-human-seg/`
  - cached `onnx/model.onnx` and `preprocessor_config.json` artifacts from `BritishWerewolf/U-2-Net-Human-Seg`
- `src/bgremoval/models/weights/mediapipe-selfie-segmentation/`
  - cached `onnx/model.onnx` and `preprocessor_config.json` artifacts from `onnx-community/mediapipe_selfie_segmentation`
- `src/bgremoval/models/weights/rvm/`
  - optional local cache area for Robust Video Matting artifacts if you decide to mirror them beside the repo

Common generated model artifacts such as `.onnx`, `.engine`, `.plan`, `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.tflite`, and `.npy` files are also ignored so local fetch/build steps can populate them without polluting the repository.

The build step can fetch `Xenova/modnet` automatically into the ONNX path if the file is missing.
The same registry-driven fetch path is also available for `onnx-community/BEN2-ONNX`.
The `u2net-human-seg` backend downloads the model's ONNX file and processor config from Hugging Face by default and can also be pointed at a local mirrored directory later.
The `mediapipe-selfie-segmentation` backend uses the same ONNXRuntime-backed cache layout and can also be mirrored locally later.
The `rvm` backend uses Torch Hub against `PeterL1n/RobustVideoMatting` and keeps recurrent state for video streams.
Use `bgremoval-model-pull-all` when you want to prefetch every registry-backed model asset into the local weights tree in one pass.
Use `bgremoval-trt-build-all` when you want the default MODNet engine plus the BEN2 shape set built in one pass.
Use `bgremoval-trt-build-int8` when you want the same setup with INT8 calibration enabled.

This layout keeps model-specific code isolated while still allowing shared runtime logic and common logging/configuration.
