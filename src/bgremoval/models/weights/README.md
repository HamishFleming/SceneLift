# Model Weights

This directory is reserved for local model weights and cached artifacts.
Git ignores the generated contents here so fetched checkpoints, exported ONNX files, and TensorRT engines stay local.

Suggested layout:

- `grabcut/` for classical config or tuning files
- `rembg/` for local rembg sessions or alternate weights
- `birefnet/` for locally mirrored BiRefNet checkpoints or exports
- `realesrgan/` for local Real-ESRGAN checkpoints such as `RealESRGAN_x4plus.pth`
- `u2net-human-seg/` for cached `onnx/model.onnx` and `preprocessor_config.json` artifacts from U-2-Net-Human-Seg
- `mediapipe-selfie-segmentation/` for cached `onnx/model.onnx` and `preprocessor_config.json` artifacts from MediaPipe Selfie Segmentation

The built-in `birefnet` backend loads from Hugging Face by default, and the local `birefnet-export` command can write ONNX exports into `birefnet/onnx/` if you want to keep a checkpoint-derived artifact beside the mirrored weights.
The TensorRT builders can also write INT8 calibration caches next to their engines in the same tree, for example `modnet/modnet.engine.int8.cache` or `ben2/ben2.engine.int8.cache`.
The upscaling command family uses the same weights tree and looks for Real-ESRGAN checkpoints under `realesrgan/` by default.
The built-in `u2net-human-seg` backend downloads its ONNX model and processor config from Hugging Face by default and can be redirected to a local mirrored directory through the registry.
The built-in `mediapipe-selfie-segmentation` backend uses the same ONNXRuntime path and local cache structure.
