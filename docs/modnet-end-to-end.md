# MODNet End-to-End Pipeline

The repository now includes a concrete MODNet workflow:

1. Download the ONNX weights from `Xenova/modnet`.
2. Build a TensorRT engine from the ONNX file.
3. Run the TensorRT engine in a long-lived webcam process.

## Export

Download the Xenova ONNX file:

```bash
modnet-fetch --output-path src/bgremoval/models/weights/modnet/onnx/model.onnx --variant fp32
```

## Build

```bash
modnet-build --onnx-path src/bgremoval/models/weights/modnet/onnx/model.onnx --engine-path src/bgremoval/models/weights/modnet/modnet.engine --fp16
```

If the ONNX file is missing, `modnet-build` will now fetch `Xenova/modnet` automatically unless you disable it:

```bash
modnet-build --onnx-path src/bgremoval/models/weights/modnet/onnx/model.onnx --engine-path src/bgremoval/models/weights/modnet/modnet.engine --no-auto-fetch-onnx
```

## Runtime

```bash
modnet-run --engine-path src/bgremoval/models/weights/modnet/modnet.engine --mode loopback
```

Or stream to SRT:

```bash
modnet-run --engine-path src/bgremoval/models/weights/modnet/modnet.engine --mode srt --srt-url "srt://host:9000?mode=caller&latency=200000"
```

The SRT path is also exposed as a dedicated zero-copy-style entry point:

```bash
modnet-zero-copy --engine-path src/bgremoval/models/weights/modnet/modnet.engine --srt-url "srt://host:9000?mode=caller&latency=200000"
```

## Package integration

- The shared model registry now knows about `modnet-trt`.
- The top-level `bgremoval` CLI also accepts `--method modnet` as a shorthand for `modnet-trt`.
- The TensorRT helper layer lives in `src/bgremoval/models/tensorrt/`.
- The runtime stays local and loads the engine once.
- The TensorRT session now creates and owns the active PyCUDA device context before it allocates buffers or streams.
- Xenova/modnet is the preferred ONNX source for this path.
- The build step now reads the source repo and ONNX filename from the model registry metadata.
