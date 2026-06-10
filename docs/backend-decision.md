# Backend Decision

The canonical backend for this project is:

- `MODNet` as the model family
- `TensorRT` as the runtime backend
- `live` mode with a single hot process
- `v4l2loopback` as the OBS handoff on Linux

Why this is the chosen path:

- It is the fastest practical local backend on NVIDIA GPUs.
- The model is lightweight enough for webcam latency targets.
- The engine can be built once and reused across sessions.
- The live pipeline already supports frame dropping and low-latency output.

What is not the primary path:

- `BEN2-TensorRT` is kept as an experimental/alternate scaffold.
- Pure Python frame-by-frame processing is not the main live path.
- Disk-based intermediate outputs are not part of the live stream path.

Current rule of thumb:

- If the target is live OBS background removal, use `modnet-trt` and keep the runtime hot.
- If the target is offline experimentation, use the Python API or the batch pipeline.
