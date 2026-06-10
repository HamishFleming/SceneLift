# Next Steps

The project should now center on the single live backend:

1. `modnet-trt` as the primary runtime.
2. A low-latency live daemon that keeps the engine hot and feeds OBS.
3. A small preset/config layer for common webcam resolutions and GPU budgets.
4. A local validation harness so latency and frame drops can be measured quickly.

## Priority 1: Live daemon

Build a single long-running process that:

- opens the webcam once
- loads the TensorRT engine once
- keeps only the newest frame
- outputs to `v4l2loopback` for OBS on Linux
- optionally outputs via FFmpeg/SRT if network transport is needed

## Priority 2: Presets

Add a small set of presets:

- `fast` for 720p and lower latency
- `balanced` for 1080p
- `quality` for the best mask at a higher cost

## Priority 3: Validation

Add a benchmark command or script that records:

- average FPS
- frame drops
- end-to-end latency
- output resolution

## Priority 4: Packaging

Once the runtime is stable, make the supported path obvious:

- keep `modnet-trt` exposed in the CLI
- keep the other model scaffolds available but clearly secondary
- document the exact install commands for the live path

