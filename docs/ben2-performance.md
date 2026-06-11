# BEN2 Performance Tuning

BEN2 in this repository runs through the shared TensorRT live pipeline. The main speed lever is input size, because the engine is built for a fixed shape.

## Highest-impact knobs

1. Reduce the engine input shape.

   The current BEN2 build path uses `1,3,1024,1024`. If you can tolerate lower spatial detail, rebuild and run at a smaller shape such as `1,3,768,768` or `1,3,512,512`.

   Example:

   ```bash
   ben2-build --input-shape 1,3,768,768
   ben2-run --width 768 --height 768
   ```

2. Keep FP16 enabled.

   The BEN2 build path already requests FP16, which is usually the right default for live inference on modern NVIDIA GPUs. On TensorRT 11 builds that do not expose `BuilderFlag.FP16`, the build step will log a warning and continue without the explicit flag.
   Repeated builds are faster when the timing cache is reused. The default BEN2 build path stores a cache next to each engine file, such as `ben2.engine.timing-cache` or `ben2-768.engine.timing-cache`.

3. Try INT8 only with representative calibration frames.

   INT8 can improve throughput further, but only when the calibration data matches your use case. Use `ben2-build --int8 --calibration-data-dir ...` and validate the output quality before relying on it for live work. The calibration cache is stored separately from the timing cache, so rebuilds can reuse both.

4. Prefer loopback over SRT when possible.

   `ben2-run --mode loopback` avoids the extra ffmpeg encode and network transport overhead that comes with SRT.

4. Keep the engine hot.

   Reuse a single long-lived process instead of spawning per frame or per clip. The runtime already does this, so the main thing is to avoid restarting it between inputs.

## Practical tradeoff

- Smaller shapes improve speed materially but reduce edge detail.
- SRT adds latency and CPU/GPU encode cost.
- FP16 usually improves throughput with little quality loss, but the exact gain depends on the GPU.
- INT8 can improve throughput again, but quality depends heavily on the calibration set.

## What to try first

If you want a faster BEN2 path, start with:

```bash
ben2-build --input-shape 1,3,768,768
ben2-run --width 768 --height 768 --mode loopback
```

If that still is not fast enough, move to `512x512` and compare output quality against throughput.

## Benchmark command

The repository includes a dedicated shape benchmark:

```bash
ben2-benchmark --input input/a.webp
```

By default it compares `1024`, `768`, and `512` square shapes, builds missing engines on demand, and reports load plus inference timings for each size.
Those builds also reuse and refresh the timing caches in the selected cache directory when available:

```bash
ben2-benchmark --input input/a.webp --cache-dir /tmp/ben2-cache
```

## Build all

To build all default BEN2 shapes in one pass, use:

```bash
ben2-build-all --cache-dir /tmp/ben2-cache
```

This uses one shared timing-cache file inside the selected cache directory, so the later shape builds can reuse timing data from the earlier ones. It also keeps the cache files together and makes repeat builds faster on the same machine and GPU.

The TensorRT runtime layer also resolves single-input engine tensor names from the engine itself, so BEN2 shape-specific engines do not depend on the ONNX input name matching the runtime tensor name exactly.

The BEN2 benchmark and build-all helpers also validate existing engine files before skipping a build. If a cached engine file was built for a different shape, the helper will rebuild it instead of trying to benchmark the stale file.

## Build-time GPU usage

It is normal for `nvidia-smi` to show low or even 0% `GPU-Util` while TensorRT is building an engine. The build path can spend a lot of time in parsing, graph compilation, tactic search, and memory planning, and those phases do not always show up as sustained GPU utilization.

If you see multiple CUDA processes on the same GPU while building, expect longer build times. On a smaller card like an RTX 2070, a second long-running process can push TensorRT into much slower tactic search and memory pressure behavior.

During a build, look for:

- GPU memory usage increasing
- TensorRT log lines such as `Compiler backend is used during engine build`
- eventual `Engine generation completed in ... seconds`

If you want a more granular view than `nvidia-smi`, use a higher refresh rate or GPU process monitoring tools such as `nvidia-smi dmon`, `nvidia-smi pmon`, or `nvtop`.
