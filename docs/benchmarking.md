# Benchmarking

`bgremoval-benchmark` measures multiple background-removal backends on the same input frame and logs the results in a structured form.

Example:

```bash
bgremoval-benchmark --input input/a.webp --methods modnet-trt,u2net-human-seg,mediapipe-selfie-segmentation,grabcut
```

It reports:

- model load time
- warmup time
- average inference time
- minimum and maximum inference time
- derived FPS
- `skipped` when a method cannot run because an optional dependency is missing or the backend fails to load

The command uses the same logging controller as the main CLI, so `--log-file` and `--log-json` work here too.
