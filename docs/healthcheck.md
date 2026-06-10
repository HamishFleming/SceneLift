# Healthcheck

`bgremoval-healthcheck` performs a one-shot readiness check for a selected backend and input source.

Example:

```bash
bgremoval-healthcheck --input input/a.webp --method mediapipe-selfie-segmentation
```

It logs:

- environment details
- input source readiness
- backend load readiness, including eager model/session initialization when the backend exposes it
- one inference pass with the resulting output shape
- `failed` or `skipped` statuses when a backend dependency is missing or inference cannot complete

The main `bgremoval` command also emits healthcheck-style logs when it opens an input source and constructs a backend, so startup issues are easier to diagnose from the same log stream.
