# SceneLift

AI background removal for creators and streamers.

SceneLift helps you isolate yourself from your background with a local CLI that works on:

- live webcam input for OBS and streaming
- image files, video files, or folders of frames
- transparent PNGs or rendered video output
- virtual camera loopback output when `pyvirtualcam` is installed

The primary CLI is still `bgremoval`; `bgremove` remains available as a compatibility alias.

## Why SceneLift

- Built for people who want a clean camera feed without wrestling with machine-learning jargon.
- Focused on low-latency live use, especially for OBS and streaming setups.
- Works with local files and local cameras, so you can keep the workflow simple and private.
- Preserves the existing CLI and Python API for people already using the project.
