# Updating TensorRT

TensorRT upgrades depend on how it was installed.

## If TensorRT is installed in a Python virtual environment

This is the usual path for this repository when `tensorrt` was installed with `pip`.

Use the venv Python to remove any cached TensorRT wheels, then upgrade the Python packages:

```bash
./.venv/bin/python -m pip cache remove "tensorrt*"
./.venv/bin/python -m pip install --upgrade tensorrt tensorrt-lean tensorrt-dispatch
```

The NVIDIA docs note that the pip method is the fastest installation path and is intended for Python development. They also document the same `pip cache remove` plus `pip install --upgrade` flow for upgrading from TensorRT 10.x to 11.x.

## If TensorRT was installed with Debian packages

Install the newer local repo package, then upgrade TensorRT through `apt`:

```bash
sudo dpkg -i nv-tensorrt-local-repo-<...>.deb
sudo cp /var/nv-tensorrt-local-repo-<...>/*-keyring.gpg /usr/share/keyrings
sudo apt-get update
sudo apt-get install tensorrt
```

NVIDIA’s upgrade guide says the Debian method can upgrade the development environment while leaving runtime components in place.

## If TensorRT was installed with RPM packages

Use the new local repo RPM, then upgrade through `dnf`:

```bash
sudo rpm -Uvh nv-tensorrt-local-repo-<...>.rpm
sudo dnf clean expire-cache
sudo dnf install tensorrt
```

## If TensorRT was installed from a tar/zip archive

Install the new version into a new location, then point your environment at the new libraries:

```bash
export LD_LIBRARY_PATH=/path/to/TensorRT/lib:$LD_LIBRARY_PATH
```

For zip installs on Windows, update `PATH` to point at the new TensorRT location.

## Verify the upgrade

After upgrading, confirm both the Python package and `trtexec` work:

```bash
./.venv/bin/python -c "import tensorrt as trt; print(trt.__version__)"
trtexec --help
```

## Notes

- TensorRT 11.0.0 is the latest release listed in the current NVIDIA installation guide.
- If you are upgrading from an unsupported older version, NVIDIA recommends upgrading incrementally or uninstalling and reinstalling the newer version.
- If you installed TensorRT outside the venv, do not mix old and new libraries on the same `LD_LIBRARY_PATH`.
