# Dependencies

The following versions describe the tested runtime and data environments.
They are a compatibility reference, not a portable installation lock.
Follow the [runtime setup guide](public_runtime.md) for installation and check
your environment with the [sample validation](VALIDATION.md).

| Component | Tested version |
|---|---|
| Python | 3.10.20 |
| GNU C++ build | GCC 11.4; Release configuration |
| CUDA compiler / TensorRT | 12.4 / 10.13.3.9 |
| ROS libraries, when enabled in the build | ROS 2 Humble |
| Data Torch / torchvision | 2.12.0 (CUDA 13.0 build) / 0.27.0 |
| Data NumPy / SciPy | 1.26.4 / 1.15.3 |
| Data Pinocchio distribution `pin` | 3.9.0 |
| Data pandas / PyArrow | 2.3.3 / 24.0.0 |
| Data datasets / LeRobot | 4.8.5 / 0.1.0 (`lerobot.common.datasets` API) |
| Video encoder PyAV / OpenCV | 15.1.0 / 4.11.0.86 |
| Data ONNX Runtime / pyzmq | 1.23.2 / 27.1.0 |
| Data MuJoCo / simulation MuJoCo | 3.8.1 / 3.10.0 |
| tyro / h5py / huggingface-hub | 1.0.15 / 3.16.0 / 0.35.3 |
| Pillow / pytest | 12.3.0 / 9.1.1 |

The controller uses C++ ONNX Runtime/TensorRT libraries, independently of Python
ONNX Runtime. Compiler CUDA and PyTorch CUDA builds are also distinct. Match
the SDK and GPU driver before installing packages; do not replace dependencies
inside a working training environment. Source the ROS setup if the controller
links against its libraries. Metadata/video processing also requires
`ffmpeg` and `ffprobe`.

## Check your environment

Check imports in the data environment. LeRobot must provide the
`lerobot.common.datasets` API used by the converters:

```bash
python -c 'import lerobot.common.datasets, pinocchio, torch; print("CUDA available:", torch.cuda.is_available())'
```

Check the video encoder in the simulation environment:

```bash
"$EGOALIGN_SIM_PYTHON" -c 'import av, mujoco; av.Codec("libx264", "w"); print("PyAV:", av.__version__, "MuJoCo:", mujoco.__version__)'
```

The [runtime guide](public_runtime.md#prepare-python-and-gpu-rendering) covers
interpreter selection and the NVIDIA EGL check. After setup, run the
[sample validation](VALIDATION.md).

For dependency troubleshooting, detailed versions and source identities are
available in the [machine-readable inventory](../configs/qualified_dependencies.json).
