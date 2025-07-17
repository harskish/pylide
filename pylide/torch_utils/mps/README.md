# PyTorch MPS Custom Ops

This is a demo of mps custom ops for PyTorch.
Original from: https://github.com/grimoire/TorchMPSCustomOpsDemo

Environment:
- PyTorch 2.0 (github.com/kulinseth/pytorch/commit/a2af8c4d3d5c4989707540c80f212c06925dc824)
- Python 3.10.9
- Xcode 14.2
- MacOS 13.1 SDK

installation:

```bash
python setup.py develop
```

test:

```bash
python custom_add_test.py
```

Build generates *.metallib, *.so
otool -L:
_mps_test.cpython-310-darwin.so:
        @rpath/libc10.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libtorch.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libtorch_cpu.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libtorch_python.dylib (compatibility version 0.0.0, current version 0.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 1300.36.0)
        /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1319.0.0)

Ops to potentially implement:
ops/filtered_lrelu.py: avoid large intermediate tensor (2x or 4x), uses upfir internally in _ref
ops/upfirdn2d.py:      avoid large intermediate tensor (add Nx zeros, covolve, drop every Nth)
ops/bias_act.py:       does not seem too inefficient...