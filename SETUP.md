# PyLide Setup Guide

## Halide PyPi Wheels?

PyPi wheels of Halide are available! However, they are missing:

- Autoschedulers
- Metal Torch extensions
- CUDA Torch extensions on Windows (no weak linkage)
- Fix for forward declarations in scheduled pipelines (during PT wrapper generation)

## Tested Platforms

- **Windows 10** + VS2022
- **Windows 11** + VS2019
- **MacOS 13.1** Ventura
- **MacOS 14.5** Sonoma + Xcode 15.4
- **MacOS 15.5** Sequoia + Nix
- **Ubuntu 22.04** + gcc 9.4.0

## Prerequisites (Windows)

1. Add "C++ ATL for latest v143 build tools (x86 & x64)" in VS2022 installer and reboot
2. Start "x64 native tools command prompt for VS 2022"

# Setup Instructions (Nix)

Simply run `nix-shell`
- Nixpkgs pinned to `nixos-25.05`
- Creates a Python venv, with dependencies installed from `requirements.txt`
- Compiles a custom version of Halide

# Setup Instructions (Other)

## Python and LLVM setup
### 1. Setup Python environment
```bash
conda activate pylide
conda install cmake=3.31 ninja=1.12.1
```

### 2. Clone LLVM

```bash
# Clone llvm (match github.com/harskish/Halide/blob/main/.github/workflows/pip.yml#L15)
# Alternatively: curl https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-19.1.0.zip
git clone --depth 1 --branch llvmorg-19.1.0 https://github.com/llvm/llvm-project.git llvm-19.1.0
cd llvm-19.1.0
```

### 3. Fix Line Endings (Linux only)

```bash
dos2unix llvm/cmake/config.guess
```

### 4. Configure LLVM Build

> **Note:** Possible targets: `X86;NVPTX;ARM;AArch64;Mips;Hexagon;WebAssembly`  
> **Windows:** targets ninja generator inside VS 2022 command prompt!  
> **macOS:** need to build X86 backend (for examples)

```bash
cmake -G "Ninja" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_ENABLE_CURL=OFF \
  -DLLVM_ENABLE_DIA_SDK=OFF \
  -DLLVM_ENABLE_EH=ON \
  -DLLVM_ENABLE_HTTPLIB=OFF \
  -DLLVM_ENABLE_IDE=OFF \
  -DLLVM_ENABLE_LIBEDIT=OFF \
  -DLLVM_ENABLE_LIBXML2=OFF \
  -DLLVM_ENABLE_OCAMLDOC=OFF \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_RUNTIMES=compiler-rt \
  -DLLVM_ENABLE_WARNINGS=OFF \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_TARGETS_TO_BUILD="WebAssembly;X86;AArch64;ARM;NVPTX" \
  -S llvm -B build
```

### 5. Build LLVM

```bash
cmake --build build
```

### 6. Install LLVM
**Windows:**
```bash
cmake --install build --prefix C:/libs/llvm19
```

**macOS:**
```bash
cmake --install build --prefix ~/llvm19
```

### 7. Set Environment Variables

**Windows:**
```bash
set LLVM_ROOT=C:/libs/llvm19
set LLVM_CONFIG=C:/libs/llvm19/bin/llvm-config.exe
```

**macOS:**
```bash
export LLVM_ROOT=~/llvm19
export LLVM_CONFIG=~/llvm19/bin/llvm-config
```

## Halide Setup

### 1. Clone Halide (Submodule)

```bash
git submodule update --init --recursive
cd Halide
```

### 2. Configure Halide Build

> **Note:** On Windows, run within x64 native tools command prompt

```bash
conda activate pylide && bash -c "cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DWITH_PYTHON_BINDINGS=ON -DLLVM_DIR=$LLVM_ROOT/lib/cmake/llvm -S . -B ~/halide_build"
```
The target python version (e.g. 3.12) is deduced from the active conda env - change as necessary!

### 3. Setup Build Aliases

**Windows** (`~/cmd_aliases/hlbuild.bat`):
```batch
@bash -c "cmake --build ~/halide_build && cmake --install ~/halide_build --prefix ~/halide-install"
```
> Add `~/cmd_aliases` to PATH

**WSL/Linux:**
```bash
alias hlbuild='conda activate pylide && cmake --build ~/halide_build && cmake --install ~/halide_build --prefix ~/halide-install'
```

### 4. Build Halide

> **Note:** On Windows, run within x64 native tools command prompt

```bash
hlbuild
```

## Create Python Wheel

```bash
python create_wheel.py
```
