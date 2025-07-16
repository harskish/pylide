let
  pkgs = import (fetchTarball {
    # Head of https://github.com/NixOS/nixpkgs/tree/nixos-25.05 (14.7.2025)
    url="https://github.com/NixOS/nixpkgs/archive/650e572363c091045cdbc5b36b0f4c1f614d3058.tar.gz";
    sha256 = "150ip7d1izr4falxvnidgjmisbfja17rp4afigigld2hlhndafm7";
  }) {
      config = {
        permittedInsecurePackages = [];
      };
    };
  pp = pkgs.python313Packages; # remove .venv after version change
  
  # Use default.nix from nixpkgs, but override with own repo
  # Versions must match (both are 19.0.0)
  customHalide = (pkgs.callPackage "${pkgs.path}/pkgs/development/compilers/halide/default.nix" {
    python3Packages = pp;
    pythonSupport = true;
  }).overrideAttrs (oldAttrs: {
    src = pkgs.fetchFromGitHub {
      owner = "harskish";
      repo = "halide";
      rev = "mps";
      hash = "sha256-LRsUpcG3bgvEg8yk3YE3p+oUDVHjBKnbGhVaY+3TsMw=";
    };
    patches = []; # included in harskish/halide
    disabledTests = oldAttrs.disabledTests ++ ["correctness_pytorch"]; # TODO: re-enable
  });
  
  # Override LLVM with custom cmake flags using devExtraCmakeFlags
  #customClang = pkgs.llvmPackages_20.libclang.override {>
  #  devExtraCmakeFlags = [
  #    "-DLLVM_ENABLE_ASSERTIONS=ON"
  #    "-DLLVM_ENABLE_BINDINGS=OFF"
  #    "-DLLVM_ENABLE_CURL=OFF"
  #    "-DLLVM_ENABLE_DIA_SDK=OFF"
  #    "-DLLVM_ENABLE_EH=ON"
  #    "-DLLVM_ENABLE_HTTPLIB=OFF"
  #    "-DLLVM_ENABLE_IDE=OFF"
  #    "-DLLVM_ENABLE_LIBEDIT=OFF"
  #    "-DLLVM_ENABLE_LIBXML2=OFF"
  #    "-DLLVM_ENABLE_OCAMLDOC=OFF"
  #    "-DLLVM_ENABLE_PROJECTS=clang;lld"
  #    "-DLLVM_ENABLE_RTTI=ON"
  #    "-DLLVM_ENABLE_RUNTIMES=compiler-rt"
  #    "-DLLVM_ENABLE_WARNINGS=OFF"
  #    "-DLLVM_ENABLE_ZLIB=OFF"
  #    "-DLLVM_ENABLE_ZSTD=OFF"
  #    "-DLLVM_INCLUDE_BENCHMARKS=OFF"
  #    "-DLLVM_INCLUDE_EXAMPLES=OFF"
  #    "-DLLVM_INCLUDE_TESTS=OFF"
  #    "-DLLVM_TARGETS_TO_BUILD=WebAssembly;X86;AArch64;ARM;NVPTX"
  #  ];
  #};

in with pkgs; pkgs.mkShell {
  buildInputs = [
    # Python
    pp.python
    pp.uv
    pp.venvShellHook # creates .venv if missing
    customHalide # native + Python
  ] ++ lib.optionals stdenv.hostPlatform.isDarwin [
    # github.com/NixOS/nixpkgs/blob/25.05/pkgs/development/python-modules/torchvision/default.nix#L64
    apple-sdk_13
    (darwinMinVersionHook "12.0")
  ];
  
  venvDir = "./.venv";
  
  postVenvCreation = ''
    uv pip compile requirements.txt | uv pip sync -
  '';
  
  postShellHook = ''
    export LD_LIBRARY_PATH=${lib.makeLibraryPath [stdenv.cc.cc]}
  '';
}
