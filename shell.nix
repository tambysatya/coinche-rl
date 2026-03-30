let
  nixpkgs = import <nixpkgs> { config.allowUnfree = true; };
  lib = nixpkgs.lib;
  pkgs = nixpkgs;

in
pkgs.mkShell {
  buildInputs = [
    pkgs.gcc
    pkgs.stdenv.cc.cc.lib  # ← IMPORTANT : runtime libstdc++ ici
    pkgs.uv
    pkgs.virtualenv
    pkgs.cudaPackages.cuda_cccl
    pkgs.cudaPackages.cudatoolkit
    pkgs.cudaPackages.cudnn                 # ← pour JAX + NN
    pkgs.cudaPackages.cuda_cupti            # ← pour profiling/tracing (optionnel)
    pkgs.nvtopPackages.nvidia

      ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.python313.sitePackages}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${lib.getLib pkgs.gcc}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.expat}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.libz}/lib:$LD_LIBRARY_PATH

    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.cudaPackages.cuda_cupti}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH



    unset SOURCE_DATE_EPOCH
  '';
}

