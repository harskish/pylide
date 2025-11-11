# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# Modified by Erik Härkönen, 17.11.2022

import glob
import hashlib
import importlib
import os
import re
import shutil
import uuid

import torch
import torch.utils.cpp_extension
from pathlib import Path
from threading import get_native_id

#----------------------------------------------------------------------------
# Global options.

VERBOSITY = 'brief' # Verbosity level: 'none', 'brief', 'full'

#----------------------------------------------------------------------------
# Internal helper funcs.

def _find_compiler_bindir():
    patterns = [
        'C:/Program Files*/Microsoft Visual Studio/*/Professional/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio/*/Community/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio */vc/bin',
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if len(matches):
            return matches[-1]
    return None

#----------------------------------------------------------------------------

def _get_mangled_gpu_name():
    name = torch.cuda.get_device_name().lower() if torch.cuda.is_available() else 'cpu'
    out = []
    for c in name:
        if re.match('[a-z0-9_-]+', c):
            out.append(c)
        else:
            out.append('-')
    return ''.join(out)

#----------------------------------------------------------------------------
# Main entry point for compiling and loading C++/CUDA plugins.

_cached_plugins = dict()

def get_plugin(
    module_name: str,
    sources: list[Path],
    headers: list[Path] = None,
    source_dir: Path = None,
    extra_files_to_hash: list[Path] = None,
    extra_include_paths: list[Path] = None,
    **build_kwargs
):
    assert VERBOSITY in ['none', 'brief', 'full']
    if headers is None:
        headers = []
    if source_dir is not None:
        sources = [source_dir / s for s in sources]
        headers = [source_dir / h for h in headers]

    # Already cached?
    if module_name in _cached_plugins:
        return _cached_plugins[module_name]

    # Print status.
    if VERBOSITY == 'full':
        print(f'Setting up PyTorch plugin "{module_name}"...')
    elif VERBOSITY == 'brief':
        print(f'Setting up PyTorch plugin "{module_name}"... ', end='', flush=True)
    verbose_build = (VERBOSITY == 'full')

    # Compile and load.
    try: # pylint: disable=too-many-nested-blocks
        # Make sure we can find the necessary compiler binaries.
        if os.name == 'nt' and os.system("where cl.exe >nul 2>nul") != 0:
            compiler_bindir = _find_compiler_bindir()
            if compiler_bindir is None:
                raise RuntimeError(f'Could not find MSVC/GCC/CLANG installation on this computer. Check _find_compiler_bindir() in "{__file__}".')
            os.environ['PATH'] += ';' + compiler_bindir

        # Some containers set TORCH_CUDA_ARCH_LIST to a list that can either
        # break the build or unnecessarily restrict what's available to nvcc.
        # Unset it to let nvcc decide based on what's available on the
        # machine.
        os.environ['TORCH_CUDA_ARCH_LIST'] = ''

        # Incremental build md5sum trickery.  Copies all the input source files
        # into a cached build directory under a combined md5 digest of the input
        # source files.  Copying is done only if the combined digest has changed.
        # This keeps input file timestamps and filenames the same as in previous
        # extension builds, allowing for fast incremental rebuilds.
        #
        # This optimization is done only in case all the source files reside in
        # a single directory (just for simplicity) and if the TORCH_EXTENSIONS_DIR
        # environment variable is set (we take this as a signal that the user
        # actually cares about this.)
        #
        # EDIT: We now do it regardless of TORCH_EXTENSIOS_DIR, in order to work
        # around the *.cu dependency bug in ninja config.
        #
        all_source_files = [p.absolute() for p in sorted(sources + headers)]

        # Find common prefix to all sources in order to
        # keep folder hierarchy in tmp dir
        common_prefix = Path(os.path.commonpath(all_source_files))

        # Compute combined hash digest for all source files.
        hash_md5 = hashlib.md5()
        for src in all_source_files + (extra_files_to_hash or []):
            hash_md5.update(src.read_bytes())

        # Select cached build directory name.
        source_digest = hash_md5.hexdigest()
        build_top_dir = Path(torch.utils.cpp_extension._get_build_directory(module_name, verbose=verbose_build)) # pylint: disable=protected-access
        cached_build_dir = build_top_dir / f'{source_digest}-{_get_mangled_gpu_name()}'
        
        # Check if prebuilt plugin is available
        pydfile = cached_build_dir / f'{module_name}.pyd' # Windows
        sofile = cached_build_dir / f'{module_name}.so' # MacOS, Linux
        prebuilt = [l for l in [pydfile, sofile] if l.is_file()]

        if VERBOSITY == 'full':
            print('Build dir:', cached_build_dir)

        # Atomically create build dir
        # Avoids race conditions
        if not cached_build_dir.is_dir():
            tmpdir = build_top_dir / f'srctmp-{uuid.uuid4().hex}'
            os.makedirs(tmpdir)
            for src in all_source_files:
                trg = tmpdir / src.relative_to(common_prefix) # keeps dir structure
                os.makedirs(trg.parent, exist_ok=True)
                shutil.copyfile(src, trg)
            try:
                os.replace(tmpdir, cached_build_dir) # atomic
            except OSError:
                # source directory already exists, delete tmpdir and its contents.
                shutil.rmtree(tmpdir)
                if not cached_build_dir.is_dir(): raise
        
        # torch.utils.cpp_extension often recompiles despite identical sources
        # manually load cached library if available
        if prebuilt:
            if VERBOSITY != 'none':
                print(f'using prebuilt ({prebuilt[0].suffix})... ',  end='')
            import sys; sys.path.append(str(cached_build_dir.resolve()))
            module = importlib.import_module(module_name)
        else:
            # Don't wait silently for dangling lockfile
            lockfile = cached_build_dir / 'lock'
            has_to_wait = lockfile.is_file()
            if has_to_wait:
                print(f'Waiting for lockfile (tid={get_native_id()}): {lockfile}')
            
            # Compile.
            cached_sources = [cached_build_dir / fname.relative_to(common_prefix) for fname in sources]
            module = torch.utils.cpp_extension.load(
                name=module_name,
                build_directory=str(cached_build_dir),
                verbose=verbose_build,
                sources=[str(s.resolve()) for s in cached_sources],
                extra_include_paths=[str(p) for p in extra_include_paths],
                **build_kwargs
            )

            if has_to_wait:
                print(f'Lockfile was released (tid={get_native_id()})')

    except:
        if VERBOSITY == 'brief':
            print('Failed!')
        raise

    # Print status and add to cache dict.
    if VERBOSITY == 'full':
        print(f'Done setting up PyTorch plugin "{module_name}".')
    elif VERBOSITY == 'brief':
        print('Done.')
    _cached_plugins[module_name] = module
    return module

#----------------------------------------------------------------------------
