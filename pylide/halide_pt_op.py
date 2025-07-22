from pylide.create_wheel import make_editable_install; make_editable_install()
import halide as hl # type: ignore
import shutil
import os
import sys
import time
import json
import io
import uuid
import inspect
import numpy as np
import torch
from pathlib import Path
from typing import Union
from hashlib import md5
from py.io import StdCaptureFD # type: ignore - pip install py==1.11.0
from contextlib import redirect_stdout, redirect_stderr
from collections import defaultdict
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from . import custom_ops_hl

VERBOSITY = 'brief' # 'brief', 'none'

# MacOS: .dylib = dynamic library, .a = static library
# See: Halide\src\Module.cpp::get_output_info()
lib_ext = { 'win32': 'lib', 'linux': 'a', 'darwin': 'a' }[sys.platform]
obj_ext = { 'win32': 'obj', 'linux': 'o', 'darwin': 'o' }[sys.platform]

# Use same output dir as other PT extensions
BUILD_TOP_DIR = Path(os.environ.get('TORCH_EXTENSIONS_DIR', os.path.expanduser('~/.cache/')))

def generate_pybind_wrapper(path: Path, op_names: list[str], has_cuda: bool, has_mps: bool):
    s = '#include "torch/extension.h"\n\n'
    if has_cuda:
        s += '#include "HalidePyTorchCudaHelpers.h"\n'
    
    if has_mps:
        s += '#include "HalidePyTorchMetalHelpers.h"\n'

    s += '#include "HalidePyTorchHelpers.h"\n'
    for op in op_names:
        s += f'#include "{op}.pytorch.h"\n'

    s += "\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
    for op in op_names:
        s += "    m.def(\"{n}\", &{n}_th_, \"PyTorch wrapper of the Halide pipeline {n}\");\n".format(n=op)
    s += "}\n"
    path.write_text(s)

def _check_tensor(t: torch.Tensor):
    assert t.ndim != 3 or t.shape[0] in [1, 3, 4], 'Halide expects CHW tensors'

import numbers


def _make_hl_scalar(v: numbers.Number):
    type = hl.Int(32) if isinstance(v, numbers.Integral) else hl.Float(32)
    return hl.InputScalar(type)

def _make_hl_buffer(t: torch.Tensor, reverse_axes=True):
    """
    Create a hl.Buffer that owns a copy of the data in t
    By default, hl.Buffer will alias the same memory.
    """
    _check_tensor(t)
    type_mapping = {
        np.dtype('float32'): hl.Float(32),
        np.dtype('int32'): hl.Int(32),
        np.dtype('uint8'): hl.UInt(8),
    }

    assert reverse_axes, 'copy_from below untested, probably broken'
    buff_np = t.detach().cpu().numpy() if torch.is_tensor(t) else t.copy()
    sizes = t.shape[::-1] if reverse_axes else t.shape
    buff = hl.Buffer(type=type_mapping[buff_np.dtype], sizes=sizes)
    buff.copy_from(hl.Buffer(buff_np)) # copy of data with ownership

    # Test broken version
    #buff = hl.Buffer(buff_np)

    # Test ownership: in-place modification should be invisible to buffer
    buff_np[:] = buff_np.max() + 1 # all same value, guaranteed different
    assert not np.all(np.asanyarray(buff) == buff_np.max()), 'Ownership seems broken'

    return buff
    

def _make_hl_imageparam(t: torch.Tensor):
    buff = _make_hl_buffer(t)
    img = hl.InputBuffer(buff.type(), dimensions=buff.dimensions()) # an ImageParam, supports dynamic shapes
    
    # "Bind an Image to this ImageParam. Only relevant for jitting" (ImageParam.h)
    img.set(buff)
    
    return img

def compile_pipeline_process_func(
    name: str,
    device: str, # cuda/mps/cpu etc.
    def_args: list,
    arg_names: list[str],
    in_args: list,
    kwargs: dict,
    debug: bool,
    # function
    marshaled_bytecode,
    pickled_name,
    pickled_arguments,
    pickled_closure,
    fname: str,
    ret_queue, # multiprocessing.Queue
    verbosity: str,
):
    """Compile pipeline in separate process to ensure deterministic stmt generation (and thus caching)"""
    import marshal
    import pickle
    import types
    
    func: hl.Func = types.FunctionType(marshal.loads(marshaled_bytecode), globals(), pickle.loads(pickled_name), pickle.loads(pickled_arguments), pickle.loads(pickled_closure))
    outdir = compile_pipeline(func, name, device, def_args, arg_names, in_args, kwargs, debug, fname, verbosity)
    ret_queue.put(outdir)
    
    return outdir

def compile_pipeline(
    func: hl.Func,
    name: str,
    device: str, # cuda/mps/cpu etc.
    def_args: list,
    arg_names: list[str],
    in_args: list,
    kwargs: dict,
    debug: bool,
    fname: str,
    verbosity: str,
):  
    from .utilities import hl_init_autosched
    hl_init_autosched()

    if verbosity != 'none':
        print(f'Creating PT op "{name}" for {device}')
    
    if def_args:
        names = arg_names[-len(def_args):]
        if verbosity == 'full':
            print('Using default positional arguments: ' + ', '.join(f'{k}={v}' for k,v in zip(names, def_args)))

    inputs = []
    for k, v in zip(arg_names, in_args):
        if torch.is_tensor(v):
            inputs.append(_make_hl_imageparam(v))
        elif isinstance(v, numbers.Number) and not isinstance(v, bool):
            inputs.append(_make_hl_scalar(v))
        else:
            raise RuntimeError(f'Positional arg "{k}" not tensor or scalar, please pass as kwarg (compile-time constant)')

    target = hl.get_host_target() #.with_feature(hl.TargetFeature.EnableLLVMLoopOpt)
    
    if device == 'cuda':
        target = target \
            .with_feature(hl.TargetFeature.CUDA) \
            .with_feature(hl.TargetFeature.UserContext)
    
    if device == 'mps':
        target = target \
            .with_feature(hl.TargetFeature.Metal) \
            .with_feature(hl.TargetFeature.UserContext)
    
    if debug:
        target = target.with_feature(hl.TargetFeature.Debug)
        
    # Pass hl.Target for scheduling logic etc.
    if 'target' in inspect.signature(func).parameters:
        kwargs = { **kwargs, 'target': target }
    
    # Get computation graph + schedule
    f: hl.Func = func(*inputs, **kwargs)

    # Pseudocode that shows iteration order
    capture = StdCaptureFD(out=False, in_=False)
    f.print_loop_nest() # logic: PrintLoopNest.cpp
    _, loop_nest = capture.reset()
    
    if debug:
        print(f'{name}:\n{loop_nest}')
    
    # GPU: must define schedule
    if target.has_gpu_feature():
        #assert 'gpu_thread' in loop_nest, 'Must specify shedule when running on GPU'
        print('Skipping GPU schedule check')

    # Build pipeline and Torch extension
    outdir = build_pipeline(f, inputs, target, fname, verbosity)

    return outdir

# Decorator for JIT-compiled pytorch extensions using Halide
__HL_JIT_CACHE = defaultdict(dict) # torch.device => func mapping
def hl_torch_op(func):
    name = func.__name__
    assert not __HL_JIT_CACHE[name], f'Duplicate Torch operator name {name}'

    # Get default positional args if not passed
    # (must be passed wrapped in InputBuffer / InputSclar)
    all_args = list(inspect.signature(func).parameters.items())
    arg_names = [k for k,_ in all_args]
    pos_args = [(k,v) for k,v in all_args if v.kind == inspect._ParameterKind.POSITIONAL_ONLY]
    pos_arg_defaults = [v.default for _,v in pos_args]

    def wrapper(*in_args, out: torch.Tensor = None, debug=False, **kwargs):
        # Halide cannot automatically infer output shapes...?
        assert torch.is_tensor(out), 'Must provide out tensor (kwarg) to Halide ops'
        _check_tensor(out)

        # Input positional args must be passed in order
        # => get any missing defaults based on length
        def_args = [] if len(in_args) >= len(pos_arg_defaults) else pos_arg_defaults[len(in_args):]
        in_args = [*in_args, *def_args]

        # Input tensors: number of dimensions must stay constant 
        ranks = ''.join([str(v.ndim) for v in in_args if torch.is_tensor(v)])
        signature = json.dumps(kwargs, sort_keys=True) + ranks
        param_hash = md5(signature.encode('utf-8')).hexdigest()
        func_key = (out.device.type, debug, param_hash)
        
        fun = __HL_JIT_CACHE[name].get(func_key, None) # put aside, cache might be cleared before next access
        if fun is None:
            import light_process as lp
            from functools import partial

            # Run type checks (at compile-time)
            # kwargs treated as compile-time constants
            assert not any(torch.is_tensor(v) for v in kwargs.values()), \
                'kwargs cannot be tensors (treated as compile-time constants)'
            
            # MPS tensor serialization not yet implemented
            device = out.device.type
            in_args_cpu = []
            for a in in_args:
                if torch.is_tensor(a):
                    assert a.device.type == device, "Input and output devices don't match"
                    in_args_cpu.append(a.detach().cpu())
                else:
                    in_args_cpu.append(a)

            # Compile pipeline in separate process to ensure deterministic stmt generation (and thus caching)
            import multiprocessing
            import marshal, pickle
            fname = f'{name}_{ranks}_{param_hash}'
            marshaled_bytecode = marshal.dumps(func.__code__) # input function must not reference globals
            pickled_name = pickle.dumps(func.__name__)
            pickled_arguments = pickle.dumps(func.__defaults__)
            pickled_closure = pickle.dumps(func.__closure__)
            queue = multiprocessing.Queue()
            compile_func = partial(compile_pipeline_process_func, name, device, def_args, arg_names, in_args_cpu, kwargs, debug, marshaled_bytecode, pickled_name, pickled_arguments, pickled_closure, fname, queue, VERBOSITY)
            proc = lp.LightProcess(target=compile_func, save_stdout=False, save_stderr=False) # won't import __main__
            proc.start()
            proc.join()
            outdir = queue.get(timeout=0.5)
            assert outdir.is_dir(), 'Pipeline creation failed'
            
            # Compile pytorch extension
            use_cuda = out.device.type == 'cuda'
            use_mps = out.device.type == 'mps'
            extension = build_pt_exts(outdir, use_cuda, use_mps, ext_name=fname)
            fun = getattr(extension, fname)
            __HL_JIT_CACHE[name][func_key] = fun

        retval = fun(*in_args, out) # kwargs not passed
        if retval != 0:
            raise RuntimeError(f'Non-zero return falue from HL-OP op {name}')
        
        return out
    
    # Save raw function for debugging
    setattr(wrapper, 'original', func)

    return wrapper

def clear_disk_cache():
    """Clear disk cache of pipelines and PyTorch ops"""
    from shutil import rmtree
    rmtree(f'{BUILD_TOP_DIR}/halide_pipelines', ignore_errors=True)
    rmtree(f'{BUILD_TOP_DIR}/torch_ops', ignore_errors=True)

def clear_jit_cache():
    """Clear runtime cache of compiled PyTorch OPs"""
    __HL_JIT_CACHE.clear()

def clear_caches():
    """Clear disk and runtime pipeline and PyTorch OP caches"""
    clear_disk_cache()
    clear_jit_cache()

# Remove forward declaration of scheduled pipeline
# (which is called internally by lib)
def _fix_scheduled_lib_header(pipeline_name: str, header: Union[Path, str]):
    header = Path(header)
    src_in = header.read_text().splitlines()
    src_out = []
    
    inside_broken = False
    n_lines = len(src_in)
    for i in range(n_lines):
        curr = src_in[i]
        next = src_in[i + 1] if i < n_lines - 1 else ''
        
        enter_func = 'HALIDE_FUNCTION_ATTRS' in curr and 'inline int' in next
        enter_correct = enter_func and (f'{pipeline_name}_th_' in next)
        enter_broken = enter_func and not enter_correct
        exit_broken = curr.rstrip() == '}' and inside_broken

        if enter_broken:
            inside_broken = True
            src_out.append('/*')
        
        src_out.append(curr)

        if exit_broken:
            inside_broken = False
            src_out.append('*/')
    
    header.write_text('\n'.join(src_out))

DEBUG_IDX = 0
# Generate pytorch wrapper and pipeline
def build_pipeline(f: hl.Func, inputs: list[hl.Var], target: hl.Target, name: str, verbosity: str):
    if verbosity != 'none':
        print(f'Building pipeline "{name}" for {target}')
    
    # Get build dir based on hash of pipeline contents
    # TODO: statements are non-deterministic w.r.t. variable names (at least)
    # => recompiles unnecessarily often, quite broken in its current state
    stmt_file = Path(f'stmt_{DEBUG_IDX}_{uuid.uuid4().hex}.txt')
    f.compile_to({ hl.OutputFileType.stmt: str(stmt_file) }, inputs, name, target)
    op_hash = md5(stmt_file.read_text().encode()).hexdigest()
    build_dir = BUILD_TOP_DIR / 'halide_pipelines' / name / f'{op_hash}-{str(target)}'
    stmt_file.unlink()

    # Build into tmp dir to avoid issues with concurrent builds
    tmp_dir = BUILD_TOP_DIR / f'tmp-{uuid.uuid4().hex}'

    outputs = {
        # For PT ext compilation
        hl.OutputFileType.pytorch_wrapper: f'{name}.pytorch.h',
        hl.OutputFileType.static_library: f'{name}.{lib_ext}',
        hl.OutputFileType.c_header: f'{name}.h',
        
        # For debugging
        hl.OutputFileType.c_source: f'{name}.c',
        hl.OutputFileType.stmt_html: f'{name}_stmt.html',
        # hl.OutputFileType.compiler_log: f'{name}_log.txt',
        
        # Unused
        # hl.OutputFileType.assembly: "",
        # hl.OutputFileType.bitcode: "",
        # hl.OutputFileType.cpp_stub: "",
        # hl.OutputFileType.featurization: "",
        # hl.OutputFileType.function_info_header: "",
        # hl.OutputFileType.llvm_assembly: "",
        # hl.OutputFileType.objec: "",
        # hl.OutputFileType.python_extension: "",
        # hl.OutputFileType.registration: "",
        # hl.OutputFileType.schedule: "",
        # hl.OutputFileType.stmt: "",
        # hl.OutputFileType.hlpipe: "",
    }

    # Check if already compiled
    if all(Path(build_dir / n).is_file() for n in outputs.values()):
        if VERBOSITY == 'full':
            print(f'Pipeline "{name}" loaded from cache')
        return build_dir
    
    # Compile into tmp dir
    os.makedirs(tmp_dir)
    tmp_outputs = { k: str(tmp_dir / v) for k,v in outputs.items() }
    f.compile_to(tmp_outputs, inputs, name, target) # produces <name>_th_() in pytorch_wrapper
    _fix_scheduled_lib_header(name, tmp_outputs[hl.OutputFileType.pytorch_wrapper])

    # Atomically replace (avoids race conditions)
    os.makedirs(build_dir.parent, exist_ok=True)
    try:
        os.replace(tmp_dir, build_dir) # atomic
    except OSError:
        # source directory already exists, delete tmpdir and its contents.
        shutil.rmtree(tmp_dir)
        if not build_dir.is_dir(): raise

    return build_dir

# Parse input directory for compiled pipelines
def _get_compile_params(input_path: Path, has_cuda: bool, has_mps: bool):
    # Path to a distribution of Halide
    dirs_to_try = [
        Path(hl.__file__).parent / 'include',
        Path(os.environ.get('HALIDE_INCLUDE_DIR', '<missing>')), # from shell.nix
    ]
    
    halide_incls = list(p for p in dirs_to_try if p.is_dir())
    if not halide_incls:
        raise RuntimeError(f'Could not find Halide includes')
    
    # Note that recent versions of PyTorch (at least 1.7.1) requires C++14
    # in order to compile extensions
    compile_args = ['/std:c++17'] if sys.platform == 'win32' else ['-std=c++17', '-g']
    if sys.platform == 'darwin':  # on osx libstdc++ causes trouble
        compile_args += ['-stdlib=libc++']
        compile_args += ['-framework', 'Metal', '-framework', 'Foundation']
        compile_args += ['-ObjC++']

    extra_includes = [input_path, halide_incls[0]]
    hl_libs = []  # Halide op libraries to link to
    hl_headers = []  # Halide op headers to include in the wrapper
    op_names = []

    for hl_src in input_path.glob('*.pytorch.h'):
        op_name = hl_src.name.split('.')[0]
        op_names.append(op_name)

        # Add all Halide-generated libraries
        hl_lib = hl_src.parent / f'{op_name}.{lib_ext}'
        hl_libs.append(str(hl_lib)) # absolute

        hl_header = hl_src.parent / f'{op_name}.h'
        hl_headers.append(hl_header)
        hl_headers.append(hl_src) # also include pytorch.h for hashing

    # C++ wrapper code that includes so that we get all the Halide ops in a
    # single python extension
    wrapper_path = input_path / 'pybind_wrapper.cpp'
    sources = [wrapper_path]
    generate_pybind_wrapper(wrapper_path, op_names, has_cuda, has_mps)

    return {
        'sources': sources,
        'headers': hl_headers,
        'extra_includes': extra_includes,
        'libs': hl_libs,
        'compile_args': compile_args,
    }

def aot_build_pt_exts(input_path: Path, with_cuda: bool, with_mps: bool):
    params = _get_compile_params(input_path, with_cuda, with_mps)
    sources = [str(p) for p in params['sources']]
    include_dirs = [str(i) for i in params['extra_includes']]

    ext_name = 'halide_pt_ops'

    if with_cuda:
        print('Generating CUDA extension')
        from torch.utils.cpp_extension import CUDAExtension
        extension = CUDAExtension(ext_name, sources, #'halide_ops_aot'
                                include_dirs=include_dirs,
                                extra_objects=params['libs'],
                                libraries=['cuda'],  # Halide ops need the full cuda lib, not just the RT library
                                extra_compile_args=params['compile_args'])
    else:
        print('Generating CPU extension')
        from torch.utils.cpp_extension import CppExtension
        extension = CppExtension(ext_name, sources,
                                include_dirs=include_dirs,
                                extra_objects=params['libs'],
                                extra_compile_args=params['compile_args'])

    # Simulate `python setup.py build``
    sys.argv.append('bdist_wheel') # produces library and whl

    # Build the Python extension module
    setup(
        name=ext_name,
        verbose=True,
        url='',
        author_email='erik.harkonen@hotmail.com',
        author='Erik Härkönen',
        version='0.0.1',
        ext_modules=[extension],
        cmdclass={ 'build_ext': BuildExtension }
    )

    out_whl = list(Path('dist').glob(f'{ext_name}*.whl'))
    out_lib = list(Path('build').glob(f'lib*/{ext_name}*'))
    return [*out_whl, *out_lib]

# Build pytorch extension (using setuptools)
# for all pipelines in provided `input_path`
def build_pt_exts(input_path: Path, has_cuda: bool, has_mps: bool, ext_name: str = ''):
    params = _get_compile_params(input_path.resolve(), has_cuda, has_mps)

    extra_libs = []
    
    # WSL: libcuda in non-standard location
    from platform import uname
    if 'microsoft-standard' in uname().release:
        extra_libs.append('-L/usr/lib/wsl/lib')
    
    if has_cuda:
        extra_libs.append('cuda.lib' if sys.platform == 'win32' else '-lcuda')

    # JIT compilation
    built_ext = custom_ops_hl.get_plugin(
        ext_name or 'halide_pt_ops', # name of python import
        params['sources'],
        params['headers'], # for hashing
        extra_cflags=params['compile_args'],
        extra_cuda_cflags=None,
        extra_files_to_hash=[Path(l) for l in params['libs']],
        extra_ldflags=params['libs'] + extra_libs,
        extra_include_paths=params['extra_includes'],
        with_cuda=has_cuda,
        is_python_module=True,
        is_standalone=False,
        keep_intermediates=False,
    )

    return built_ext
