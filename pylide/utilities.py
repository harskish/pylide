import torch
from socket import gethostname
from platform import system
from pathlib import Path
import os
import time
import numpy as np
import random
from . import dnnlib
from typing import Mapping

def get_default_device():
    mps = getattr(torch.backends, 'mps', None)
    if torch.cuda.is_available():
        return 'cuda'
    elif mps and mps.is_available() and mps.is_built():
        return 'mps'
    else:
        return 'cpu'

# Recursive EasyDict maker
def make_easydict_recursive(d_in):
    d_out = dnnlib.EasyDict()
    
    for k,v in d_in.items():
        if isinstance(v, Mapping): # dict-like
            d_out[k] = make_easydict_recursive(v)
        else:
            d_out[k] = v

    return d_out

# For automated testing.
# Use random seed (based on time) instead of deterministic one.
# This way previously missed bugs can be caught by future runs.
def random_seed(seed=None, silent=False):
    seed = int(time.time()) if seed is None else seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if not silent:
        print('RNG seed:', seed)
    return seed

def hl_init_autosched():
    import halide as hl # type: ignore
    for plugin in ['li2018', 'mullapudi2016', 'adams2019', 'anderson2021']:
        suff = 'dll' if system() == 'Windows' else 'so'
        pref = '' if system() == 'Windows' else 'lib'
        hlroot = Path(hl.__file__).parent
        lib_manual = hlroot / f'{pref}autoschedule_{plugin}.{suff}'
        lib_nix = hlroot.parents[2] / f'{pref}autoschedule_{plugin}.{suff}'
        if lib_nix.is_file():
            hl.load_plugin(str(lib_nix))
        else:
            hl.load_plugin(str(lib_manual))