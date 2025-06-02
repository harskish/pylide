# Create Halide wheel by hand, since setup.py doesn't work

from platform import system
import os
from zipfile import ZipFile, ZIP_STORED
from pathlib import Path
from textwrap import dedent
from shutil import copy2, rmtree, which
import hashlib
import subprocess
import sys
import re

HALIDE_ROOTS = {
    15: Path(__file__).parent / 'Halide_llvm15',
    19: Path(__file__).parent / 'Halide_llvm19',
}
HALIDE_DIRS = {
    15: Path('~/halide-install_llvm15').expanduser(),
    19: Path('~/halide-install_llvm19').expanduser(),
}

# Tag if available, else hash
def get_halide_ver(llvm_ver):
    return subprocess.check_output(['git', 'describe', '--always'], cwd=HALIDE_ROOTS[llvm_ver]).decode('utf-8').strip()

# Copy halide files (for wheel or local editable install)
def _copy_files(write: callable, llvm_ver):
    HALIDE_DIR = HALIDE_DIRS[llvm_ver]
    indir = Path(HALIDE_DIR)
    libdir = Path(HALIDE_DIR) / 'lib/python3/site-packages/halide'

    # Include headers for pytorch OP compilation
    headers = list(indir.glob('include/**/*.h'))

    # Include binary outputs
    dlls = list(indir.glob('**/*.dll'))
    pyds = list(indir.glob('**/*.pyd'))
    sobs = list(indir.glob('**/*.so*'))   # Mac, Linux; allow version postfix
    dyls = list(indir.glob('**/*.dylib')) # Mac

    # Patch install names on macos
    if system() == 'Darwin':
        # otool -L halide_.cpython-310-darwin.so:
        #    @rpath/libHalide.16.dylib
        # otool -l halide_.cpython-310-darwin.so:
        #    path @loader_path/../../../ (offset 12)

        # dylibs are next to .so in wheel
        for p in sobs:
            curr = subprocess.check_output(['otool', '-l', p]).decode('utf-8')
            if not re.search('^\s*path @loader_path/ \(offset \d+\)$', curr, re.MULTILINE):
                os.system(f'install_name_tool -add_rpath @loader_path/ {p}') # won't duplicate
        
        # TODO: handle symlinks
        # replace @rpath/XX.dylib with real one in case of symlink
    
    if system() == 'Linux':
        # objdump -x halide_.cpython-310-x86_64-linux-gnu.so | grep RUNPATH
        assert which('patchelf'), 'patchelf executable not in path'

        for p in sobs:
            os.system(f"patchelf --set-rpath '$ORIGIN/' {p}") # works even if no rpath is set

    # Make unique
    names = []
    paths = []
    for p in [*dlls, *pyds, *sobs, *dyls]:
        if p.name not in names:
            names.append(p.name)
            paths.append(p)

    for f in paths:
        write(f)

    # Copy includes over, keeping folder structure
    for abs in headers:
        write(abs, name=abs.relative_to(HALIDE_DIR).as_posix())

    write(libdir / '__init__.py')
    write(libdir / '_generator_helpers.py')
    write(libdir / 'imageio.py')

def make_wheel(llvm_ver):
    assert list(HALIDE_DIR.glob('*')), f'Could not find Halide install at {HALIDE_DIR}'
    HALIDE_DIR = HALIDE_DIRS[llvm_ver]
    halide_ver = get_halide_ver(llvm_ver)
    pkg_name = f'halide-{halide_ver}'
    pkg_tag = {
        'Darwin': 'cp310-cp310-macosx_12_0_arm64',
        'Windows': 'cp310-cp310-win_amd64',
        'Linux': 'cp310-cp310-manylinux1_x86_64',
    }[system()]
    outfile = Path(f'{pkg_name}-{pkg_tag}.whl').resolve()
    
    whl = ZipFile(outfile, 'w', compression=ZIP_STORED)
    def write_fun(infile: Path, name=None):
        outname = name or infile.name
        whl.write(infile, f'halide/{outname}')
        print('Added', outname)
    _copy_files(write_fun)
    
    whl.writestr(f'{pkg_name}.dist-info/top_level.txt', data='halide')
    whl.writestr(f'{pkg_name}.dist-info/METADATA',
        dedent(f'''
            Metadata-Version: {halide_ver}
            Name: halide
            Version: 1.0
            Summary: Halide python bindings
            Home-page: dunno
            Author: Erik Härkönen
            Author-email: erik.harkonen@hotmail.com
        ''').strip()
    )

    whl.writestr(f'{pkg_name}.dist-info/WHEEL',
        dedent(f'''
            Wheel-Version: 1.0
            Generator: bdist_wheel (0.37.1)
            Root-Is-Purelib: false
            Tag: {pkg_tag}
        ''').strip()
    )
    whl.close()

    infos = []
    shas = []
    with ZipFile(outfile, 'r') as whl: # must reopen?
        infos = whl.infolist()
        shas = []
        for i in infos:
            import base64
            bytes = whl.read(i)
            digest = 'sha256=' + base64.urlsafe_b64encode(
                hashlib.sha256(bytes).digest()
            ).decode('latin1').rstrip('=')
            shas.append(digest)

    with ZipFile(outfile, 'a', compression=ZIP_STORED) as whl:
        record_file = f'{pkg_name}.dist-info/RECORD' 
        whl.writestr(record_file,
            '\n'.join([f'{i.filename},{sha256},{i.file_size}' for sha256, i in zip(shas, infos)]) + f'\n{record_file},,'
        )

    print(f'\nInstall with:\npip install --upgrade --force-reinstall {outfile.as_posix()}')

HL_INSTALLED_VER = None
def make_editable_install(llvm_ver, root=Path('~'), add_path=True):
    global HL_INSTALLED_VER
    if HL_INSTALLED_VER:
        assert HL_INSTALLED_VER == llvm_ver or llvm_ver is None, 'Cannot install multiple versions'
        return
    
    assert llvm_ver is not None, 'Must specify llvm_ver on first call'
    HALIDE_DIR = HALIDE_DIRS[llvm_ver]
    assert list(HALIDE_DIR.glob('*')), f'Could not find Halide install at {HALIDE_DIR}'
    assert llvm_ver in [15, 19]

    dir = root.expanduser().resolve() / 'hl_dev_install'
    if dir.is_dir():
        rmtree(dir)

    # Copy runtime header files directly (see: Halide/src/runtime/CMakeLists.txt)
    # => can skip time-consuming dummy compile step
    cmakelist_src = (HALIDE_ROOTS[llvm_ver] / 'src/runtime/CMakeLists.txt').read_text()
    RUNTIME_HEADER_FILES = dedent(re.findall('set\(RUNTIME_HEADER_FILES\n([\s\S]*?)\)', cmakelist_src)[0]).splitlines()
    for h in RUNTIME_HEADER_FILES:
        copy2(HALIDE_ROOTS[llvm_ver] / f'src/runtime/{h}', HALIDE_DIR / f'include/{h}')

    def write_fun(infile: Path, name=None):
        outname = name or infile.name
        outpath = dir / 'halide' / outname
        os.makedirs(outpath.parent, exist_ok=True)
        copy2(infile, outpath)
        #print('Added', outname)
    _copy_files(write_fun, llvm_ver)
    
    if add_path:
        sys.path.insert(0, dir.as_posix())

    HL_INSTALLED_VER = llvm_ver

if __name__ == '__main__':
    make_wheel()