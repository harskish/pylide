# Create Halide wheel by hand, since setup.py doesn't work

from platform import system
import os
from zipfile import ZipFile, ZIP_STORED
from pathlib import Path
from textwrap import dedent
from shutil import copy2, rmtree, which
from wheel.vendored.packaging import tags # pip install wheel
import hashlib
import subprocess
import sys
import re

HALIDE_ROOT = Path(__file__).parent.parent / 'Halide'
HALIDE_DIR = Path('~/halide-install').expanduser()

# Tag if available, else hash
def get_halide_ver():
    return subprocess.check_output(['git', 'describe', '--always'], cwd=HALIDE_ROOT).decode('utf-8').strip()

# Copy halide files (for wheel or local editable install)
def _copy_files(write: callable):
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
            if not re.search(r'^\s*path @loader_path/ \(offset \d+\)$', curr, re.MULTILINE):
                os.system(f'install_name_tool -add_rpath @loader_path/ {p}') # won't duplicate
        
        # TODO: handle symlinks
        # replace @rpath/XX.dylib with real one in case of symlink
    
    if system() == 'Linux':
        # objdump -x halide_.cpython-310-x86_64-linux-gnu.so | grep RUNPATH
        assert which('patchelf'), 'patchelf executable not in path'

        for p in sobs:
            os.system(f"patchelf --set-rpath '$ORIGIN/' {p}") # works even if no rpath is set
    
    if system() == 'Windows':
        # dumpbin /imports, dumpbin /dependents
        pass

    # Make unique
    names = []
    paths = []
    for p in [*dlls, *pyds, *sobs, *dyls]:
        if p.name not in names:
            names.append(p.name)
            paths.append(p)

    # Check that libs for correct Python version have been compiled
    suffix = sorted(tags.EXTENSION_SUFFIXES, key=lambda v: len(v))[-1] # longest
    lib_name = f'halide_{suffix}' # halide_.cpython-310
    if lib_name not in names:
        raise RuntimeError(f'Halide not compiled for current Python version ({lib_name}), see SETUP.md')

    for f in paths:
        write(f)

    # Copy includes over, keeping folder structure
    for abs in headers:
        write(abs, name=abs.relative_to(HALIDE_DIR).as_posix())

    write(libdir / '__init__.py')
    write(libdir / '_generator_helpers.py')
    write(libdir / 'imageio.py')

def make_wheel():
    assert list(HALIDE_DIR.glob('*')), f'Could not find Halide install at {HALIDE_DIR}'
    halide_ver = get_halide_ver()
    pkg_name = f'halide-{halide_ver}'
    pkg_tag = str(next(tags.sys_tags())) # most specific, e.g. 'cp312-cp312-win_amd64'
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

HL_INSTALLED = False
def make_editable_install(root=Path('~'), add_path=True):
    global HL_INSTALLED
    if HL_INSTALLED or 'halide' in sys.modules:
        return

    if not list(HALIDE_DIR.glob('*')):
        print(f'Could not find Halide install at {HALIDE_DIR}, trying normal import')
        import halide as hl
        return

    dir = root.expanduser().resolve() / 'hl_dev_install'
    try:
        if dir.is_dir():
            dll = (dir / 'halide/Halide.dll')
            if dll.is_file():
                dll.unlink()
            rmtree(dir)
        
        # Copy runtime header files directly (see: Halide/src/runtime/CMakeLists.txt)
        # => can skip time-consuming dummy compile step
        cmakelist_src = (HALIDE_ROOT / 'src/runtime/CMakeLists.txt').read_text()
        RUNTIME_HEADER_FILES = dedent(re.findall(r'set\(RUNTIME_HEADER_FILES\n([\s\S]*?)\)', cmakelist_src)[0]).splitlines()
        for h in RUNTIME_HEADER_FILES:
            copy2(HALIDE_ROOT / f'src/runtime/{h}', HALIDE_DIR / f'include/{h}')

        def write_fun(infile: Path, name=None):
            outname = name or infile.name
            outpath = dir / 'halide' / outname
            os.makedirs(outpath.parent, exist_ok=True)
            copy2(infile, outpath)
            #print('Added', outname)
        _copy_files(write_fun)
    except:
        pass # Windows: Halide.dll probably loaded in another process
    
    if add_path:
        sys.path.insert(0, dir.as_posix())
    
    HL_INSTALLED = True

if __name__ == '__main__':
    make_wheel()