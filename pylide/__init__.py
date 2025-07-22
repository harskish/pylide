from .create_wheel import make_editable_install
make_editable_install()

import halide as hl # type: ignore
def rdom(*args) -> hl.RDom:
    return hl.RDom([hl.Range(lower, sz) for lower,sz in zip(args[0::2], args[1::2])])