
from torch.utils.ffi import _wrap_function
#from ffiex import _wrap_function
#from torch.utils.cpp_extension import _wrap_function
from ._roi_pooling import lib as _lib, ffi as _ffi

__all__ = []
def _import_symbols(locals):
    for symbol in dir(_lib):
        fn = getattr(_lib, symbol)
        if callable(fn):
            locals[symbol] = _wrap_function(fn, _ffi)
        else:
            locals[symbol] = fn
        __all__.append(symbol)

_import_symbols(locals())
