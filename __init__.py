"""
This modules processes the HRS data and produces a 1D spectrum
"""
import six
import distutils.version

from .model import FITS
from .window import MainWindow
from .model import Data, ListOfFiles
from .controller import Controller

def compare_versions(a, b):
    "return True if a is greater than or equal to b"
    if a:
        if six.PY3:
            if isinstance(a, bytes):
                a = a.decode('ascii')
            if isinstance(b, bytes):
                b = b.decode('ascii')
        a = distutils.version.LooseVersion(a)
        b = distutils.version.LooseVersion(b)
        return a >= b
    else:
        return False

try:
    import astropy
except ImportError:
    raise ImportError("Matplotlib requires astropy")
else:
    if not compare_versions(astropy.__version__, '2.0.1'):
        raise ImportError(
            "Matplotlib requires astropy>=4.0.1; you have %s"
            % astropy.__version__)
