import pytest
from pathlib import Path

from pipeline.hrspy import ListOfFiles


def test_listoffile():
    directory = Path('./tests/data')
    lof = ListOfFiles(directory)
    assert(len(lof.bias) == 1)
    assert(len(lof.flat) == 1)
    assert(len(lof.thar) == 1)
    assert(len(lof.science) == 3)
    assert(len(lof.specphot) == 1)
    assert(len(lof.object) == 1)
    assert(len(lof.sky) == 1)
