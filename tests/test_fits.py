import pytest
from pathlib import Path
import numpy as np

from pipeline.hrspy import ListOfFiles, HRS


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


def test_hrs():
    pass
    directory = Path('./tests/data')
    hrs = HRS(directory/'test.fits')
    assert (hrs.dataX1 == 27)
    assert (hrs.dataX2 == 2074)
    assert (hrs.dataY1 == 1)
    assert (hrs.dataY2 == 4102)
    assert (hrs.mode == 'MEDIUM RESOLUTION')
    assert (hrs.type == 'Science')
    assert (hrs.name == 'P3J061950.0-531212')
    assert (hrs.chip == 'HBDET')
    assert (np.isclose(hrs.dataminzs, 697))
    assert (np.isclose(hrs.datamaxzs, 725.65))

def test_normalise():
    directory = Path('./tests/data')
    