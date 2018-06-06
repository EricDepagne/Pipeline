#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pipeline.hrspy import HRS
from astropy.io import fits


def test_FITS():
    file = 'test/R201704090006.fits'
    f = HRS(file)
