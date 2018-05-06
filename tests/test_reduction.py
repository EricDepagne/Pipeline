#!/usr/bin/env python

# -*- coding: utf-8 -*-


import unittest
from pipeline.stability import set_parameters
from astropy.io import fits


class TestFitsFiles(unittest.TestCase):

    def test_set_parameters():
        file = '../R201704120017.fits'
        parameters = set_parameters(file)

if __name__ == '__main__':
    unittest.main()
