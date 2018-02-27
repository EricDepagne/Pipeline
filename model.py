from astropy.io import fits
from pathlib import Path

import numpy as np

from astropy.visualization import ZScaleInterval


class Data(object):
    """
    How to handle data
    """
    def __init__(self, datadir='/home/eric/Temp/'):
        self.datadir = Path(datadir)
        lof = ListOfFiles(self.datadir)


class ListOfFiles(object):
    """
    List all the  HRS raw files in the directory
    Returns the description of the files
    """
    def __init__(self, datadir):
        self.path = Path(datadir)
        self.thar = []
        self.bias = []
        self.flat = []
        self.science = []
        self.object = []
        self.sky = []
        self.crawl()
        self.calibrations_check()

    def update(self, file):
        """
        This function updates the bias and flat attributes of the listoffile
        in order to take into account the master bias/flats that have been created after
        the datadir has been parsed
        """
        print('updating')
        with fits.open(datadir/file) as fh:
            propid = fh[0].header['propid']
            print(propid)
            print(datadir/file)
            if 'BIAS' in propid and datadir/file not in self.bias:
                self.bias.append(datadir/file)
            else:
                print('File already included')

    def calibrations_check(self):
        if not self.flat and not self.bias:
            print('No Flats and no biases found in {datadir}\nGo to https://hrscal.salt.ac.za/ to download the biases and flats that are needed to reduce your science data.'.format(datadir=self.path))

    def crawl(self, path=None):
        """
        Function that goes through the files in datadir
        and sets the attributes of the ListOfFiles to the proper value:
        self.thar is the list of ThAr files, self.science is the list of science files, aso.
        """
        thar = []
        bias = []
        flat = []
        science = []
        objet = []
        sky = []
        path = path if path is not None else self.path
        for item in path.glob('*.fits'):
            if item.name.startswith('H') or item.name.startswith('R'):
                # print('File : {file}'.format(file=path / item.name))
                with fits.open(path / item.name) as fh:
                    try:
                        h = fh[0].header['PROPID']
                    except KeyError:
                        # no propid, probably not a SALT FITS file.
                        continue
                    if 'STABLE' in h:
                        thar.append(self.path / item.name)
                    if 'CAL_FLAT' in h:
                        flat.append(self.path / item.name)
                    if 'BIAS' in h:
                        bias.append(self.path / item.name)
                    if 'SCI' in h or 'MLT' in h or 'LSP' in h:
                        science.append(self.path / item.name)
            if item.name.startswith('pH') or item.name.startswith('pR'):
                if 'obj' in item.name:
                    objet.append(self.path / item.name)
                if 'sky' in item.name:
                    sky.append(self.path / item.name)

        # files.update({'Science': science, 'ThAr': thar, 'Bias': bias, 'Flat': flat})
# We sort the lists to avoid any side effects
        science.sort()
        bias.sort()
        flat.sort()
        thar.sort()
        objet.sort()
        sky.sort()

        self.science = science
        self.bias = bias
        self.thar = thar
        self.flat = flat
        self.object = objet
        self.sky = sky


class FITS(object):
    """
    Class describing how to handle FITS file.
    """
    def __init__(self, file):
        self.file = file
        with fits.open(self.file) as fh:
            self.header = fh[0].header
            self.data = fh[0].data

    def __add__(self, other):
        """
        Defining what it is to add two HRS objects
        """
        import copy
        new = copy.copy(self)
        dt = new.data.dtype

        if isinstance(other, HRS):
            new.data = np.asarray(self.data + other.data, dtype=np.float64)
        elif isinstance(other, np.int):
            new.data = np.asarray(self.data + other, dtype=np.float64)
        elif isinstance(other, np.float):
            new.data = np.asarray(self.data + other, dtype=np.float64)
        else:
            return NotImplemented
        new.data[new.data <= 0] = 0
        new.data = np.asarray(new.data, dtype=dt)
        return new

    def __sub__(self, other):
        """
        Defining what substracting two HRS object is.

        """
        import copy
        new = copy.copy(self)
        dt = new.data.dtype

        if not isinstance(other, HRS):
            if isinstance(other, np.int) or isinstance(other, np.float):
                new.data = np.asarray(self.data, dtype=np.float64) - other
                print('pas HRS', new.data.dtype, new.data.min(), new.data.max())
            else:
                return NotImplemented
        else:
            new.data = np.asarray(self.data, dtype=np.float64) - np.asarray(other.data, dtype=np.float64)
# updating the datamin and datamax attributes after the substraction.
        new.data[new.data <= 0] = 0
        new.data = np.asarray(new.data, dtype=dt)

        return new

    def __truediv__(self, other):
        """
        Define what it is to divide a HRS object
        """
        import copy
        new = copy.copy(self)

        if not isinstance(other, HRS):
            if isinstance(other, np.int) or isinstance(other, np.float):
                new.data = self.data / other
            else:
                return NotImplemented
        else:
            new.data = self.data / other.data

        return new


class HRS(FITS):
    """
    Class to deal with HRS files.
    the data attribute is such that all frames have the same orientation
    """
    def __init__(self,
                 hrsfile=''):
        self.file = hrsfile
        self.hdulist = fits.open(self.file)
        self.header = self.hdulist[0].header
        self.dataX1 = int(self.header['DATASEC'][1:8].split(':')[0])
        self.dataX2 = int(self.header['DATASEC'][1:8].split(':')[1])
        self.dataY1 = int(self.header['DATASEC'][9:15].split(':')[0])
        self.dataY2 = int(self.header['DATASEC'][9:15].split(':')[1])
        self.mode = self.header['OBSMODE']
        self.name = self.header['OBJECT']
        self.chip = self.header['DETNAM']
        self.data = self.prepare_data(self.file)
        self.shape = self.data.shape
        (self.dataminzs, self.datamaxzs) = ZScaleInterval().get_limits(self.data)
        parameters = {'HBDET': {'OrderShift': 83,
                                'XPix': 2048,
                                'BiasLevel': 690},
                      'HRDET': {'OrderShift': 52,
                                'XPix': 4096,
                                'BiasLevel': 920}}
        self.biaslevel = parameters[self.chip]['BiasLevel']
        self.ordershift = parameters[self.chip]['OrderShift']
        self.xpix = parameters[self.chip]['XPix']
        self.mean = self.data.mean()
        self.std = self.data.std()
        self.counter = 0

    def __repr__(self):
        color = 'blue'
        if 'HR' in self.chip:
            color = 'red'
        description = 'HRS {color} Frame\nSize : {x}x{y}\nObject : {target}'.format(target=self.name, color=color, x=self.data.shape[0], y=self.data.shape[1])
        return description

    def prepare_data(self, hrsfile):
        """
        This method sets the orientation of both the red and the blue files to be the same, which is red is up and right
        """
        d = self.hdulist[0].data
        if self.chip == 'HRDET':
            d = d[:, self.dataX1-1:self.dataX2]
        else:
            d = d[::-1, self.dataX1-1:self.dataX2]
#
        return d
