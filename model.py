from astropy.io import fits
from pathlib import Path


class Data(object):
    """
    How to handle data
    """
    def __init__(self, datadir='/home/eric/Temp/'):
        self.datadir = Path(datadir)
        lof = ListOfFiles(self.datadir)
        print('files in {dir} : {lof}'.format(dir=self.datadir, lof=lof))
        print('init', self.datadir, lof.science)

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
        print('path : {path}'.format(path=path))
        for item in path.glob('*.fits'):
            if item.name.startswith('H') or item.name.startswith('R'):
                # print('File : {file}'.format(file=path / item.name))
                with fits.open(path / item.name) as fh:
                    try:
                        h = fh[0].header['PROPID']
                        t = fh[0].header['TIME-OBS']
                        d = fh[0].header['DATE-OBS']
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
