#!/usr/bin/env python
# -*- coding: utf-8 -*-

# sys imports

# python imports
from pathlib import Path
import configparser
from glob import glob

# numpy imports
import numpy as np

# astropy imports
from astropy.io import fits

# scipy imports
from scipy.signal import savgol_filter
from scipy.signal import find_peaks_cwt

# pandas imports
import pandas as pd

# matplotlib imports
import matplotlib.pylab as plt

# astropy imports
from astropy.visualization import ZScaleInterval


def classify_files(directory):
    files = {}
    thar = []
    bias = []
    flat = []
    science = []

    ffile = glob(directory + '*.fits')
    ffile.sort()
    for f in ffile:
        with fits.open(f) as fh:
            try:
                h = fh[0].header['PROPID']
            except KeyError:
                print('no propid, probably not a SALT FITS file.')
                continue
            if 'STABLE' in h:
                thar.append(f)
            if 'CAL_FLAT' in h:
                flat.append(f)
            if 'BIAS' in h:
                bias.append(f)
            if 'SCI' in h or 'MLT' in h or 'LSP' in h:
                science.append(f)
    files.update({'Science': science, 'ThAr': thar, 'Bias': bias, 'Flat': flat})
    return files


def match_orders(sci_data):

        # get wavelength calibration files
        cal_file = fits.open('npH201510210012_obj.fits')
        cal_data = cal_file[1].data

        # check OrderShift
        # if parameters['HRDET']['OrderShift'] != cal_data['Order'][0] and parameters['HBDET']['OrderShift'] != cal_data['Order'][0] :
        # cal_data=correct_orders(cal_data,sci_data) #need to write if necessary
        # create temp as a copy of calibrated data
        temp = cal_data
        # Determine which points to remove from sci_data
        excess = np.empty(0, dtype=(int))
        for i in range(1, 38):
                excess = np.append(excess, np.array(range(i*2074-27, i*2074-1)))
        # returns sci_data without excess data points
        temp['Flux'] = np.delete(sci_data['Flux'], excess)
        return temp


# Un moyen d'aller plus vite, c'est de vectoriser le calcul des fits. Cela se fait avec np.vectorize une fois qu'on a défini des fonctions qui vont faire un calcul sur un élément des tableaux. C'est dans find_orders, vgf et vadd.


def extract_orders(positions, data):
    """ positions est un array à 3 dimensions représentant pour chaque point des ordres detectes la limite inférieure, le centre et la limite supérieure des ordres.
    [:,:,0] est la limite inférieure
    [:,:,1] le centre,
    [:,:,2] la limite supérieure
    """
# TODO penser à mettre l'array en fortran, vu qu'on travaille par colonnes, ça ira plus vite.

    # data = parameters['data']
    orders = np.zeros((positions.shape[0], data.shape[1]))
    npixels = orders.shape[1]
    print(npixels)
    x = [i for i in range(npixels)]
    for o in range(2, orders.shape[0]):
        print('Extracting order : ', o)
        X = [50 * (i + 1) for i in range(positions[o, :, 0].shape[0])]
        try:
            foinf = np.poly1d(np.polyfit(X, positions[o, :, 0], 7))
            fosup = np.poly1d(np.polyfit(X, positions[o, :, 2], 7))
            orderwidth = np.floor(np.mean(fosup(x) - foinf(x))).astype(int)
            print("Largeur de l'ordre : {orderwidth}".format(orderwidth=orderwidth))
        except ValueError:
            continue
        orderwidth = 30
        for i in x:
            orders[o, i] = data[np.int(foinf(i)):np.int(foinf(i)) + orderwidth, i].sum()
# TODO: avant de sommer les pixels, il serait bon de tout mettre dans un np.array, pour pouvoir tenir compte de la rotation de la fente.
# Normaliser au nombre de pixels, peut-être.
    return orders


def assess_stability(directory):
    arclist = classify_files(directory)
    # print(arclist)
    return arclist


def set_parameters(arcfile):
    print('extracting information from file {arcfile}'.format(arcfile=arcfile))
    ff = fits.open(arcfile)
    parameters = {'HBDET': {'Level': 15,
                            'Distance': 30,
                            'OrderShift': 83,
                            'XPix': 2048,
                            'BiasLevel': 690},
                  'HRDET': {'Level': 15,
                            'Distance': 40,
                            'OrderShift': 53,
                            'XPix': 4096,
                            'BiasLevel': 920},
                  'X': ff[0].header['NAXIS1'],
                  'Y': ff[0].header['NAXIS2'],
                  'center': int(ff[0].header['NAXIS1'] / 2),
                  'chip': ff[0].header['DETNAM'],
                  'data': ff[0].data,
                  'mode': ff[0].header['OBSMODE'],
                  'name': ff[0].header['OBJECT'],
                  'X1': int(ff[0].header['DATASEC'][1:8].split(':')[0]),
                  'X2': int(ff[0].header['DATASEC'][1:8].split(':')[1]),
                  'Y1': int(ff[0].header['DATASEC'][9:15].split(':')[0]),
                  'Y2': int(ff[0].header['DATASEC'][9:15].split(':')[1]),
                  'nbpixperstep': 11,
                  'ccdtype': ff[0].header['CCDTYPE']}
    parameters['data'] = parameters['data'][:, parameters['X1']-1:parameters['X2']]
    if 'HBD'in parameters['chip']:
        print('Blue detector')
        parameters['data'] = parameters['data'][::-1, :]
    return parameters


def plot_orders(data):
    orderframe = data['data']
    orderpositions = data['order']
    (vmin, vmax) = ZScaleInterval().get_limits(orderframe)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for o in orderpositions.keys():
        print('Plotting order {o}'.format(o=o))
        if 'X' in o:
            continue
        ysky = []
        yscience = []
        x = []
        if 'X' in o:
            continue
        # for i in range(len(orderpositions[o]['fit'])):
        for pixel in range(orderframe.shape[1]):

            if pixel < orderpositions['X'][0]:
                continue
            try:
                i = orderpositions['X'].index(pixel)
                keepi = i
            except ValueError:
                i = keepi
            if i > len(orderpositions[o]['fit']) - 1:
                continue
            x.append(pixel)
            # We append the center of the order, the lower and the upper limits to ysky and yscience.

            ysky.append(orderpositions[o]['yscience'](pixel))
            ysky.append(orderpositions[o]['yscience'](pixel) - 2.5 * orderpositions[o]['fit'][i].stddev_1)
            ysky.append(orderpositions[o]['yscience'](pixel) + 2.5 * orderpositions[o]['fit'][i].stddev_1)
            yscience.append(orderpositions[o]['ysky'](pixel))
            yscience.append(orderpositions[o]['ysky'](pixel) - 2.5 * orderpositions[o]['fit'][i].stddev_0)
            yscience.append(orderpositions[o]['ysky'](pixel) + 2.5 * orderpositions[o]['fit'][i].stddev_0)
        xlabelleft = 0.25 * (x[-1] + x[0])
        xlabelcentre = 0.5 * (x[-1] + x[0])
        xlabelright = 0.75 * (x[-1] + x[0])
        ylabelleft = orderpositions[o]['yscience'](xlabelleft)
        ylabelcentre = orderpositions[o]['yscience'](xlabelcentre)
        ylabelright = orderpositions[o]['yscience'](xlabelright)
        ax.annotate(o, xy=(xlabelleft, ylabelleft), color='peachpuff')
        ax.annotate(o, xy=(xlabelcentre, ylabelcentre), color='turquoise')
        ax.annotate(o, xy=(xlabelright, ylabelright), color='orange')
        # print(x, y1)
        # plt.plot(x, y1c, 'blue')  # , x, y1b, 'blue', x, y1t, 'blue')
        ax.plot(x, ysky[::3], 'green', x, ysky[1::3], 'blue', x, ysky[2::3], 'blue')
        ax.plot(x, yscience[::3], 'green', x, yscience[1::3], 'red', x, yscience[2::3], 'red')
    ax.imshow(orderframe, vmin=vmin, vmax=vmax)


def wavelength(extracted_data, pyhrs_data, star):
    '''
    In order to get the wavelength solution, we will merge the wavelength solution
    obtained from the pyhrs reduced spectra, with our extracted data
    This is a temporary solution until we have a working wavelength solution.
    '''
    list_orders = np.unique(pyhrs_data[1].data['Order'])
    dex = pd.DataFrame()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    for o in list_orders:
        print('Ordre :', o)
        a = pyhrs_data[1].data[np.where(pyhrs_data[1].data['Order'] == o)[0]]
        ax1.plot(a['Wavelength'], a['Flux']*1)
        line = 2*(int(o)-parameters[parameters['chip']]['OrderShift'])
        ax2.plot(a['Wavelength'], extracted_data[line]-extracted_data[line-1])  # Correction du ciel en meme temps.
        dex = dex.append(pd.DataFrame({'Wavelength': a['Wavelength'], 'Object': extracted_data[line], 'Sky': extracted_data[line-1], 'Order': [o for i in range(2048)]}))
    if 'HBDET' in parameters['chip']:
        ext = 'B'
    else:
        ext = 'R'
    name = star + '_' + ext + '.csv.gz'
    print(name)
    dex.to_csv(name, compression='gzip')
# Removing the duplicate indices from the append()
    dex = dex.reset_index()
# Reordering the columns
    dex = dex[['Wavelength', 'Object', 'Sky', 'Order']]
    return dex


def getshape(orderinf, ordersup):
    """
    In order to derive the shape of a given order, we use one order after and one order before
    We suppose that the variations are continuous.
    Thus approximating order n using orders n+1 and n-1 is close enough from reality
    """
    # Problems when the boundary order has some intense line, like order 65 and Hα.
    from scipy.signal import butter, filtfilt
    # Now we find the shape, it needs several steps and various fitting/smoothing
    b, a = butter(10, 0.025)
    # the shape of the orders does not vary much between orders, but there can be cosmic rays or emission lines
    # Since it's unlikely that the same pixel is affected on two non contiguous orders, we pick the minimum of the
    # two orders, and we create an artificial order between them and it's the one that we'll fit.
    shape = np.minimum(orderinf, ordersup)
    ysh2 = filtfilt(b, a, shape)
    x = np.arange(ysh2.shape[0])
    ysh3 = np.poly1d(np.polyfit(x, ysh2, 11))(x)
    ysh4 = np.minimum(ysh2, ysh3)
    ysh5 = np.poly1d(np.polyfit(x, ysh4, 11))(x)
    # ysh5 fits now the shape of the ThAr order quite well, and we can start from there to identify the lines.
    return ysh5


class Reduced(object):
    pass


class Order(object):
    """
    Creates an object that defines the position of the orders.
    """
    def __init__(self,
                 hrs=''):
        self.hrs = hrs
        self.got_flat = self.check_type(self.hrs)
        self.orderguess = self.find_peaks(self.hrs)
        self.order = self.identify_orders(self.orderguess)
        self.extracted, self.order_fit = self.find_orders(self.order)

    def check_type(self, frame):
        if 'Flat field' not in frame.name:
            return False
        return True

    def find_peaks(self, frame):
        """
        Identifies in a Flat-Field frame where the orders are located
        The procedure is as follows:
        1 -
        """
        if not self.got_flat:
            print("Not a flat, can't determine the position of the orders")
            return None
        else:
            pixelstart = 50
            pixelstop = frame.dataX2
            step = 50
            xb = np.arange(pixelstart, pixelstop, step)
            temp = []
            for pixel in xb:
                if pixel > frame.xpix:
                    print(pixel)
                    break
# TODO : Older version of scipy output a list and not a numpy array. Test it.
                xp = find_peaks_cwt(savgol_filter(frame.data[:, pixel], 31, 5), widths=np.arange(1, 20))
                if pixel == 3050:
                    print(xp, pixel)
            #    plt.scatter(pixel * np.ones(len(xp)), xp, s=30)
                temp.append(xp)
            # Storing the location of the peaks in a numpy array
            size = max([len(i) for i in temp])
            peaks = np.ones((size, len(temp)), dtype=np.int)
            for index in range(len(temp)):
                temp[index].resize(size, refcheck=False)
                peaks[:, index] = temp[index]

            return peaks

    def identify_orders(self, pts):
        """
        This function extracts the real location of the orders
        The input parameter is a numpy array containing the probable location of the orders. It has been filtered to remove the false detection of the algorithm.

        """
        o = np.zeros_like(pts)
        # Detection of the first order shifts.
        gr = np.where(np.gradient(pts[0]) > np.gradient(pts[0]).std())[0]
        p = gr[1::2]

        print('changement à', p, len(p))
# The indices will allow us to know when to switch row in order to follow the orders.
# The first one has to be zero and the last one the size of the orders, so that the automatic procedure picks them properly
        indices = [0] + list(p) + [pts.shape[1]]
        print('indices : ', indices)
        for i in range(73):
            # The orders come in three section, so we coalesce them
            print('indice', i)
            ind = np.arange(i, i - (len(p) + 1), -1) + 1
            ind[np.where(ind <= 0)] = 0
            a = ind > 0
            a = a * 1
            for j in range(len(a)):
                print('j:', j)
                print(indices[j] * 50, indices[j + 1] * 50)
                arr1 = pts[i - j, indices[j]:indices[j + 1]] * a[j]
                o[i, indices[j]:indices[j + 1]] = arr1
        return o

# Un moyen d'aller plus vite, c'est de vectoriser le calcul des fits. Cela se fait avec np.vectorize une fois qu'on a défini des fonctions qui vont faire un calcul sur un élément des tableaux. C'est dans find_orders, vgf et vadd.
    def _gaussian_fit(self, a, k):
        from astropy.modeling import fitting, models
        fitter = fitting.SLSQPLSQFitter()
        gaus = models.Gaussian1D(amplitude=1., mean=a, stddev=5.)
        # print(gaus)
        # print(a, k)
        y1 = a - 25
        y2 = a + 25
        y = np.arange(y1, y2)
        try:
            gfit = fitter(gaus, y, self.hrs.data[y, 50 * (k + 1)] / self.hrs.data[y, 50 * (k + 1)].max(), verblevel=0)
        except IndexError:
            return
        return gfit

    def _add_gaussian(self, a):
        """ Computes the size of the orders by adding/substracting 2.7 times the standard dev of the gaussian
        fit to the mean of the same fit.
        Returns the lower limit, the center and the upper limit.
        """
        try:
            return(a.mean.value - 2.7 * a.stddev.value, a.mean.value, a.mean.value + 2.7 * a.stddev.value)
        except AttributeError:
            return(np.nan, np.nan, np.nan)

    def find_orders(self, op):
        """ Computes the location of the orders
        Returns a 3D numpy array
        """
        vgf = np.vectorize(self._gaussian_fit)
        vadd = np.vectorize(self._add_gaussian)
        fit = np.zeros_like(op, dtype=object)
        positions = np.zeros((op.shape[0], op.shape[1], 3))
        for i in range(op.shape[1]):
            tt = vgf(op[:, i], i)
            fit[:, i] = tt
        positions[:, :, 0], positions[:, :, 1], positions[:, :, 2] = vadd(fit)
        return positions, fit


class HRS(object):
    """
    Class that allows to set the parameters of each files
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
                      'HRDET': {'OrderShift': 53,
                                'XPix': 4096,
                                'BiasLevel': 920}}
        self.biaslevel = parameters[self.chip]['BiasLevel']
        self.ordershift = parameters[self.chip]['OrderShift']
        self.xpix = parameters[self.chip]['XPix']

    def __repr__(self):
        color = 'blue'
        if 'HR' in self.chip:
            color = 'red'
        description = 'HRS {color} Frame\nSize : {x}x{y}, Object : {target}'.format(target=self.name, color=color, x=self.data.shape[0], y=self.data.shape[1])
        return description

    def __add__(self, other):
        """
        Defining what it is to add two HRS objects
        """
        import copy
        new = copy.copy(self)

        if isinstance(other, HRS):
            new.data = self.data + other.data
        elif isinstance(other, np.int):
            new.data = self.data + other
        elif isinstance(other, np.float):
            new.data = self.data + other
        else:
            return NotImplemented
        (newdataminzs, new.datamaxzs) = ZScaleInterval().get_limits(new.data)
        return new

    def __sub__(self, other):
        """
        Defining what substracting two HRS object is.

        """
        import copy
        new = copy.copy(self)

        if not isinstance(other, HRS):
            if isinstance(other, np.int) or isinstance(other, np.float):
                new.data = self.data - other
            else:
                return NotImplemented
        else:
            new.data = self.data - other.data
# updating the datamin and datamax attributes after the substraction.
        (new.dataminzs, new.datamaxzs) = ZScaleInterval().get_limits(new.data)

        return new

    def prepare_data(self, hrsfile):
        """
        This method sets the orientation of both the red and the blue files to be the same, which is red is up and right
        """
        d = self.hdulist[0].data
        if self.chip == 'HRDET':
            d = d
        else:
            d = d[::-1, :]
#
        return d


if __name__ == "__main__":
    # TODO : préparer un objet qui contiendra la configuration complete: répertoire ou se trouvent les données, listera les calibrations à utiliser une fois que le fichier à réduire aura été choisi, préparera les données, etc.
    print('HRS Data reduction pipeline')
