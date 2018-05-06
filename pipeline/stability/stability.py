#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

# sys imports

# python imports
from pathlib import Path
import argparse

# numpy imports
import numpy as np

# astropy imports
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval

# scipy imports
import scipy as sp
from scipy.signal import savgol_filter
from scipy.signal import find_peaks_cwt
# Problems when the boundary order has some intense line, like order 65 and Hα.
from scipy.signal import butter, filtfilt

import numpy.ma as ma

from statsmodels.api import nonparametric

# pandas imports
import pandas as pd

# matplotlib imports
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colorbar as cb


def getshape(orderinf, ordersup):
    """
    In order to derive the shape of a given order, we use one order after and one order before
    We suppose that the variations are continuous.
    Thus approximating order n using orders n+1 and n-1 is close enough from reality
    """
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


class FITS(object):

    def __add__(self, other):
        """
        Defining what it is to add two HRS objects
        """
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
        (newdataminzs, new.datamaxzs) = ZScaleInterval().get_limits(new.data)
        new.data[new.data <= 0] = 0
        new.data = np.asarray(new.data, dtype=dt)
        return new

    def __sub__(self, other):
        """
        Defining what substracting two HRS object is.

        """
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
        (new.dataminzs, new.datamaxzs) = ZScaleInterval().get_limits(new.data)
        new.data[new.data <= 0] = 0
        new.data = np.asarray(new.data, dtype=dt)

        return new

    def __truediv__(self, other):
        """
        Define what it is to divide a HRS object
        """
        new = copy.copy(self)

        if not isinstance(other, HRS):
            if isinstance(other, np.int) or isinstance(other, np.float):
                new.data = self.data / other
            else:
                return NotImplemented
        else:
            new.data = self.data / other.data
# updating the datamin and datamax attributes after the substraction.
        (new.dataminzs, new.datamaxzs) = ZScaleInterval().get_limits(new.data)

        return new


class Order(object):
    """
    Creates an object that defines the position of the orders.
    """
    def __init__(self,
                 hrs='',
                 sigma=5.0):
        self.hrs = hrs
        self.step = 50
        self.sigma = sigma
        self.spversion = sp.__version__
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
        print(self.spversion)
        splitscipyversion = self.spversion.split('.')
# To avoid weird detection, we set any value below zero to zero.
        data = frame.data.copy()
        if not self.got_flat:
            print("Not a flat, can't determine the position of the orders")
            return None
        else:
            pixelstart = self.step
            pixelstop = frame.data.shape[1]
            # step = 20
            xb = np.arange(pixelstart, pixelstop, self.step)
            temp = []
            for pixel in xb:
                if 'LOW' in frame.mode:
                    window = 31
                    polyorder = 5
                    f = 15
                else:
                    window = 37
                    polyorder = 3
                    f = 1
                test = data[:, pixel]
                b, c = np.histogram(np.abs(test))
                mask = test < c[1]/f
                t = ma.array(test, mask=mask)
                if pixel > frame.xpix:
                    print(pixel)
                    break
                filtereddata = savgol_filter(t, window, polyorder)
                xp = find_peaks_cwt(filtereddata, widths=np.arange(1, 20))
# TODO : Older version of scipy output a list and not a numpy array. Test it. Change occurred in version 0.19
                if splitscipyversion[0] == '0' and np.int(splitscipyversion[1]) < 19:
                    xp = np.array(xp)
# We now extract the valid entries from the peaks_cwt()
                x = xp[~t[xp].mask].copy()
                # print(pixel, x)
                temp.append(x)
            print('-----')
            # print(temp)0
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
        The input parameter is a numpy array containing the probable location of the orders.
        It has been filtered to remove the false detection of the algorithm.

        """
        o = np.zeros_like(pts)
        # Detection of the first order shifts.
        p = np.where((pts[2, 1:] - pts[2, :-1]) > 10)[0]
        print('changement à', p, len(p))
# The indices will allow us to know when to switch row in order to follow the orders.
# The first one has to be zero and the last one the size of the orders.
# This is so that the automatic procedure picks them properly
        indices = [0] + list(p+1) + [pts.shape[1]]
        print('\nindices : ', indices)
        for i in range(pts.shape[0]):
            # The orders come in three section, so we coalesce them
            print('indice', i)
            ind = np.arange(i, i - (len(p) + 1), -1) + 1
            ind[np.where(ind <= 0)] = 0
            a = ind > 0
            a = a * 1
            for j in range(len(a)):
                # TODO FIXER CETTE PARTIE LA QUI NE MARCHE PAS ET QUI FOUT LE BORDEL
                # LE COLLAGE DES ORDRES N'EST PAS CORRECT.

                arr1 = pts[i - j, indices[j]:indices[j + 1]] * a[j]
                o[i, indices[j]:indices[j + 1]] = arr1
        return o

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
            gfit = fitter(gaus, y, self.hrs.data[y, self.step * (k + 1)] / self.hrs.data[y, self.step * (k + 1)].max(), verblevel=0)
        except IndexError:
            return
        return gfit

    def _add_gaussian(self, a):
        """ Computes the size of the orders by adding/substracting 2.7 times the standard dev of the gaussian
        fit to the mean of the same fit.
        Returns the lower limit, the center and the upper limit.
        """
        try:
            return(a.mean.value - self.sigma * a.stddev.value, a.mean.value, a.mean.value + self.sigma * a.stddev.value)
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


class HRS(FITS):
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
        self.type = self.header['OBSTYPE']
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
        self._zoom1 = 100

    def __repr__(self):
        color = 'blue'
        if 'HR' in self.chip:
            color = 'red'
        description = 'HRS {color} Frame\nSize : {x}x{y}\nObject : {target}\nMode: {mode}'.format(
            target=self.name,
            color=color,
            x=self.data.shape[0],
            y=self.data.shape[1],
            mode=self.mode)
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

    def _zoom(self, event):
        self.counter += 1
        if self.counter % 4:  # Limiting the frequency of the update to every 4 moves.
            return
        if event.inaxes is self.ax1:
                # Mouse is in subplot 1.
            xinf2 = np.int(event.xdata - self._zoom1)
            xsup2 = np.int(event.xdata + self._zoom1)
            yinf2 = np.int(event.ydata - self._zoom1)
            ysup2 = np.int(event.ydata + self._zoom1)
            ax2data = self.data[yinf2:ysup2, xinf2:xsup2]
            self.plot2.set_data(ax2data)
            self.ax2.figure.canvas.draw()
            self.fig.canvas.blit(self.ax2.bbox)
        self.counter = 0

    def _plot(self, event):
        if event.inaxes is self.ax1:
            ax3ydata = self.data[:, np.int(event.xdata)]
            ax3xdata = self.data[np.int(event.ydata), :]
            if event.button == 1:  # Left button
                self.ax3.cla()
                self.ax3.yaxis.tick_left()
                self.ax3.set_xlabel('Pixel', color=self.ax3.color)
                self.ax3.set_ylabel('Intensity', color=self.ax3.color)
                self.ax3.xaxis.set_label_position('bottom')
                self.ax3.yaxis.set_label_position('left')
                self.ax3.tick_params(axis='y', colors=self.ax3.color)
                self.ax3.plot(ax3ydata, color=self.ax3.color)
                self.ax3.figure.canvas.draw()
            elif event.button == 3:  # Right button
                self.ax4.cla()
                self.ax4.xaxis.tick_top()
                self.ax4.yaxis.tick_right()
                self.ax4.set_xlabel('Pixel', color=self.ax4.color)
                self.ax4.set_ylabel('Intensity', color=self.ax4.color)
                self.ax4.xaxis.set_label_position('top')
                self.ax4.yaxis.set_label_position('right')
                self.ax4.tick_params(axis='y', colors=self.ax4.color)
                self.ax4.plot(ax3xdata, color=self.ax4.color)
                self.ax4.figure.canvas.draw()

    def plot(self, fig=None):
        """
        Creates a matplotlib window to display the frame
        Adds a small window which is a zoom on where the cursor is
        """
        # Peut-être faire une classe Plot() qui se chargera de tout, et faire que plot() appelle Plot().
        # https://matplotlib.org/examples/animation/subplots.html
        # Defining the grid on which the plot will be shown
        grid = gridspec.GridSpec(6, 2)
        if fig is None:
            fig = plt.figure(num="HRS Frame visualisation",
                             figsize=(9.6, 6.4), clear=True)
        self.fig = fig
        self.ax1 = fig.add_subplot(grid[:-1, 0])
        self.ax2 = fig.add_subplot(grid[0:3, 1])
        self.ax3 = fig.add_subplot(grid[3:, 1], label='x')
        self.ax4 = fig.add_subplot(grid[3:, 1], label='y', frameon=False)
        cbax1 = fig.add_subplot(grid[-1, 0])
        fig.subplots_adjust(wspace=0.3, hspace=2.2)

        # Convenience names
        ax1 = self.ax1
        ax2 = self.ax2
        ax3 = self.ax3
        ax3.color = 'xkcd:cerulean'
        ax4 = self.ax4
        ax4.color = 'xkcd:tangerine'
        data = self.data
        zoom = self._zoom1

        # Adding the plots
        self.plot1 = ax1.imshow(data, vmin=self.dataminzs, vmax=self.datamaxzs)
        ax1.title.set_text('CCD')
        cb.Colorbar(ax=cbax1, mappable=self.plot1, orientation='horizontal', ticklocation='bottom')
        zoomeddata = self.data[
            np.int(self.data.shape[0]//2)-zoom:np.int(self.data.shape[0]//2)+zoom,
            np.int(self.data.shape[1]//2)-zoom:np.int(self.data.shape[1]//2)+zoom
            ]
        self.plot2 = ax2.imshow(zoomeddata, vmin=self.dataminzs, vmax=self.datamaxzs)
        # We need to tidy the bottom right plot a little bit first
        # Ax3 first
        self.ax3.tick_params(axis='x', colors=ax3.color)
        self.ax3.tick_params(axis='y', colors=ax3.color)
        self.ax3.xaxis.tick_bottom()
        self.ax3.yaxis.tick_left()
        self.ax3.set_xlabel('Pixel', color=ax3.color)
        self.ax3.set_ylabel('Intensity', color=ax3.color)
        self.ax3.xaxis.set_label_position('bottom')
        self.ax3.yaxis.set_label_position('left')
        # Ax4
        self.ax4.tick_params(axis='x', colors=ax4.color)
        self.ax4.tick_params(axis='y', colors=ax4.color)
        self.ax4.xaxis.tick_top()
        self.ax4.yaxis.tick_right()
        self.ax4.set_xlabel('Pixel', color=ax4.color)
        self.ax4.set_ylabel('Intensity', color=ax4.color)
        self.ax4.xaxis.set_label_position('top')
        self.ax4.yaxis.set_label_position('right')

        self.ax3.plot(self.data[:, np.int(self.data.shape[1]/2)], color=ax3.color)
        self.ax4.plot(self.data[np.int(self.data.shape[0]/2), :], color=ax4.color)
        ax1.figure.canvas.mpl_connect('motion_notify_event', self._zoom)
        ax1.figure.canvas.mpl_connect('button_press_event', self._plot)
        plt.show()


class Master(object):
    """


    Parameters:
    -----------
     masterflat = median( (flat_i - masterbias)/exptime_i ) - masterdark/exptime\n"
"            - background.\n"

    flat : orders
    """

    def makemasterbias(lof):
        blue = []
        red = []
        for b in lof.bias:
            if b.name.startswith('H'):
                blue.append(HRS(b).data)
                blueheader = HRS(b).header
            elif b.name.startswith('R'):
                red.append(HRS(b).data)
                redheader = HRS(b).header
            else:
                continue
        bshape = list(blue[0].shape)
        bshape[:0] = [len(blue)]
        ba = np.concatenate(blue).reshape(bshape)
        mbdata = np.int16(np.average(ba, axis=0))
        mbfile = b.parent/'bluemasterbias.fits'
        rshape = list(red[0].shape)
        rshape[:0] = [len(red)]
        ra = np.concatenate(red).reshape(rshape)
        mrdata = np.int16(np.average(ra, axis=0))
        mrfile = b.parent/'redmasterbias.fits'

        try:
            fits.writeto(mbfile, mbdata, blueheader, overwrite=True)
            fits.writeto(mrfile, mrdata, redheader, overwrite=True)
        except FileNotFoundError:
            fits.writeto(mrfile, mrdata, redheader, overwrite=False)
            fits.writeto(mbfile, mbdata, blueheader, overwrite=False)
        ListOfFiles.update(lof, mbfile)
        ListOfFiles.update(lof, mrfile)

    def makemasterflat(lof):
        blue = []
        red = []
        for b in lof.flat:
            if b.name.startswith('H'):
                blue.append(b.parent/b.name)
            if b.name.startswith('R'):
                red.append(b.parent/b.name)
            else:
                continue
        t = []


class Normalise(object):
    """
    Normalise each order
    """
    def __init__(self,
                 science,
                 specphot):
        self.science = science
        self.specphot = specphot
        self.flatfielded = self.normalise(self.science)
        self.normalised = self.deblaze(self.science)

    def _shape(self, source, field, o, frac=0.05):
        """
        Determine the shape of an order, by using a Locally Weighted Scatterplot Smoothing method
        One could use a polynomial fitting too
        """
        lowess = nonparametric.lowess
        order = source.wlcrorders.Order == o
        # Because of the weird shape of the orders, we need to split the fit into two separate fits
        # The break is at pixel 1650
        bk = 1650
        ya = source.wlcrorders.loc[order, [field]].values.flatten()[:bk]
        xa = source.wlcrorders.loc[order, ['Wavelength']].values.flatten()[:bk]
        yb = source.wlcrorders.loc[order, [field]].values.flatten()[bk:]
        xb = source.wlcrorders.loc[order, ['Wavelength']].values.flatten()[bk:]
        # print(x)
        # print(y)
        lowessfita = lowess(ya, xa, frac=frac)
        lowessfitb = lowess(yb, xb, frac=frac)
        # lowessfit = lowess(source.wlcrorders.Object[order], source.wlcrorders.Wavelength[order], frac=frac)
        return np.concatenate((lowessfita, lowessfitb))

    def normalise(self, science):
        """
        First step, correction of the pixel-pixel variations
        """
        # fshape = self._shape(self.specphot, 'Object')
        # We add one column that will hold the normalisez flux
        science.wlcrorders = science.wlcrorders.assign(FlatField=science.wlcrorders.CosmicRaysObject)
        for order in science.wlcrorders.Order.unique():
            o = science.wlcrorders.Order == order
            fshape = self._shape(self.specphot, 'Object', order)
            science.wlcrorders.loc[science.wlcrorders.Order == order,
                                   ['FlatField']] = science.wlcrorders.CosmicRaysObject[o]/fshape[:, 1]
        return science

    def deblaze(self, science):
        """
        We now deblaze the orders to have their flux set to unity
        """
        # oshape = self._shape(self.science, 'FlatField')
        science.wlcrorders = science.wlcrorders.assign(Normalised=science.wlcrorders.CosmicRaysObject)
        science.wlcrorders = science.wlcrorders.assign(oshape=science.wlcrorders.CosmicRaysObject)
        for order in science.wlcrorders.Order.unique()[2:-1]:
            o = science.wlcrorders.Order == order
            # print(order)
            oshape = self._shape(self.science, 'FlatField', order)
            print(oshape.shape)
            orderlength = science.hrsfile.data.shape[1]
            if orderlength > 4000:
                orderlength = 4040
            if len(oshape) != orderlength:
                os = np.pad(oshape[:, 1], (0, orderlength - len(oshape[:, 1]) % orderlength), 'edge')
            else:
                os = oshape[:, 1]
            print(os.shape)
            science.wlcrorders.loc[science.wlcrorders.Order == order,
                                   ['oshape']] = os
            print('apres', order, os.shape)
            science.wlcrorders.loc[science.wlcrorders.Order == order,
                                   ['Normalised']] = science.wlcrorders.FlatField[o]/os
        return science


class Extract(object):
    """
    With the location of the orders defined, we can now extract the orders from the science frame
    and perform a wavelength calibration

    Parameters:
    -----------
    orderposition : location of the orders.

    sciencedata : HRS Science frame that will be extracted. FITS file.

    save:   if set to True, a file with the extracted content will be created. False (default) prevents saving.
            Anything else is considered False.

    Output:
    -------

    .orders : Numpy array containing the extracted orders
    .worders : Numpy array containing the wavelength calibrated extracted orders
    .wlcrorders : Numpy array containing the wavelength calibrated, cosmic rays corrected orders.
    """
    def __init__(self,
                 orderposition='',
                 hrsscience='',
                 extract=False,
                 save=False):
        # self.orderposition = orderposition
        self.hrsfile = hrsscience
        self.step = orderposition.step
        self.extract = extract

        self.orders = self._extract_orders(
            orderposition.extracted,
            self.hrsfile.data)
        if self.extract:
            self.extraction()
        if self.checksave(save):
            self.save()

    def extraction(self):
        self.worders = self._wavelength(
            self.orders)  # , pyhrsfile, name)
        self.wlcrorders = self._cosmicrays(
            self.worders)

    def checksave(self, save):
        if not isinstance(save, bool):
            print('Save option must be True or False not {save}\nSaving disabled'.format(save=type(save)))
            return False
        return save

    def _cosmicrays(self, orders):
        orders = orders.assign(
            CosmicRaysSky=orders['Sky'])
        orders = orders.assign(
            CosmicRaysObject=orders['Object'])
        for o in orders.Order.unique():
            filt = orders.Order == o
            crs = sigma_clip(orders.Sky[filt])
            cro = sigma_clip(orders.Object[filt])
            orders.loc[orders.Order == o, ['CosmicRaysSky']] = orders.Sky[filt][~crs.mask]
            orders.loc[orders.Order == o, ['CosmicRaysObject']] = orders.Object[filt][~cro.mask]
        return orders

    def _extract_orders(self, positions, data):
        """ positions est un array à 3 dimensions représentant pour chaque point des ordres detectes :
        la limite inférieure,
        le centre,
        la limite supérieure des ordres.
        [:,:,0] est la limite inférieure
        [:,:,1] le centre,
        [:,:,2] la limite supérieure
        """
# TODO penser à mettre l'array en fortran, vu qu'on travaille par colonnes, ça ira plus vite.

        # data = parameters['data']
        orders = np.zeros((positions.shape[0], data.shape[1]))
        npixels = orders.shape[1]
        print(orders.shape)
        x = [i for i in range(npixels)]
        # Sliced orders are not located at the same place as non sliced orders.
        # This is a crude way of correcting that effect.
        # It works, so let's do it that way for now.
        if 'MEDIUM' in self.hrsfile.mode:
            xshift = 6
        elif 'LOW' in self.hrsfile.mode:
            xshift = 0
        for o in range(2, orders.shape[0]):
            print('Extracting order : ', o)
            X = [self.step * (i + 1) for i in range(positions[o, :, 0].shape[0])]
            try:
                foinf = np.poly1d(np.polyfit(X, positions[o, :, 0], 7))
                fosup = np.poly1d(np.polyfit(X, positions[o, :, 2], 7))
                orderwidth = np.floor(np.mean(fosup(x) - foinf(x))).astype(int)
                print("↳ Largeur de l'ordre : {orderwidth}".format(orderwidth=orderwidth))
            except ValueError:
                continue
            for i in x:
                try:
                    orders[o, i] = data[np.int(foinf(i))+xshift: np.int(foinf(i)) + orderwidth + xshift, i].sum()
                except ValueError:
                    continue
# TODO:
# avant de sommer les pixels, tout mettre dans un np.array, pour pouvoir tenir compte de la rotation de la fente.
# Normaliser au nombre de pixels, peut-être.
        return orders

    def _ordersums():
        """
        Do the summation over the order width to obtain the signal
        """
        pass

    def _wavelength(self, extracted_data):
        '''
        In order to get the wavelength solution, we will merge the wavelength solution
        obtained from the pyhrs reduced spectra, with our extracted data
        This is a temporary solution until we have a working wavelength solution.
        '''
        print(self.hrsfile.file.name)
        pyhrsfile = 'p' + self.hrsfile.file.stem + '_obj' + self.hrsfile.file.suffix
        try:
            pyhrs_data = fits.open(self.hrsfile.file.parent/pyhrsfile)
        except FileNotFoundError:
            print("File {hrsfile} not found, can't do a wavelength calibration.".format(
                hrsfile=self.hrsfile.file.parent/pyhrsfile))
            return None
        list_orders = np.unique(pyhrs_data[1].data['Order'])
        dex = pd.DataFrame()

        for o in list_orders[::-1]:
            print('Ordre :', o)
            a = pyhrs_data[1].data[np.where(pyhrs_data[1].data['Order'] == o)[0]]
            line = 2*(int(o) - self.hrsfile.ordershift)
            orderlength = 2048
            if 'HR' in self.hrsfile.chip:
                if o == 53:
                    orderlength = 3269
                else:
                    orderlength = 4040
            try:
                dex = dex.append(pd.DataFrame(
                    {
                        'Wavelength': a['Wavelength'],
                        'Sky': extracted_data[line, :orderlength],
                        'Object': extracted_data[line-1, :orderlength],
                        'Order': [o for i in range(orderlength)]}
                    ))
            except (IndexError, ValueError):
                continue
        dex = dex.reset_index()
        # Reordering the columns
        dex = dex[['Wavelength', 'Object', 'Sky', 'Order']]
        return dex

    def save(self):
        """
        Saving the DataFrame to disk.
        """
        if 'HBDET' in self.hrsfile.chip:
            ext = 'B'
        else:
            ext = 'R'
        name = self.hrsfile.name + '_' + ext + '.csv.gz'
        print('Saving extracted data as {name}'.format(name=name))
        self.wlcrorders.to_csv(name, compression='gzip', index=False)


class ListOfFiles(object):
    """
    List all the  HRS raw files in the directory
    Returns the description of the files
    """
    def __init__(self, datadir):
        self.path = datadir
        self.thar = []
        self.bias = []
        self.flat = []
        self.science = []
        self.object = []
        self.sky = []
        self.specphot = []
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
            print('No Flats and no biases found in {datadir}\nGo to https://hrscal.salt.ac.za/ to download the biases and flats that are needed to reduce your science data.'.format(datadir=self.path))  # noqa

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
        specphot = []
        path = path if path is not None else self.path
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
                    if 'SPST' in h:
                        specphot.append(self.path / item.name)
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
        specphot.sort()

        self.science = science
        self.bias = bias
        self.thar = thar
        self.flat = flat
        self.object = objet
        self.sky = sky
        self.specphot = specphot


if __name__ == "__main__":
    import astropy as ap
    apv = ap.__version__.split('.')
    if apv[0] == '1' and int(apv[1]) <= 4:
        from sys import exit
        print('You need to upgrade to a version of Astropy greater than 1.3.1')
        exit()
    parser = argparse.ArgumentParser(description='HRS Data Reduction pipeline')
    parser.add_argument('-d',
                        '--datadir',
                        help='Directory where the data to be reduced are',
                        default='.')
    args = parser.parse_args()
    datadir = Path(args.datadir)
