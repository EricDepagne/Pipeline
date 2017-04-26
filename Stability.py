#!/usr/bin/env python
# -*- coding: utf-8 -*-

# sys imports

# python imports
from glob import glob

# numpy imports
import numpy as np

# astropy imports
from astropy.io import fits
from astropy.modeling import fitting, models

# scipy imports
from scipy.signal import savgol_filter
from scipy.signal import find_peaks_cwt

# matplotlib imports
import matplotlib.pylab as plt


def find_thar_files():
    l = []
    ffile = glob('*.fits')
    for f in ffile:
        with fits.open(f) as fh:
            if 'STABLE' in fh[0].header['PROPID']:
                l.append(f)
    return l


def find_peaks(arc):
    goodpeaks = []
    # print('dimensions : {dim}, parameters :{param}'.format(dim=arc.shape, param=parameters))
    cutfiltered = savgol_filter(arc, 11, 3)
    peaks = find_peaks_cwt(cutfiltered, widths=np.arange(1, 20))
    for i in range(peaks.shape[0]-1, 0, -1):
        if cutfiltered[peaks[i]] < parameters[parameters['chip']]['Level'] or (peaks[i]-peaks)[i-1] > parameters[parameters['chip']]['Distance']:  # We find doublets of peaks, in order to fit both sky and object at the same time
            # print('Wrong !{0}, {1}, {2}'.format(i, peaks[i], peaks[i-1]))
            continue
        # print(i, peaks[i], cutfiltered[peaks[i]])
        goodpeaks.append(i)
    return peaks[goodpeaks]


def fit_orders_pair(arcdata):
    # plt.clf()
    cut = arcdata[:, parameters['center']]
    orderpositions = {}
    skyfiberposition = {}
    sciencefiberposition = {}
    # print('peaks : {goodpeaks}'.format(goodpeaks=goodpeaks))
    # plt.plot(cutfiltered)
    # plt.scatter(peaks[goodpeaks], cutfiltered[peaks[goodpeaks]], c='green')
    goodpeaks = find_peaks(cut)
    for i in range(len(goodpeaks)-1):
    # for i in range(6, 24):
        print('Detecting order number {order} pixel. Peak : {peak} at {pixel}'.format(order=i, peak=goodpeaks[i], pixel=cut[goodpeaks[i]]))
        xg = np.arange(goodpeaks[i]-50, goodpeaks[i]+20)
        # print(xg)
        # plt.plot(xg, cutfiltered[peaks[goodpeaks][i]-50:peaks[goodpeaks][i]+20])
        g1 = models.Gaussian1D(amplitude=1., mean=goodpeaks[i], stddev=5)
        g2 = models.Gaussian1D(amplitude=1., mean=goodpeaks[i]-30, stddev=5)
        gg_init = g1 + g2
        fitter = fitting.SLSQPLSQFitter()
    # Pour fitter les deux gaussiennes, il faut normaliser les flux à 1
        gg_fit = fitter(gg_init, xg, cut[xg]/cut[xg].max(), verblevel=0)

        # gg_fit = fitter(gg_init, xg, cutfiltered[xg]/cutfiltered[goodpeaks[i]], verblevel=0)
        # print('Center of the order {order} : {science} and {sky}'.format(order=i, science=gg_fit.mean_0, sky=gg_fit.mean_1))
        sci = gg_fit.mean_0
        sky = gg_fit.mean_1
        skyorder = []
        scienceorder = []
        positions = []
        # print(sci, sky, amp)
        for index in range(180):
            try:
                y = parameters['center']+10*index
            except IndexError:
                print('Out of bounds')
                break
            xmobile = np.arange(sky.value-20, sci.value+20, dtype=np.int)
            ymobile = arcdata[xmobile, y]
            g1 = models.Gaussian1D(amplitude=1., mean=sci, stddev=5)
            g2 = models.Gaussian1D(amplitude=1., mean=sky, stddev=5)
            g = g1 + g2
            gfit = fitter(g, xmobile, ymobile/ymobile.max(), verblevel=0)
            if gfit.mean_0 > parameters['Y'] or gfit.mean_1 > parameters['Y']:
                print('Out of bounds')
                break
            sci = gfit.mean_0
            sky = gfit.mean_1
            # print('Center of fibres at position {p} : sky : {sky}, sci : {sci}.'.format(p=y, sci=sci.value, sky=sky.value))
            skyorder.append(sky.value)
            scienceorder.append(sci.value)
            positions.append(y)
        sciencefiberposition.update({str(i): scienceorder})
        skyfiberposition.update({str(i): skyorder})
        orderpositions.update({str(i): positions})

        # plt.plot(xg, cutfiltered[peaks[goodpeaks[i]]]*gg_fit(xg))

    return sciencefiberposition, skyfiberposition, orderpositions


def assess_stability():
    arclist = find_thar_files()
    print(arclist)
    return arclist


def set_parameters(arcfile):
    print('extracting information from file {arcfile}'.format(arcfile=arcfile))
    ff = fits.open(arcfile)
    parameters = {
            'HBDET': {
                'Level': 800,
                'Distance': 30
                    },
            'HRDET': {
                'Level': 800,
                'Distance': 40
                    },
            'X': ff[0].header['NAXIS1'],
            'Y': ff[0].header['NAXIS2'],
            'center': int(ff[0].header['NAXIS1']/2),
            'chip': ff[0].header['DETNAM'],
            'data': ff[0].data
                }
    return parameters


if __name__ == "__main__":
    arcfiles = assess_stability()
    parameters = set_parameters(arcfiles[3])
    bbias = fits.open('H201704150021.fits')
    rbias = fits.open('R201704150021.fits')
    tp = fits.open('R201704120017.fits')
    # Test with bias removal
    #sci, sky, pos = fit_orders_pair(tp[0].data)
    sci, sky, pos = fit_orders_pair(tp[0].data-rbias[0].data)
