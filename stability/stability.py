#!/usr/bin/env python
# -*- coding: utf-8 -*-

# sys imports

# python imports
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


def find_peaks(parameters):
    pixelstart = 50
    pixelstop = parameters['X2']
    step = 50
    xb = np.arange(pixelstart, pixelstop, step)
    temp = []
    for pixel in xb:
        if pixel > parameters[parameters['chip']]['XPix']:
            print(pixel)
            break
#TODO : Older version of scipy output a list and not a numpy array. Test it.
        xp = find_peaks_cwt(savgol_filter(parameters['data'][:, pixel], 11, 5), widths=np.arange(1, 20))
# The wavelet transform sometimes picks noise. Let's remove it now.
        # m = np.isclose(parameters['data'][:, pixel][xp], np.zeros_like(parameters['data'][:, pixel][xp]), atol=20)
        # print(m)
        # xxp = xp[np.invert(m)]
        plt.scatter(pixel * np.ones(len(xp)), xp, s=30)
        temp.append(xp)
    # Storing the location of the peaks in a numpy array
    size = max([len(i) for i in temp])
    peaks = np.ones((size, len(temp)), dtype=np.int)
    for index in range(len(temp)):
        temp[index].resize(size, refcheck=False)
        peaks[:, index] = temp[index]
# We need to remove the zeros.

    return peaks


def match_orders(sci_data):

	#get wavelength calibration files
	cal_file = fits.open('npH201510210012_obj.fits')
	cal_data = cal_file[1].data

	#check OrderShift
	#if parameters['HRDET']['OrderShift'] != cal_data['Order'][0] and parameters['HBDET']['OrderShift'] != cal_data['Order'][0] :
		#cal_data=correct_orders(cal_data,sci_data) #need to write if necessary
	

	#create temp as a copy of calibrated data
	temp = cal_data

	#Determine which points to remove from sci_data
	excess=np.empty(0,dtype=(int))
	for i in range(1,38):
		excess=np.append(excess,np.array(range(i*2074-27,i*2074-1)))
	
	#returns sci_data without excess data points
	temp['Flux'] = np.delete(sci_data['Flux'],excess)

	return temp
	


def identify_orders(pts):
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
def gaussian_fit(a, k):
    from astropy.modeling import fitting, models
    fitter = fitting.SLSQPLSQFitter()
    gaus = models.Gaussian1D(amplitude=1., mean=a, stddev=5.)
    # print(gaus)
    # print(a, k)
    y1 = a - 25
    y2 = a + 25
    #print(y1, y2)
    y = np.arange(y1, y2)
    gfit = fitter(gaus, y, parameters['data'][y, 50 * (k + 1)] / parameters['data'][y, 50 * (k + 1)].max(), verblevel=0)
    if k == 10:
        print(gfit)
    return gfit


def add_gaussian(a):
    """ Computes the size of the orders by adding/substracting 2.7 times the standard dev of the gaussian
    fit to the mean of the same fit.
    Returns the lower limit, the center and the upper limit.
    """
    #print(a)
    return(a.mean.value - 2.7 * a.stddev.value, a.mean.value, a.mean.value + 2.7 * a.stddev.value)


def find_orders(op):
    """ Computes the location of the orders
    Returns a 3D numpy array
    """
    vgf = np.vectorize(gaussian_fit)
    vadd = np.vectorize(add_gaussian)
    fit = np.zeros_like(op, dtype=object)
    positions = np.zeros((op.shape[0], op.shape[1], 3))
    for i in range(op.shape[1]):
        tt = vgf(op[:, i], i)
        fit[:, i] = tt
    positions[:, :, 0], positions[:, :, 1], positions[:, :, 2] = vadd(fit)
    return positions, fit


def extract_orders(positions, data):
    """ positions est un array à 3 dimensions représentant pour chaque point des ordres detectes la limite inférieure, le centre et la limite supérieure des ordres.
    [:,:,0] est la limite inférieure
    [:,:,1] le centre,
    [:,:,2] la limite supérieure
    """
# TODO penser à mettre l'array en fortran, vu qu'on travaille par colonnes, ça ira plus vite.

    # data = parameters['data']
    orders = np.zeros((positions.shape[0], data.shape[1]))
    nborder = orders.shape[1]
    print(nborder)
    x = [i for i in range(nborder)]
    for o in range(2, orders.shape[0]):
        X = [50 * (i + 1) for i in range(positions[o, :, 0].shape[0])]
        foinf = np.poly1d(np.polyfit(X, positions[o, :, 0], 7))
        fosup = np.poly1d(np.polyfit(X, positions[o, :, 2], 7))
        orderwidth = np.floor(np.mean(fosup(x) - foinf(x))).astype(int)
        # orderwidth = 30
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


def prepare_data(parameters, data, directory):
    obs = fits.open(data)
    if parameters['chip'] == 'HRDET':
        bias = fits.open(directory + 'R201704150021.fits')
        d = obs[0].data  # - bias[0].data
    else:
        bias = fits.open(directory + 'H201704150021.fits')
        d = obs[0].data[::-1, :]  # - bias[0].data

    dt = d[np.int(parameters['Y1']):np.int(parameters['Y2']), np.int(parameters['X1']):np.int(parameters['X2'])]  # - bias[0].data.mean()
# We crudely remove the cosmics by moving all pixels in the highest bin of a 50-bin histogram to the second lowest.
    hist, bins = np.histogram(dt, bins=50)
    d = d - bias[0].data.mean()
    if 'Flat' in parameters['ccdtype']:
        print('Flatfield : removing cosmics')
        d[np.where(dt >= bins[-2])] = bins[2]
    else:
        print('Not a flat, not doing anything')
    d[np.where(dt >= bins[-2])] = bins[2]

    # d[np.where(d<=0)] = 0

    return d


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
        a = pyhrs_data[1].data[np.where(pyhrs_data[1].data['Order'] == o)[0]]
        ax1.plot(a['Wavelength'], a['Flux']*1)
        line = 2*(int(o)-parameters[parameters['chip']]['OrderShift'])
        print(line)
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



if __name__ == "__main__":
    directory = '../'
    hrsfiles = assess_stability(directory)
    # f = 'R201510210012.fits'
    f = hrsfiles['Flat'][1]
    parameters = set_parameters(f)
    if 'HBD'in parameters['chip']:
        print('Blue detector')
        parameters['data'] = parameters['data'][::-1, :]
    else:
	print('Red detector')
	#parameters['data'] = parameters['data'][::-1, :]<--------Not Sure what to put here. please advise
    # tp = fits.open('H201704120017.fits')
    tp = 'H201704120017.fits'
    parameters['data'] = prepare_data(parameters, f, directory)
    # parameters['order'] = fit_orders_pair(parameters['data'])
    # wavelength(order)
