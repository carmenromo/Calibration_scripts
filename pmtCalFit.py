import sys
import numpy             as np
import tables            as tb
import matplotlib.pyplot as plt

from functools import partial

import invisible_cities.core.fit_functions   as fitf
import invisible_cities.reco.spe_response    as speR
import invisible_cities.io.channel_param_io  as pIO

from   invisible_cities.database             import load_db           as DB
from   invisible_cities.reco.calib_functions import seeds_and_bounds
from   invisible_cities.reco.calib_functions import dark_scaler
from   invisible_cities.reco.calib_functions import SensorType
from   invisible_cities.types.ic_types       import AutoNameEnumBase
from   invisible_cities.cities.components    import get_run_number

"""
run as:
python pmtCalFit.py <input file name> <function name: ngau, intgau, dfunc, conv> opt: <minimum stats per bin to set fit limits> <number of sigmas for bound on pedestal parameters>
function name meanings:
ngau : 7 gaussians fitted, can be changed adding line respF.nGau = x
intgau : Gaussians until integral < min_integ=100, can be changed adding line respF.min_integ = x
dfunc : scaled dark spectrum + Gaussians up to integ < 100
conv : expicit convolution of dark spectrum with Gaussians up to integ < 100

"""

## this should probably be somehwere useful if it doesn't already exist
def weighted_av_std(values, weights):
    avg = np.average(values, weights=weights)
    var = np.average((values-avg)**2, weights=weights)
    # renormalize
    var = weights.sum() * var / (weights.sum()-1)
    return avg, np.sqrt(var)


def main():
    """ Fitting for pmt response to ~spe led pulses """
    
    file_name = sys.argv[1]
    func_name = sys.argv[2]
    min_stat  = 0
    fix_ped = False
    if len(sys.argv) > 3:
        use_db_gain_seeds = True if 'true' in sys.argv[3] else False
        min_stat = int(sys.argv[4])
    
    dats   = tb.open_file(file_name, 'r')
    bins   = np.array(dats.root.HIST.pmt_dark_bins)
    specsD = np.array(dats.root.HIST.pmt_dark).sum(axis=0)
    specsL = np.array(dats.root.HIST.pmt_spe).sum(axis=0)

    run_no      = get_run_number(dats)
    sensor_type = SensorType.SIPM if 'sipm' in file_name else SensorType.PMT

    ffuncs = {'ngau':speR.poisson_scaled_gaussians(n_gaussians=7),
    'intgau':speR.poisson_scaled_gaussians(min_integral=100),
        'dfunc':partial(speR.scaled_dark_pedestal, min_integral=100),
            'conv':partial(speR.dark_convolution, min_integral=100)}


    pOrders = {'ngau':'norm err poismu err ped err pedSig err gain err 1peSig err', 'intgau':'norm err poismu err ped err pedSig err gain err 1peSig err', 'dfunc':'norm err poismu err gain err 1peSig err', 'conv':'norm err poismu err gain err 1peSig err'}

    fnam = {'ngau':'poisson_scaled_gaussians_ngau', 'intgau':'poisson_scaled_gaussians_min',
        'dfunc':'scaled_dark_pedestal', 'conv':'dark_convolution'}

    ## pOut = open('pmtCalParOut_R'+file_name[-7:-3]+'_F'+func_name+'.dat', 'w')
    posRunNo = file_name.find('R')
    pOut = tb.open_file('pmtCalParOut_R'+file_name[posRunNo+1:posRunNo+5]+'_F'+func_name+'.h5', 'w')

    param_writer = pIO.channel_param_writer(pOut,
                                            sensor_type='pmt',
                                            func_name=fnam[func_name],
                                            param_names=pIO.generic_params)
    test_names = ['normalization', 'Pedestal', 'Pedestal_sig']
    pTest_writer = pIO.channel_param_writer(pOut,
                                        sensor_type='pmt',
                                        func_name='Pedestal_gaussian',
                                        param_names=test_names,
                                        covariance=(3, 3))
    outDict = {}
    testDict = {}

    for ich, (dspec, lspec) in enumerate(zip(specsD, specsL)):
    
        b1 = 0
        b2 = len(dspec)
        if min_stat != 0:
            valid_bins = np.argwhere(lspec>=min_stat)
            b1 = valid_bins[0][0]
            b2 = valid_bins[-1][0]

        outDict[pIO.generic_params[-2]] = (bins[b1], bins[min(len(bins)-1, b2)])
        
        ## Fit the dark spectrum with a Gaussian (not really necessary for the conv option)
        gb0 = [(0, -100, 0), (1e99, 100, 10000)]
        av, rms = weighted_av_std(bins[dspec>100], dspec[dspec>100])
        sd0 = (dspec.sum(), av, rms)
        errs = np.sqrt(dspec[dspec>100])
        errs[errs==0] = 0.0001
        gfitRes = fitf.fit(fitf.gauss, bins[dspec>100], dspec[dspec>100], sd0, sigma=errs, bounds=gb0)
        outDict[pIO.generic_params[2]] = (gfitRes.values[1], gfitRes.errors[1])
        outDict[pIO.generic_params[3]] = (gfitRes.values[2], gfitRes.errors[2])
        
        testDict[test_names[0]] = (gfitRes.values[0], gfitRes.errors[0])
        testDict[test_names[1]] = (gfitRes.values[1], gfitRes.errors[1])
        testDict[test_names[2]] = (gfitRes.values[2], gfitRes.errors[2])
        testDict["covariance"]  = gfitRes.cov
        pTest_writer(ich, testDict)
        
        ## Scale just in case we lost a different amount of integrals in dark and led
        scale = lspec.sum() / dspec.sum()
        print('Scale check: ', scale)

        if 'dfunc' in func_name:
            respF = ffuncs[func_name](dark_spectrum=dspec[b1:b2] * scale,
                                     pedestal_mean=gfitRes.values[1],
                                     pedestal_sigma=gfitRes.values[2])
        elif 'conv' in func_name:
            respF = ffuncs[func_name](dark_spectrum=dspec[b1:b2] * scale,
                                     bins=bins[b1:b2])
        elif fix_ped:
            respF = partial(ffuncs[func_name],
                            pedestal_mean =gfitRes.values[1],
                            pedestal_sigma=gfitRes.values[2])
        else:
            respF = ffuncs[func_name]


        ped_vals = np.array([gfitRes.values[0] * scale, gfitRes.values[1], gfitRes.values[2]])

        dark = dspec[b1:b2]
        scaler_func = dark_scaler(dark[bins[b1:b2]<0])

        seeds, bounds = seeds_and_bounds(sensor_type, run_no, ich, scaler_func, bins[b1:b2],
                                         lspec[b1:b2], ped_vals, 'new', gfitRes.errors,
                                         func='dfunc', use_db_gain_seeds=True)

        ## The fit
        errs = np.sqrt(lspec[b1:b2])
        if not 'gau' in func_name:
            errs = np.sqrt(errs**2 + np.exp(-2 * seeds[1]) * dspec[b1:b2])
        errs[errs==0] = 1#0.001
        rfit = fitf.fit(respF, bins[b1:b2], lspec[b1:b2], seeds, sigma=errs, bounds=bounds)

        ## plot the result
        plt.errorbar(bins, lspec, xerr=0.5*np.diff(bins)[0], yerr=np.sqrt(lspec), fmt='b.')
        plt.plot(bins[b1:b2], rfit.fn(bins[b1:b2]), 'r')
        plt.plot(bins[b1:b2], respF(bins[b1:b2], *seeds), 'g')
        plt.title('Spe response fit to channel '+str(ich))
        plt.xlabel('ADC')
        plt.ylabel('AU')

        outDict[pIO.generic_params[0]] = (rfit.values[0], rfit.errors[0])
        outDict[pIO.generic_params[1]] = (rfit.values[1], rfit.errors[1])
        gIndx = 2
        if 'gau' in func_name:
            gIndx = 4
        outDict[pIO.generic_params[4]] = (rfit.values[gIndx], rfit.errors[gIndx])
        outDict[pIO.generic_params[5]] = (rfit.values[gIndx+1], rfit.errors[gIndx+1])
        outDict[pIO.generic_params[-1]] = (respF.n_gaussians, rfit.chi2)
        param_writer(ich, outDict)
        plt.show(block=False)
        next_plot = input('press enter to move to next fit')
        if 's' in next_plot:
            plt.savefig('FitPMTCh'+str(ich)+'.png')
        plt.clf()
        plt.close()

    pOut.close()


if __name__ == '__main__':
    main()

