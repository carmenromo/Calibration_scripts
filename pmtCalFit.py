import sys
import numpy as np
import tables as tb
from scipy.signal import find_peaks_cwt
from pandas import DataFrame
import matplotlib.pyplot as plt
from functools import partial

import invisible_cities.core.fit_functions as fitf
import invisible_cities.reco.spe_response  as speR
from invisible_cities.database import load_db as DB
import invisible_cities.io.channel_param_io as pIO

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

## Probably shite
darr = np.zeros(3)
def scaler(x, mu):
    global darr
    return mu * darr

## Seeding points to avoid awkward functions where not necessary
## Gain and Gain sigma
useSavedSeeds = True
GainSeeds = [21.3, 23.4, 26.0, 25.7, 30.0, 22.7, 25.1, 32.7, 23.1, 25.5, 20.8, 22.0]
SigSeeds  = [11.3, 11.5, 10.6, 11.9, 13.1, 9.9, 11.0, 14.7, 10.6, 10.4, 9.3, 10.0]


def seeds_and_bounds(indx, func, bins, spec, ped_vals, ped_errs, lim_ped):
    
    norm_seed = spec.sum()
    
    ped_seed = ped_vals[1]
    ped_min  = ped_seed - lim_ped * ped_errs[1]
    ped_max  = ped_seed + lim_ped * ped_errs[1]
    #print('ped check: ', ped_seed, ped_min, ped_max)
    
    ped_sig_seed = ped_vals[2]
    ped_sig_min  = max(0.001, ped_sig_seed - lim_ped * ped_errs[2])
    ped_sig_max  = ped_sig_seed + lim_ped * ped_errs[2]
    #print('rms check: ', ped_sig_seed, ped_sig_min, ped_sig_max)
    
    ## Remove the ped prediction and check try to get seeds for 1pe
    # first scale the dark pedestal
    dscale = spec[bins<0].sum() / fitf.gauss(bins[bins<0], *ped_vals).sum()
    GSeed  = 0
    GSSeed = 0

    if not useSavedSeeds: # Case of not having seeds
        l_subtract_d = spec - fitf.gauss(bins, *ped_vals) * dscale
        pDL = find_peaks_cwt(l_subtract_d, np.arange(10, 20), min_snr=1, noise_perc=5)
        print('pDL', pDL)
        p1pe = pDL[(bins[pDL]>15) & (bins[pDL]<50)]
        p1pe = p1pe[spec[p1pe].argmax()]  # The maximum is taken
        ## Now fit a Gaussian
        fgaus = fitf.fit(fitf.gauss, bins[p1pe-10:p1pe+10], l_subtract_d[p1pe-10:p1pe+10],
                         (l_subtract_d[p1pe-10:p1pe+10].max(), bins[p1pe], 7), sigma=np.sqrt(l_subtract_d[p1pe-10:p1pe+10]))
        #print('1pe fit check: ', fgaus.values, fgaus.errors)
        GSeed  = fgaus.values[1]-ped_vals[1]
        GSSeed = np.sqrt(fgaus.values[2]**2 - ped_vals[2]**2)
    else:
        GSeed  = GainSeeds[indx]
        GSSeed = SigSeeds[indx]
    
    ## Test scale
    ftest = fitf.fit(scaler, bins[bins<0], spec[bins<0], (dscale))
    print(ftest)
    print(dscale)
    #print('ftest par = ', ftest.values[0], -np.log(ftest.values[0]))

    if 'gau' in func:
        # There are 6 variables: normalization, pedestal pos., spe mean, poisson mean, pedestal sigma, 1pe sigma
        sd0 = (norm_seed, -np.log(ftest.values[0]), ped_seed, ped_sig_seed, GSeed, GSSeed)
        bd0 = [(0, 0, ped_min, ped_sig_min, 0, 0.001), (1e10, 10000, ped_max, ped_sig_max, 10000, 10000)]
        #print('Seed check: ', sd0)
        return sd0, bd0
    ## The other functions only have four parameters: normalization, spe mean, poisson mean, 1pe sigma
    sd0 = (norm_seed, -np.log(ftest.values[0]), GSeed, GSSeed)
    bd0 = [(0, 0, 0, 0.001), (1e10, 10000, 10000, 10000)]
    #print('Seed check: ', sd0)
    return sd0, bd0

def main():
    """ Fitting for pmt response to ~spe led pulses """
    
    fileName = sys.argv[1]
    funcName = sys.argv[2]
    min_stat  = 0
    limit_ped = 10000.
    fix_ped = False
    if len(sys.argv) > 3:
        min_stat = int(sys.argv[3])
        limit_ped = int(sys.argv[4])
        if limit_ped == 0:
            fix_ped = True
            limit_ped = 10000

    dats = tb.open_file(fileName, 'r')
    bins = np.array(dats.root.HIST.pmt_dark_bins)
    specsD = np.array(dats.root.HIST.pmt_dark).sum(axis=0)
    specsL = np.array(dats.root.HIST.pmt_spe).sum(axis=0)
    ## bins = np.array(dats.root.HIST.pmtdar_bins)
    ## specsD = np.array(dats.root.HIST.pmtdar).sum(axis=0)
    ## specsL = np.array(dats.root.HIST.pmtspe).sum(axis=0)

    #respF = fitf.SensorSpeResponse(bins)
    #ffuncs = {'ngau':respF.set_gaussians, 'intgau':respF.min_integ_gaussians, 'dfunc': respF.scaled_dark_pedestal, 'conv':respF.dark_convolution}
    #pOrders = {'ngau':'norm err ped err gain err poismu err pedSig err 1peSig err', 'intgau':'norm err ped err gain err poismu err pedSig err 1peSig err', 'dfunc':'norm err gain err poismu err 1peSig err', 'conv':'norm err gain err poismu err 1peSig err'}
    ffuncs = {'ngau':speR.poisson_scaled_gaussians(n_gaussians=7),
    'intgau':speR.poisson_scaled_gaussians(min_integral=100),
        'dfunc':partial(speR.scaled_dark_pedestal, min_integral=100),
            'conv':partial(speR.dark_convolution, min_integral=100)}


    pOrders = {'ngau':'norm err poismu err ped err pedSig err gain err 1peSig err', 'intgau':'norm err poismu err ped err pedSig err gain err 1peSig err', 'dfunc':'norm err poismu err gain err 1peSig err', 'conv':'norm err poismu err gain err 1peSig err'}
    ## Not ideal...
    fnam = {'ngau':'poisson_scaled_gaussians_ngau', 'intgau':'poisson_scaled_gaussians_min',
        'dfunc':'scaled_dark_pedestal', 'conv':'dark_convolution'}

    ## pOut = open('pmtCalParOut_R'+fileName[-7:-3]+'_F'+funcName+'.dat', 'w')
    posRunNo = fileName.find('R')
    pOut = tb.open_file('pmtCalParOut_R'+fileName[posRunNo+1:posRunNo+5]+'_F'+funcName+'.h5', 'w')
    ## pOut.write('InputFile: '+fileName+'\n')
    ## pOut.write('FuncName: '+funcName+'\n')
    ## pOut.write('Minimum stats: '+str(min_stat)+'\n')
    ## pOut.write('Pedestal nsig limits: +/-'+str(limit_ped)+'\n')
    ## pOut.write('\n \n')
    ## pOut.write('Parameter order: '+pOrders[funcName]+'\n')
    param_writer = pIO.channel_param_writer(pOut,
                                            sensor_type='pmt',
                                            func_name=fnam[funcName],
                                            param_names=pIO.generic_params)
    test_names = ['normalization', 'Pedestal', 'Pedestal_sig']
    pTest_writer = pIO.channel_param_writer(pOut,
                                        sensor_type='pmt',
                                        func_name='Pedestal_gaussian',
                                        param_names=test_names,
                                        covariance=(3, 3))
    outDict = {}
    testDict = {}

    for i, (dspec, lspec) in enumerate(zip(specsD, specsL)):
    
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
        pTest_writer(i, testDict)
        
        ## Scale just in case we lost a different amount of integrals in dark and led
        scale = lspec.sum() / dspec.sum()
        #print('Scale check: ', scale)
        #respF.set_dark_func(dspec[b1:b2], cent=gfitRes.values[1], sig=gfitRes.values[2], scale=scale)
        #respF.redefine_bins(bins[b1:b2])
        if 'dfunc' in funcName:
            respF = ffuncs[funcName](dark_spectrum=dspec[b1:b2] * scale,
                                     pedestal_mean=gfitRes.values[1],
                                     pedestal_sigma=gfitRes.values[2])
        elif 'conv' in funcName:
            respF = ffuncs[funcName](dark_spectrum=dspec[b1:b2] * scale,
                                     bins=bins[b1:b2])
        elif fix_ped:
            respF = partial(ffuncs[funcName],
                            pedestal_mean =gfitRes.values[1],
                            pedestal_sigma=gfitRes.values[2])
        else:
            respF = ffuncs[funcName]


        ## Take into account the scale in seed finding (could affect Poisson mu)????
        ped_vals = np.array([gfitRes.values[0] * scale, gfitRes.values[1], gfitRes.values[2]])
    
        binR = bins[b1:b2]
        global darr
        darr = dspec[b1:b2] * scale
        darr = darr[binR<0]
        seeds, bounds = seeds_and_bounds(i, funcName, bins[b1:b2], lspec[b1:b2],
                                         ped_vals, gfitRes.errors, limit_ped)

        print(seeds)
        #print(bounds)
            
        ## The fit
        errs = np.sqrt(lspec[b1:b2])
        if not 'gau' in funcName:
            errs = np.sqrt(errs**2 + np.exp(-2 * seeds[1]) * dspec[b1:b2])
        errs[errs==0] = 1#0.001
        ## rfit = fitf.fit(ffuncs[funcName], bins[b1:b2], lspec[b1:b2], seeds, sigma=errs, bounds=bounds)
        rfit = fitf.fit(respF, bins[b1:b2], lspec[b1:b2], seeds, sigma=errs, bounds=bounds)
        ## plot the result
        plt.errorbar(bins, lspec, xerr=0.5*np.diff(bins)[0], yerr=np.sqrt(lspec), fmt='b.')
        ## plt.plot(bins[b1:b2], rfit.fn(bins[b1:b2]), 'r')
        ## plt.plot(bins[b1:b2], ffuncs[funcName](bins[b1:b2], *seeds), 'g')
        plt.plot(bins[b1:b2], rfit.fn(bins[b1:b2]), 'r')
        plt.plot(bins[b1:b2], respF(bins[b1:b2], *seeds), 'g')
        plt.title('Spe response fit to channel '+str(i))
        plt.xlabel('ADC')
        plt.ylabel('AU')
        #print('Sensor index: ', i)
        #print('Fit values: ', rfit.values)
        #print('Fit errors: ', rfit.errors)
        ## print('Number of Gaussians: ', respF.nGau)
        #print('Number of Gaussians: ', respF.n_gaussians)
        #print('Fit chi2: ', rfit.chi2)
        ## pOut.write('Indx: '+str(i)+', params: '+str(np.vstack((rfit.values, rfit.errors)).reshape((-1,), order='F'))+', ngaus = '+str(respF.nGau)+', chi2 = '+str(rfit.chi2)+'\n')
        ##pOut.write('Indx: '+str(i)+', params: '+str(np.vstack((rfit.values, rfit.errors)).reshape((-1,), order='F'))+', ngaus = '+str(respF.n_gaussians)+', chi2 = '+str(rfit.chi2)+'\n')
        outDict[pIO.generic_params[0]] = (rfit.values[0], rfit.errors[0])
        outDict[pIO.generic_params[1]] = (rfit.values[1], rfit.errors[1])
        gIndx = 2
        if 'gau' in funcName:
            gIndx = 4
        outDict[pIO.generic_params[4]] = (rfit.values[gIndx], rfit.errors[gIndx])
        outDict[pIO.generic_params[5]] = (rfit.values[gIndx+1], rfit.errors[gIndx+1])
        outDict[pIO.generic_params[-1]] = (respF.n_gaussians, rfit.chi2)
        param_writer(i, outDict)

        next_plot = input('press enter to move to next fit')
        if 's' in next_plot:
            plt.savefig('FitPMTCh'+str(i)+'.png')
        plt.clf()
        plt.close()

    pOut.close()


if __name__ == '__main__':
    main()

