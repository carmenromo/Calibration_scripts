import sys
import numpy             as np
import tables            as tb
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt
from pandas       import DataFrame
from functools    import partial

import invisible_cities.core.fit_functions       as fitf
import invisible_cities.reco.spe_response        as speR
import invisible_cities.io.channel_param_io      as pIO

from   invisible_cities.database                 import load_db           as DB
from   invisible_cities.reco.calib_functions     import seeds_and_bounds
from   invisible_cities.reco.calib_functions     import dark_scaler
from   invisible_cities.reco.calib_functions     import SensorType
from   invisible_cities.types.ic_types           import AutoNameEnumBase
from   invisible_cities.cities.components        import get_run_number

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


#def dark_scaler(dark_spectrum):
#    def scaled_spectrum(x, mu):
#        return np.exp(-mu) * dark_spectrum
#    return scaled_spectrum


#def seeds_db(sensor_type, run_no, n_chann):
#    if sensor_type == 'sipm':
#        gain_seed       = DB.DataSiPM(run_no).adc_to_pes.iloc[n_chann]
#        gain_sigma_seed = DB.DataSiPM(run_no).Sigma     .iloc[n_chann]
#    else:
#        gain_seed       = DB.DataPMT(run_no).adc_to_pes.iloc[n_chann]
#        gain_sigma_seed = DB.DataPMT(run_no).Sigma     .iloc[n_chann]
#    return gain_seed, gain_sigma_seed
#
#
#def poisson_mu_seed(sensor_type, bins, spec, ped_vals, scaler):
#    if sensor_type == 'sipm':
#        gdist         = fitf.gauss(bins[(bins>=-5) & (bins<=5)], *ped_vals)
#        dscale        = spec[(bins>=-5) & (bins<=5)].sum() / gdist.sum()
#        errs          = np.sqrt(spec[(bins>=-5)&(bins<=5)])
#        errs[errs==0] = 1
#        return fitf.fit(scaler,
#                        bins[(bins>=-5)&(bins<=5)],
#                        spec[(bins>=-5)&(bins<=5)],
#                        (dscale), sigma=errs).values[0]
#    
#    dscale = spec[bins<0].sum() / fitf.gauss(bins[bins<0], *ped_vals).sum()
#    return fitf.fit(scaler,
#                    bins[bins<0],
#                    spec[bins<0],
#                    (dscale)).values[0]
#
#
#def sensor_values(sensor_type, n_chann, scaler, spec, bins, ped_vals):
#    if sensor_type == 'sipm':
#        spectra         = spec
#        peak_range      = np.arange(4, 20)
#        min_bin_peak    = 10
#        max_bin_peak    = 22
#        half_peak_width = 5
#        lim_ped         = 10000
#    else:
#        scale           = spec[bins<0].sum() / fitf.gauss(bins[bins<0], *ped_vals).sum()
#        spectra         = spec - fitf.gauss(bins, *ped_vals) * scale
#        peak_range      = np.arange(10, 20)
#        min_bin_peak    = 15
#        max_bin_peak    = 50
#        half_peak_width = 10
#        lim_ped         = 10000
#    return spectra, peak_range, min_bin_peak, max_bin_peak, half_peak_width, lim_ped
#
#
#def pedestal_values(ped_vals, lim_ped, ped_errs):
#    
#    ped_seed     = ped_vals[1]
#    ped_min      = ped_seed - lim_ped * ped_errs[1]
#    ped_max      = ped_seed + lim_ped * ped_errs[1]
#    ped_sig_seed = ped_vals[2]
#    ped_sig_min  = max(0.001, ped_sig_seed - lim_ped * ped_errs[2])
#    ped_sig_max  = ped_sig_seed + lim_ped * ped_errs[2]
#    
#    return ped_seed, ped_sig_seed, ped_min, ped_max, ped_sig_min, ped_sig_max
#
#
#def seeds_and_bounds(sensor_type, run_no, n_chann, scaler, bins, spec, ped_vals,
#                     ped_errs, func='dfunc', use_db_gain_seeds=True):
#    
#    norm_seed = spec.sum()
#    gain_seed, gain_sigma_seed = seeds_db(sensor_type, run_no, n_chann)
#    spectra, p_range, min_b, max_b, hpw, lim_ped = sensor_values(sensor_type, n_chann,
#                                                                 scaler, spec, bins, ped_vals)
#        
#    if not use_db_gain_seeds:
#        pDL  = find_peaks_cwt(spectra, p_range, min_snr=1, noise_perc=5)
#        p1pe = pDL[(bins[pDL]>min_bin) & (bins[pDL]<max_bin)][0]
#        if not p1pe:
#            p1pe = np.argwhere(bins==(min_bin+max_bin)/2)[0][0]
#        else:
#            p1pe = p1pe[spectra[p1pe].argmax()]
#                                 
#        fgaus = fitf.fit(fitf.gauss, bins[p1pe-hpw:p1pe+hpw],
#                        spectra[p1pe-hpw:p1pe+hpw],
#                        seed=(spectra[p1pe], bins[p1pe], 7),
#                        sigma=np.sqrt(spectra[p1pe-hpw:p1pe+hpw]))
#        gain_seed = fgaus.values[1] - ped_vals[1]
#        if fgaus.values[2] <= ped_vals[2]:
#            gain_sigma_seed = 0.5
#        else:
#            gain_sigma_seed = np.sqrt(fgaus.values[2]**2 - ped_vals[2]**2)
#
#    mu_seed = poisson_mu_seed(sensor_type, bins, spec, ped_vals, scaler)
#    if mu_seed < 0: mu_seed = 0.001
#    
#    sd0 = [norm_seed, mu_seed, gain_seed, gain_sigma_seed]
#    bd0 = [[0, 0, 0, 0.001], [1e10, 10000, 10000, 10000]]
#    
#    if 'gau' in func:
#        p_seed, p_sig_seed, p_min, p_max, p_sig_min, p_sig_max = pedestal_values(ped_vals,
#                                                                                 lim_ped, ped_errs)
#        sd0[2:2] = p_seed, p_sig_seed
#        bd0[0][2:2] = p_min, p_sig_min
#        bd0[1][2:2] = p_max, p_sig_max
#
#    sd0 = tuple(sd0)
#    bd0 = [tuple(bd0[0]), tuple(bd0[1])]
#    return sd0, bd0


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

    #    run_no = file_name[file_name.find('R')+1:file_name.find('R')+5]
    #    run_no = int(run_no)
    run_no    = get_run_number(dats)
    #sensor_type = file_name[file_name.find('R')-8:file_name.find('R')-4]
    sensor_type = SensorType.SIPM if 'sipm' in file_name else SensorType.PMT

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

    ## pOut = open('pmtCalParOut_R'+file_name[-7:-3]+'_F'+func_name+'.dat', 'w')
    posRunNo = file_name.find('R')
    pOut = tb.open_file('pmtCalParOut_R'+file_name[posRunNo+1:posRunNo+5]+'_F'+func_name+'.h5', 'w')
    ## pOut.write('InputFile: '+file_name+'\n')
    ## pOut.write('func_name: '+func_name+'\n')
    ## pOut.write('Minimum stats: '+str(min_stat)+'\n')
    ## pOut.write('Pedestal nsig limits: +/-'+str(limit_ped)+'\n')
    ## pOut.write('\n \n')
    ## pOut.write('Parameter order: '+pOrders[func_name]+'\n')
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
        #respF.set_dark_func(dspec[b1:b2], cent=gfitRes.values[1], sig=gfitRes.values[2], scale=scale)
        #respF.redefine_bins(bins[b1:b2])
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


        #import pdb
        #pdb.set_trace()

        dark = dspec[b1:b2]
        scaler_func = dark_scaler(dark[bins[b1:b2]<0])

        seeds, bounds = seeds_and_bounds(sensor_type, run_no, ich, scaler_func, bins[b1:b2],
                                         lspec[b1:b2], ped_vals, 'new', gfitRes.errors,
                                         func='dfunc', use_db_gain_seeds=True)
        print(seeds)
        print(bounds)

        ## The fit
        errs = np.sqrt(lspec[b1:b2])
        if not 'gau' in func_name:
            errs = np.sqrt(errs**2 + np.exp(-2 * seeds[1]) * dspec[b1:b2])
        errs[errs==0] = 1#0.001
        ## rfit = fitf.fit(ffuncs[func_name], bins[b1:b2], lspec[b1:b2], seeds, sigma=errs, bounds=bounds)
        rfit = fitf.fit(respF, bins[b1:b2], lspec[b1:b2], seeds, sigma=errs, bounds=bounds)
        ## plot the result
        plt.errorbar(bins, lspec, xerr=0.5*np.diff(bins)[0], yerr=np.sqrt(lspec), fmt='b.')
        ## plt.plot(bins[b1:b2], rfit.fn(bins[b1:b2]), 'r')
        ## plt.plot(bins[b1:b2], ffuncs[func_name](bins[b1:b2], *seeds), 'g')
        plt.plot(bins[b1:b2], rfit.fn(bins[b1:b2]), 'r')
        plt.plot(bins[b1:b2], respF(bins[b1:b2], *seeds), 'g')
        plt.title('Spe response fit to channel '+str(ich))
        plt.xlabel('ADC')
        plt.ylabel('AU')
        #print('Sensor index: ', ich)
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

