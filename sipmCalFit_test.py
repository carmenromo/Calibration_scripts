import argparse
import numpy             as np
import tables            as tb
import matplotlib.pyplot as plt

from scipy.signal        import find_peaks_cwt
from functools           import partial
from argparse            import Namespace
from enum                import auto

from   invisible_cities.icaro   .hst_functions   import display_matrix
from   invisible_cities.core    .stat_functions  import poisson_sigma
from   invisible_cities.reco    .calib_functions import seeds_and_bounds
from   invisible_cities.reco    .calib_functions import dark_scaler
from   invisible_cities.reco    .calib_functions import SensorType
from   invisible_cities.types   .ic_types        import AutoNameEnumBase
from   invisible_cities.cities  .components      import get_run_number

from   invisible_cities.database import load_db as DB

import invisible_cities.reco.spe_response   as speR
import invisible_cities.core.fit_functions  as fitf
import invisible_cities.io.channel_param_io as pIO

from numpy.testing import assert_array_equal
from numpy.testing import assert_approx_equal


parser = argparse.ArgumentParser(description='Fit function on SiPM spectra.')
parser.add_argument('file_in',           type=str,  help='input spectra',                      )
parser.add_argument('func_name',         type=str,  help='function that will be used to fit',  default='dfunc')
parser.add_argument('use_db_gain_seeds', type=bool, help='option to take gain values from db', default=True)
parser.add_argument('min_stat',          type=int,  help='min statistics for the peaks',       default=10)
parser.parse_args()
#parser.add_argument('--func-name', type=lambda fun: gettatr(ffuncs, fun), help='function that will be used to fit',  required=False, default='dfunc')

args = parser.parse_args()

def fit_dataset(file_name=None, func_name=None, minStat=None):

    db_file = '/Users/carmenromoluque/IC/invisible_cities/database/localdb.NEWDB.sqlite3'
    
    min_stat = minStat
    optimise = True
    if not file_name:
        optimise          = False
        file_name         = args.file_in
        func_name         = args.func_name
        use_db_gain_seeds = args.use_db_gain_seeds
        min_stat          = args.min_stat

    sipmIn = tb.open_file(file_name, 'r')
    #run_no    = file_name[file_name.find('R')+1:file_name.find('R')+5]
    #run_no    = int(run_no)
    run_no    = get_run_number(sipmIn)
    channs    = DB.DataSiPM(db_file, run_no).SensorID.values
    sens_type = SensorType.SIPM if 'sipm' in file_name else SensorType.PMT

    masked_ch = DB.DataSiPM(db_file, run_no).index[DB.DataSiPM(db_file, run_no).Active==0].values

    if use_db_gain_seeds:
        GainSeeds = DB.DataSiPM(db_file, run_no).adc_to_pes.values
        SigSeeds  = DB.DataSiPM(db_file, run_no).Sigma     .values
        ## Give generic values to previously dead or dodgy channels
        GainSeeds[masked_ch] = 15
        SigSeeds [masked_ch] = 2



    ## Bins are the same for dark and light, just use light for now
    bins   = np.array(sipmIn.root.HIST.sipm_spe_bins)
    ## LED correlated and anticorrelated spectra:
    specsL = np.array(sipmIn.root.HIST.sipm_spe) .sum(axis=0)
    specsD = np.array(sipmIn.root.HIST.sipm_dark).sum(axis=0)
    
    #ffuncs = Namespace(ngau   = speR.poisson_scaled_gaussians(n_gaussians=7),
    #                   intgau = speR.poisson_scaled_gaussians(min_integral=100),
    #                   dfunc  = partial(speR.scaled_dark_pedestal, min_integral=100),
    #                   conv   = partial(speR.dark_convolution, min_integral=100))



    ffuncs = {'ngau'  :speR.poisson_scaled_gaussians(n_gaussians=7),
              'intgau':speR.poisson_scaled_gaussians(min_integral=100),
              'dfunc' :partial(speR.scaled_dark_pedestal, min_integral=100),
              'conv'  :partial(speR.dark_convolution, min_integral=100)}

    ## Loop over the spectra:
    outData = []
    outDict = {}
    llchans = []
    if not optimise:
        fnam = {'ngau':'poisson_scaled_gaussians_ngau', 'intgau':'poisson_scaled_gaussians_min', 'dfunc':'scaled_dark_pedestal', 'conv':'dark_convolution'}
        out_file = tb.open_file('sipmCalParOut_R'+str(run_no)+'_F'+func_name+'.h5', 'w')
        param_writer = pIO.channel_param_writer(out_file, sensor_type='sipm', func_name=fnam[func_name], param_names=pIO.generic_params)

    #knownDead    = [3056, 11009, 12005, 12048, 14010, 22028, 22029, 25049] #12058 not dead anymore
    knownDead    = [3056, 11009, 14010, 16016, 22028, 22029, 25049]
    specialCheck = [1006,  1007,  3000,  3001,  5010,  7000, 22029, 28056, 28057]
        
    for ich, (led, dar) in enumerate(zip(specsL, specsD)):
        if channs[ich] in channs[masked_ch]:
            outData.append([channs[ich], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], 0, 0])
            if not optimise:
                for kname in pIO.generic_params:
                    outDict[kname] = (0, 0)
                param_writer(channs[ich], outDict)
            print('no peaks in dark spectrum, spec ', channs[ich])
            continue

        ## Limits for safe fit
        b1 = 0
        b2 = len(dar)
        if min_stat != 0:
            try:
                valid_bins = np.argwhere(led>=min_stat)
                b1 = valid_bins[ 0][0] # This is due to the nature of np.argwhere. b1 first bin, b2 last bin
                b2 = valid_bins[-1][0]
            except IndexError:
                continue
        outDict[pIO.generic_params[-2]] = (bins[b1], bins[min(len(bins)-1, b2)]) ## Fit limits
        # Seed finding
        peaks_dark = find_peaks_cwt(dar, np.arange(2, 20), min_snr=2)
        if len(peaks_dark) == 0:
            ## Try to salvage in case not a masked channel
            ## Masked channels have al entries in one bin.
            if led[led>0].size == 1:
                outData.append([0., 0., 0., 0., 0., 0., 0.])
                print('no peaks in dark spectrum, spec ', channs[ich])
                continue
            else:
                peaks_dark = np.array([dar.argmax()])

        ## Fit the dark spectrum with a Gaussian (not really necessary for the conv option)
        gb0  = [(0, -100, 0), (1e99, 100, 10000)]
        sd0  = (dar.sum(), 0, 2)
        sel  = np.arange(peaks_dark[0]-5, peaks_dark[0]+5)
        errs = poisson_sigma(dar[sel], default=0.1)

        gfitRes = fitf.fit(fitf.gauss, bins[sel], dar[sel], sd0, sigma=errs, bounds=gb0)
        outDict[pIO.generic_params[2]] = (gfitRes.values[1], gfitRes.errors[1])
        outDict[pIO.generic_params[3]] = (gfitRes.values[2], gfitRes.errors[2])

        ## Scale just in case we lost a different amount of integrals in dark and led
        ## scale = led.sum() / dar.sum()
        scale = 1

        ## Take into account the scale in seed finding (could affect Poisson mu)????
        ped_vals    = np.array([gfitRes.values[0] * scale, gfitRes.values[1], gfitRes.values[2]])
        scaler_func = dark_scaler(dar[b1:b2][(bins[b1:b2]>=-5) & (bins[b1:b2]<=5)])

        seeds, bounds = seeds_and_bounds(sens_type, run_no, ich, scaler_func, bins[b1:b2], led[b1:b2],
                                         ped_vals, 'new', gfitRes.errors, func_name, use_db_gain_seeds)
        if seeds[2] == 0:
            ## Channel was bad but maybe recovered
            seeds, bounds = seeds_and_bounds(sens_type, run_no, ich, scaler_func, bins[b1:b2], led[b1:b2],
                                             ped_vals, 'new', gfitRes.errors, func_name, use_db_gain_seeds=False)

        ## Protect low light channels
        if seeds[1] < 0.2:
            llchans.append(channs[ich])
            ## Dodgy setting of high charge dark bins to zero
            dar[bins>gfitRes.values[1] + 3*gfitRes.values[2]] = 0

        if 'dfunc' in func_name:
            respF = ffuncs[func_name](dark_spectrum  = dar[b1:b2] * scale,
                                      pedestal_mean  = gfitRes.values[1],
                                      pedestal_sigma = gfitRes.values[2])
        elif 'conv' in func_name:
            respF = ffuncs[func_name](dark_spectrum = dar [b1:b2] * scale,
                                      bins          = bins[b1:b2])
        else:
            respF = ffuncs[func_name]


        ## The fit
        errs = poisson_sigma(led, default=0.001)
        if not 'gau' in func_name:
            #errs = np.sqrt(errs**2 + np.exp(-2 * seeds[1]) * dar)
            errs = np.sqrt(errs**2 + np.exp(-2 * seeds[1]) * dar * ((0.1*seeds[1])**2 + 1))

        #print('About to fit channel ', channs[ich])

        #rfit = fitf.fit(respF, bins[b1:b2], led[b1:b2], seeds, sigma=errs[b1:b2], bounds=bounds)
        #chi = rfit.chi2
        #print('Fit values', rfit.values)
        #print('Chi:', chi)

        try:
            rfit = fitf.fit(respF, bins[b1:b2], led[b1:b2], seeds, sigma=errs[b1:b2], bounds=bounds)
            chi = rfit.chi2
        except RuntimeError:
            print('Fit doesnt converge, saving zeros for channel ', ich)
            outData.append([0., 0., 0., 0., 0., 0., 0.])
            if not optimise:
                plt.errorbar(bins, led, xerr=0.5*np.diff(bins)[0], yerr=errs, fmt='b.')
                plt.title('Spe distribution for channel '+str(ich))
                plt.xlabel('ADC')
                plt.ylabel('AU')
                plt.show()
            continue
#        except ValueError:
#            print('Channel: ', ich, channs[ich])
#            print('x0 is infeasible, gain is negative')
#            outData.append([0., 0., 0., 0., 0., 0., 0.])
#            print(seeds)
#            continue


        ## Attempt to catch bad fits and refit (currently only valid for dfunc and conv)
        if chi >= 7 or rfit.values[3] >= 2.5 or rfit.values[3] <= 1:
            ## The offending parameter seems to be the sigma in most cases
            nseed = rfit.values
            nseed[3] = 1.7
            nbound = [(bounds[0][0], bounds[0][1], bounds[0][2], 1),
                      (bounds[1][0], bounds[1][1], bounds[1][2], 2.5)]
            rfit = fitf.fit(respF, bins[b1:b2], led[b1:b2], nseed, sigma=errs[b1:b2], bounds=nbound)
            chi  = rfit.chi2
        #if not optimise:
        if channs[ich] == '25043':
                #if channs[ich] in specialCheck or chi >= 10 or rfit.values[2] < 12 or rfit.values[2] > 19 or rfit.values[3] > 3:
                #    if channs[ich] in specialCheck: print('Special check channel '+str(channs[ich]))
                #    print('Channel fit: ', rfit.values, 'Chi: ', chi)
                plt.errorbar(bins, led, xerr=0.5*np.diff(bins)[0], yerr=errs, fmt='b.')
                plt.plot(bins[b1:b2], respF(bins[b1:b2], *rfit.values), 'r')
                plt.plot(bins[b1:b2], respF(bins[b1:b2], *seeds), 'g')
                plt.title('Spe response fit to channel '+str(channs[ich]))
                plt.xlabel('ADC')
                plt.ylabel('AU')
                plt.show()

        outData.append([channs[ich], rfit.values, rfit.errors, respF.n_gaussians, chi])
        outDict[pIO.generic_params[0]] = (rfit.values[0], rfit.errors[0])
        outDict[pIO.generic_params[1]] = (rfit.values[1], rfit.errors[1])
        gIndx = 2
        if 'gau' in func_name:
            gIndx = 4
        outDict[pIO.generic_params[4]]  = (rfit.values[gIndx]  , rfit.errors[gIndx])
        outDict[pIO.generic_params[5]]  = (rfit.values[gIndx+1], rfit.errors[gIndx+1])
        outDict[pIO.generic_params[-1]] = (respF.n_gaussians, rfit.chi2)
        #outDict[pIO.generic_params[-1]] = (rfit.chi2)
        if not optimise:
            param_writer(channs[ich], outDict)

    ## Couple of plots
    gainIndx = 2
    if 'gau' in func_name:
        gainIndx = 4

    plot_names = ["Gain", "1pe sigma", "Poisson mu", "chi2"]
    pVals = [np.fromiter((ch[1][gainIndx] for ch in outData), np.float),
             np.fromiter((ch[1][gainIndx+1] for ch in outData), np.float),
             np.fromiter((ch[1][1] for ch in outData), np.float),
             np.fromiter((ch[4]    for ch in outData), np.float)]
    if optimise:
        sipmIn.close()
        return pVals
    out_file.close()

    #global scalerChis
    pos_x  = DB.DataSiPM(db_file, run_no).X       .values
    pos_y  = DB.DataSiPM(db_file, run_no).Y       .values
    channs = DB.DataSiPM(db_file, run_no).SensorID.values
    ## vals2D = np.zeros((int((pos_x.max()-pos_x.min())/10)+1, int((pos_y.max()-pos_y.min())/10)+1))
    ## print('shape: ', vals2D.shape)
    ## *_, chis = display_matrix(pos_x, pos_y, pVals[3])
    ## Trampa
    #pVals[3][pVals[3]>10] = 0
    plt.scatter(pos_x, pos_y, c=pVals[3])
    plt.title("Fit chi^2 map")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.colorbar()
    plt.show()
    #mask = np.argwhere((pVals[2]>=2) & (pVals[2]<8))
    #mask = np.argwhere((channs<9000) | (channs>=11000))
    #plt.scatter(pos_x[mask], pos_y[mask], c=pVals[2][mask])
    plt.scatter(pos_x, pos_y, c=pVals[2])
    plt.title("Fit poisson mu")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.colorbar()
    plt.show()
    plt.scatter(pos_x, pos_y, c=pVals[0])
    plt.title("Fit conversion gain")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.colorbar()
    plt.show()
    ## fg2, ax2 = plt.subplots()
    ## p = ax2.pcolor(pos_x, pos_y, scalerChis, cmap=cm.Spectral, vmin=np.abs(scalerChis).min(), vmax=np.abs(scalerChis).max())
    ## plt.colorbar(p, ax=ax2)
    ## fg2.show()
    #plt.hist(scalerChis, bins=1000)
    #plt.show()
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16,6))
    chiVs = pVals[3]
    for ax, val, nm in zip(axes.flatten(), pVals, plot_names):
        ax.hist(val[(chiVs<10) & (chiVs!=0)], bins=100)
        ax.set_title(nm)
    plt.tight_layout()
    fig.show()

    #next_plot = input('press enter to move to next fit')
    #if 's' in next_plot:
    #plt.savefig('FitSiPMCh'+str(i)+'.png')

    input('finished with plots?')

    print('Low light chans:', llchans)

    ## with open(file_name[:-3]+'_Fit_'+func_name+'.dat', 'w') as dbF:
    ##     dbF.write('Minimum statistics: '+str(min_stat)+'\n\n')
    ##     for vals in outData:
    ##         dbF.write(str(vals)+'\n')

if __name__ == '__main__':
    fit_dataset()
