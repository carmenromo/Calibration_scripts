import numpy  as np
import tables as tb

import matplotlib.pyplot as plt

from invisible_cities.core.core_functions import weighted_mean_and_var

import invisible_cities.database.load_db   as DB

db_file = '/Users/carmenromoluque/IC/invisible_cities/database/localdb.NEWDB.sqlite3'

def plot_spectra(binnings, spectras, errorsx, labels,
                 title='Sensor spectra',log=True):
    """
    Plot a list of spectra as histograms on
    the same canvas labeling with labels
    """
    if log:
        plt.yscale('log')
    for bins, spec, errx, label in zip(binnings, spectras, errorsx, labels):
        plt.errorbar(bins, spec, xerr=errx,
                     yerr=np.sqrt(spec),
                     fmt='.',
                     label=label)
        plt.title(title)
        plt.xlabel('ADC')
        plt.ylabel('AU')
    plt.legend()
    plt.show(block=False)
    

def sipm_connectivity_check(elec_name, dark_name, run_no):
    """
    Compare basic parameters of electronics only
    and dark current spectra and have the user classify
    channels with little difference.

    input:
    elec_name : name of file with electronics only spectra
    dark_name : name of file with dark counts only spectra
    run_no    : run number for classification file
    """

    sensors = DB.DataSiPM(db_file, int(run_no)).SensorID.values

    with tb.open_file(elec_name, 'r') as eFile, \
         tb.open_file(dark_name, 'r') as dFile:

        min_incr = 0.5

        try:
            binsE = np.array(eFile.root.HIST.sipm_dark_bins)
            binsD = np.array(dFile.root.HIST.sipm_dark_bins)

            specsE = np.array(eFile.root.HIST.sipm_dark).sum(axis=0)
            specsD = np.array(dFile.root.HIST.sipm_dark).sum(axis=0)
        except tb.NoSuchNodeError:
            print('Node not found in files, spectra in HIST node required')
            raise

        ## Bin half widths as x errors
        xerrE = 0.5 * np.diff(binsE)[0]
        xerrD = 0.5 * np.diff(binsD)[0]

        with open('bad_channels_'+str(run_no)+'.txt', 'w') as dChan:
            dChan.write('Run \t Channel \t classification \n')

            for sipm, espec, dspec in zip(sensors, specsE, specsD):

                avE, rmsE = weighted_mean_and_var(binsE, espec, unbiased=True)
                avD, rmsD = weighted_mean_and_var(binsD, dspec, unbiased=True)

                mean_low   = avE + min_incr >= avD
                rms_low    = rmsE + min_incr >= rmsD

                if mean_low or rms_low:
                    print("Possible bad channel: "+str(sipm))
                    print("identified as having: ")
                    if mean_low:
                        print('low mean: meanE='+str(avE)+', meanD='+str(avD))
                    if rms_low:
                        print('low rms: rmsE='+str(rmsE)+', rmsD='+str(rmsD))

                    plot_title = 'Elec/dark current comparison ch'+str(sipm)
                    plot_spectra([binsE, binsD],
                                 [espec, dspec],
                                 [xerrE, xerrD],
                                 ['Electronics only', 'Dark current'],
                                 title=plot_title)

                    clif = input("How do you classify this channel? [dead/noisy/suspect/ok]")
                    dChan.write(run_no+' \t '+str(sipm)+' \t '+clif+' \n');
                    plt.clf()
                    plt.close()

            chan = input("Want to check specific channels? [y/n]")
            if chan == 'y':
                chan = input("Which channel number? [num/stop]")
                while chan != 'stop':
                    indx = np.argwhere(sensors==int(chan))
                    if len(indx) == 0:
                        print('Channel not found')
                        continue
                    indx = indx[0][0]

                    plot_title = 'Elec/dark current comparison ch'+chan
                    plot_spectra([binsE, binsD],
                                 [specsE[indx], specsD[indx]],
                                 [xerrE, xerrD],
                                 ['Electronics only', 'Dark current'],
                                 title=plot_title)

                    clif = input("How do you classify this channel? [dead/noisy/suspect/ok]")
                    dChan.write(run_no+' \t '+str(chan)+' \t '+clif+' \n');
                    plt.clf()
                    plt.close()
                    chan = input("Another channel? [num/stop]")
    return


## Event by event simulation. Do you see clear DC peaks in the positve?
def sipm_rms_check(wf_file):
    """
    Simple comparison of rms values to check
    sipm connectivity without making spectra
    """

    # Place holder since rms not read by load_db at the moment
    dice_rms_mins = [2.2, 2.0, 2.1, 2.0, 3.5, 3.1, 2.7,
                     2.5, 3.3, 2.8, 2.5, 2.6, 1.9, 2.4,
                     2.2, 2.1, 3.6, 3.3, 2.6, 2.5, 3.0,
                     2.7, 2.3, 2.5, 2.0, 2.3, 2.1, 2.1]

    with tb.open_file(wf_file, 'r') as data:

        bad_log = {}

        try:
            sipmrwf = data.root.RD.sipmrwf
        except tb.NoSuchNodeError:
            print('No raw wafeforms in file')
            exit()

        ch_nums = [[si['channel'], si['sensorID']]
                   for si in data.root.Sensors.DataSiPM]

        xbins = np.arange(sipmrwf.shape[2])
        xerrs = np.full(sipmrwf.shape[2], 0.5)
        for evt, tp in enumerate(sipmrwf):
            print('Checking event ', evt)
            for chNo, sipm in zip(ch_nums, tp):
                rms = np.std(sipm, ddof=1)

                dice_indx = chNo[1] // 1000 - 1
                if rms < dice_rms_mins[dice_indx]:
                    plot_title = 'Raw waveform for atca ch '+str(chNo[0])+', sensor id '+str(chNo[1])
                    plot_spectra([xbins],
                                 [sipm],
                                 [xerrs],
                                 ['Sensor '+str(chNo[1])],
                                 title=plot_title, log=False)

                    clif = input("How do you classify this channel? [OK/bad] ")
                    plt.clf()
                    plt.close()
                    if 'q' in clif:
                        exit()
                        
                    if 'bad' in clif:
                        if evt not in bad_log.keys():
                            bad_log[evt] = [chNo[1]]
                        else:
                            bad_log[evt].append(chNo[1])

        print('bad channel summary')
        for key, evts in bad_log.items():
            print('Event ', key, ' channels ', evts)

                
