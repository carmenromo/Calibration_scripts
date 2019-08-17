import sys
import argparse
import numpy             as np
import tables            as tb
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
#parser.add_argument("-i", "--files-in", type=str, nargs='+', help="input spectra file", required=True)
parser.add_argument("files_in", type=str, nargs='+', help="input spectra file")

inputs = parser.parse_args()

files    = inputs.files_in
run_nos  = [f[f.find('R')+1:f.find('R')+5] for f in files]
fspecs   = [[] for i in range(12)]
bins     = 0


for i, file in enumerate(files):
    dats   = tb.open_file(file, 'r')
    bins   = np.array(dats.root.HIST.pmt_dark_bins)
    specsL = np.array(dats.root.HIST.pmt_spe) .sum(axis=0)
    for ich in range(12):
        fspecs[ich].append(specsL[ich])


for ich in range(len(fspecs)):
    fig, ax = plt.subplots()
    lspec = fspecs[ich]
    ax.errorbar(bins, lspec[0], xerr=0.5*np.diff(bins)[0], yerr=np.sqrt(lspec), label='R'+str(run_nos[0]), fmt='b.')
    ax.errorbar(bins, lspec[1], xerr=0.5*np.diff(bins)[0], yerr=np.sqrt(lspec), label='R'+str(run_nos[1]), fmt='r.')
    ax.legend(loc=0, fontsize='x-large')

    ax.set_title('Spe response channel '+str(ich))
    ax.set_xlabel('ADC', {'fontsize':16})
    ax.set_ylabel( 'AU', {'fontsize':16})

    plt.show(block=False)
    next_plot = input('press enter to move to next channel')
    if 's' in next_plot:
        plt.savefig('FitPMTCh'+str(ich)+'.png')
    plt.clf()
    plt.close()
