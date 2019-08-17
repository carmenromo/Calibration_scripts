import sys

import numpy             as np
import tables            as tb
import matplotlib.pyplot as plt

#from scipy import signal


def getWF(inF, ipm, ievt):
    """ Get a specific waveform from file """

    return inF.root.RD.pmtrwf[ievt][ipm]


def check_pmt_fft():

    file_name1 = sys.argv[1]
    file_name2 = sys.argv[2]

    with tb.open_file(file_name1) as file1, tb.open_file(file_name1) as file2:
        ## just 10 events for now
        for ievt in range(10):

            for ipm in range(12):

                wf1 = getWF(file1, ipm, ievt)
                wf2 = getWF(file2, ipm, ievt)

                zeroed1 = wf1 - np.mean(wf1[:10000])
                zeroed2 = wf2 - np.mean(wf2[:10000])

                freq = np.fft.rfftfreq(len(zeroed1), d=25E-9)
                ft1 = np.fft.rfft(zeroed1)
                ft2 = np.fft.rfft(zeroed2)

                plt.plot(freq, np.absolute(ft1), label='file1, pmt '+str(ipm))
                plt.plot(freq, np.absolute(ft2), label='file2, pmt '+str(ipm))
                plt.legend()
                plt.show()


if __name__ == '__main__':
    check_pmt_fft()
