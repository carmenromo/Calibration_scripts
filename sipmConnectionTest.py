import sys
import getopt

from calutils import sipm_connectivity_check
from calutils import sipm_rms_check


def shelp():
    print('\n Usage: sipmConnectionTest [-w waveforms] [-e elec -d dark] RUN')
    print()
    print('    Launch connectivity tests for SiPMs. ')
    print()
    print('    Options: ')
    print('      * -w, --waveform: file with SiPM waveforms, RMS comparison ')
    print('      * -e, --elec: electronics spectra file, requires -d be used ')
    print('      * -d, --dark: dark noise spectra file, requires -e be used ')


if __name__ == '__main__':

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'w:e:d:',
                                   ["waveform=", "elec=", "dark="])
    except getopt.GetoptError as err:
        print(err)
        shelp()
        sys.exit()


    waveform_file    = None
    electronics_file = None
    darkcurrent_file = None
    for o, v, in opts:
        if o in ('-w', '--waveform'): waveform_file = v
        elif o in ('-e', '--elec'): electronics_file = v
        elif o in ('-d', '--dark'): darkcurrent_file = v

    try:
        run_no = args[0]
    except IndexError:
        print('Wrong arguments!')
        shelp()
        sys.exit()

    if electronics_file and darkcurrent_file:
        sipm_connectivity_check(electronics_file, darkcurrent_file, run_no)
    elif waveform_file:
        sipm_rms_check(waveform_file)
    else:
        print('Processing not possible with arguments given')
        shelp()
        sys.exit()
