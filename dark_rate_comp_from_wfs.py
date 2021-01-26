import argparse

import numpy  as np
import tables as tb

from invisible_cities.database                     import load_db
from invisible_cities.reco.calib_sensors_functions import subtract_mode


parser = argparse.ArgumentParser(description='Calculation of the dark rate from the waveforms of a selected run.')
parser.add_argument('in_files',     help='Name of the files', nargs='+') #positional argument
parser.add_argument('out_path',     help='Output files path')
parser.add_argument('--detector',   help='Detector name', type=str, required=False, default='new')
parser.add_argument('--min_sample', help='Minimum bin',   type=int, required=False, default=0)
parser.add_argument('--max_sample', help='Maximum bin',   type=int, required=False, default=800)

ns         = parser.parse_args()
in_files   = ns.in_files
out_path   = ns.out_path
detector   = ns.detector
min_sample = ns.min_sample
max_sample = ns.max_sample

run_no    = int(in_files[0][in_files[0].find('run_')+4:in_files[0].find('run_')+8])
num_wf    = int(in_files[0][in_files[0].find('run_')+9:in_files[0].find('run_')+13])
out_file  = f'{out_path}/dark_rate_comp_q_values_R{run_no}_{num_wf}'
conv_gain = load_db.DataSiPM(detector, run_no).adc_to_pes.values[:, np.newaxis]
conv_gain[conv_gain == 0] = 17
conv_gain = conv_gain.reshape(1792)

dark_qs_all = []
for file in in_files:
    dark_rwf = tb.open_file(file)
    for evt_sipm in dark_rwf.root.RD.sipmrwf:
        cwf = subtract_mode(evt_sipm[:,min_sample:max_sample])
        charges_cd = cwf.sum(1)
        dark_qs_all.append(charges_cd/conv_gain)
tqs = np.array(dark_qs_all).T

np.savez(out_file, tqs=tqs)
