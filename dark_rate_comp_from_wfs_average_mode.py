import argparse

import numpy  as np
import tables as tb

from enum  import Enum
from scipy import stats

from invisible_cities.database                     import load_db


parser = argparse.ArgumentParser(description='Calculation of the dark rate from the waveforms of a selected run.')
parser.add_argument('in_files',     help='Name of the files', nargs='+') #positional argument
parser.add_argument('out_path',     help='Output files path')
parser.add_argument('--detector',   help='Detector name', type=str, required=False, default='new')
parser.add_argument('--min_sample', help='Minimum bin',   type=int, required=False, default=0)
parser.add_argument('--max_sample', help='Maximum bin',   type=int, required=False, default=800)


def mode(wfs, axis=0):
    def wf_mode(wf):
        positive = wf > 0
        return np.bincount(wf[positive]).argmax() if np.count_nonzero(positive) else 0
    return np.apply_along_axis(wf_mode, axis, wfs).astype(float)

ns         = parser.parse_args()
in_files   = ns.in_files
out_path   = ns.out_path
detector   = ns.detector
min_sample = ns.min_sample
max_sample = ns.max_sample

run_no    = int(in_files[0][in_files[0].find('run_')+4:in_files[0].find('run_')+8])
num_wf    = int(in_files[0][in_files[0].find('run_')+9:in_files[0].find('run_')+13])
out_file  = f'{out_path}/dark_rate_comp_q_values_mode_modes_R{run_no}_{num_wf}'
conv_gain = load_db.DataSiPM(detector, run_no).adc_to_pes.values[:, np.newaxis]
conv_gain[conv_gain == 0] = 17
conv_gain = conv_gain.reshape(1792)

dark_qs_all = []
for file in in_files:
    dark_rwf     = tb.open_file(file)
    wfs_evt      = dark_rwf.root.RD.sipmrwf
    modes_in_evt = []
    for evt_sipm in wfs_evt:
        modes_in_evt.append(np.array([mode(wf) for wf in evt_sipm]))
    modes_in_evt_t = np.array(modes_in_evt).T
    mode_per_sipm  = np.array([tuple(map(lambda s_id: stats.mode(s_id)[0][0], modes_in_evt_t))])[0]
    sum_bins       = np.array([(wf.T - mode_per_sipm).T.sum(1) for wf in wfs_evt])
    sum_bins_pes   = sum_bins/conv_gain
    dark_qs_all.append(sum_bins_pes)

a, b, c = np.array(dark_qs_all).shape
tqs = np.reshape(np.array(dark_qs_all), (a*b, c))
np.savez(out_file, tqs=tqs)
