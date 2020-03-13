import sys
import argparse
import tables   as tb
import numpy    as np

from invisible_cities.reco.peak_functions import indices_and_wf_above_threshold


def get_run_number(filename):
    with tb.open_file(filename) as file:
        run_info   = file.root.Run.runInfo
        run_number = run_info.cols.run_number[0]
    return run_number

def get_number_of_events(filename):
    with tb.open_file(filename) as file:
        pmtrd    = file.root.RD. pmtrwf
        sipmrd   = file.root.RD.sipmrwf
        run_evts = file.root.Run.events

        nevt_pmt   =  pmtrd.shape[0]
        nevt_sipm  = sipmrd.shape[0]
        nevt_run   = run_evts.cols.evt_number.shape[0]

    ok = nevt_pmt == nevt_sipm == nevt_run
    return (nevt_pmt if ok else 0)

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', type=str, nargs="*", help="input files"        )
    parser.add_argument('charge_thr' , type=int,            help="threshold in charge")
    parser.add_argument('data_path'  , type=str,            help="output files path"  )
    return parser.parse_args()

arguments   = parse_args(sys.argv)
input_files = arguments.input_files
charge_thr  = arguments.charge_thr
data_path   = arguments.data_path

run_number = get_run_number(input_files[0])
outf_name  = f'spec_R{run_number}_thr{charge_thr}pes.npz'
out_file   = data_path + '/' + outf_name

charges_all_sensors = [[] for j in range(64)]

for wvf_file in input_files:
    rwf = tb.open_file(wvf_file, mode='r')

    num_events_in_file = get_number_of_events(wvf_file)

    for evt in range(num_events_in_file):
        for chann in range(192, 256):
            sns          = chann - 192
            indices, wfs = indices_and_wf_above_threshold(rwf.root.RD.sipmrwf[evt][chann], charge_thr)
            if len(indices)>1 and len(wfs)>1:
                diff_ind   = np.ediff1d(indices)
                grouped_ch = [[wfs[0]]]
                for i,j in zip(diff_ind, wfs[1:]):
                    if i == 1:
                        grouped_ch[-1].append(j)
                    else:
                        grouped_ch.append([j])
                charges_all_sensors[sns].append([sum(l) for l in grouped_ch])
    rwf.close()

charges_all_sensors_new = []
for sns in charges_all_sensors:
    charges_all_sensors_new.append(np.hstack(sns))
charges_all_sensors = np.array(charges_all_sensors_new)

np.savez(out_file, charges_all_sensors=charges_all_sensors)
print(f'File {outf_name} created')
