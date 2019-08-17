import sys

import numpy as np
import tables as tb

from functools import partial

from invisible_cities.io.channel_param_io import subset_param_reader as spr


## run number for output file
run_number = sys.argv[1]
in_files = []
for i in range(2, len(sys.argv)):
    in_files.append(sys.argv[i])

param_names = ['gain', 'gain_sigma']

read_params = partial(spr, table_name='FIT_pmt_scaled_dark_pedestal',
                      param_names=param_names)

gainAv    = np.zeros(12)
gainAvErr = np.zeros(12)
sigmAv    = np.zeros(12)
sigmAvErr = np.zeros(12)
counter   = 0
for inF_name in in_files:
    with tb.open_file(inF_name) as inF:
        for sens, (pars, errs) in read_params(inF):
            gainAv   [sens] += pars['gain'      ]
            gainAvErr[sens] += errs['gain'      ] * errs['gain']
            sigmAv   [sens] += pars['gain_sigma']
            sigmAvErr[sens] += errs['gain_sigma'] * errs['gain_sigma']
        counter += 1

with open('PMTGainR'+run_number+'.txt', 'w') as data_out:
    for j in range(len(gainAv)):
        data_out.write(run_number+', 100000, '+str(j)+', '+str(gainAv[j]/counter)+', '+str(np.sqrt(gainAvErr[j])/counter)+', '+str(sigmAv[j]/counter)+', '+str(np.sqrt(sigmAvErr[j])/counter)+'\n')
    




