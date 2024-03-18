import h5py
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import warnings

'''
    1. Process the raw mat data and extract the required information
    2. Generate data files batch1.pkl, batch2.pkl, batch3.pkl
'''

# matFilename = './DATA/00 raw_mat_data/2017-05-12_batchdata_updated_struct_errorcorrect.mat'
# matFilename = './DATA/00 raw_mat_data/2017-06-30_batchdata_updated_struct_errorcorrect.mat'
matFilename = './DATA/00 raw_mat_data/2018-04-12_batchdata_updated_struct_errorcorrect.mat'
f = h5py.File(matFilename)
batch = f['batch']

num_cells = batch['summary'].shape[0]
bat_dict = {}
for i in range(num_cells):
    cl = f[batch['cycle_life'][i, 0]][:]
    policy = f[batch['policy_readable'][i, 0]][:].tobytes()[::2].decode()
    summary_QC = np.hstack(f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())
    summary_QD = np.hstack(f[batch['summary'][i, 0]]['QDischarge'][0, :].tolist())
    summary_CY = np.hstack(f[batch['summary'][i, 0]]['cycle'][0, :].tolist())
    summary = {'QC': summary_QC, 'QD': summary_QD, 'cycle': summary_CY}
    cycles = f[batch['cycles'][i, 0]]
    cycle_dict = {}
    for j in range(cycles['I'].shape[0]):
        I = np.hstack((f[cycles['I'][j, 0]][:]))
        Qc = np.hstack((f[cycles['Qc'][j, 0]][:]))
        Qd = np.hstack((f[cycles['Qd'][j, 0]][:]))
        T = np.hstack((f[cycles['T'][j, 0]][:]))
        V = np.hstack((f[cycles['V'][j, 0]][:]))
        t = np.hstack((f[cycles['t'][j, 0]][:]))
        cd = {'I': I, 'Qc': Qc, 'Qd': Qd, 'T': T, 'V': V, 't': t}
        cycle_dict[str(j)] = cd

    cell_dict = {'cycle_life': cl, 'charge_policy': policy, 'summary': summary, 'cycles': cycle_dict}
    key = 'b3c' + str(i)
    bat_dict[key] = cell_dict

print(bat_dict.keys())
plt.figure()
plt.plot(bat_dict['b3c43']['summary']['cycle'], bat_dict['b3c43']['summary']['QD'])
plt.show()
plt.figure()
plt.plot(bat_dict['b3c43']['cycles']['10']['Qc'], bat_dict['b3c43']['cycles']['10']['V'])
plt.show()

with open('./DATA/02 raw_pkl_data/batch3.pkl', 'wb') as fp:
    pickle.dump(bat_dict, fp)
