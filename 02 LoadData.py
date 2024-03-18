import numpy as np
import matplotlib.pyplot as plt
import pickle

'''
    1. Delete the non-recorded cycles and the batteries with large noise, and splice the batch1 part of batch2
    2. Generate data files: batch1_pro.pkl, batch2_pro.pkl, batch3_pro.pkl
    3. Delete the part with less than 80% capacity
    4. Plot the discharge capacity curve (find the outliers but do not delete them)
    5. Generate data file: batch_all_pro.pkl ([batch1, batch2, batch3])

    batch/batch_pro structure:
        'bxcy': dict_keys(['cycle_life', 'charge_policy', 'summary', 'cycles'])
        'summary': dict_keys(['QC', 'QD', 'cycle'])
        'cycles': '1': dict_keys(['I', 'Qc', 'Qd', 'T', 'V', 't'])
'''

'''Process batch1'''
batch1 = pickle.load(open(r'./DATA/02 raw_pkl_data/batch1.pkl', 'rb'))
# remove batteries that do not reach 80% capacity
del batch1['b1c8']
del batch1['b1c10']
del batch1['b1c12']
del batch1['b1c13']
del batch1['b1c22']

# The first cycle data is not recorded in the first batch, so the first cycle data is removed
for key, data in batch1.items():
    data['cycle_life'] = data['cycle_life'] - 1
    del data['cycles']['0']
    temp = {}
    for key2, data2 in data['cycles'].items():
        temp[str(int(key2) - 1)] = data2
    data['cycles'] = temp
    for key3, data3 in data['summary'].items():
        data3 = np.delete(data3, 0)
        data['summary'][key3] = data3
    data['summary']['cycle'] = data['summary']['cycle'] - 1



'''Process batch2'''
batch2 = pickle.load(open(r'./DATA/02 raw_pkl_data/batch2.pkl', 'rb'))
# There are four cells from batch1 that carried into batch2, we'll remove the data from batch2
# and put it with the correct cell from batch1
batch2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
batch1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
add_len = [662, 981, 1060, 208, 482]
for i, bk in enumerate(batch1_keys):
    batch1[bk]['cycle_life'] = batch1[bk]['cycle_life'] + add_len[i]
    for j in batch1[bk]['summary'].keys():
        if j == 'cycle':
            batch1[bk]['summary'][j] = np.hstack(
                (batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j] + len(batch1[bk]['summary'][j])))
        else:
            batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j]))
    last_cycle = len(batch1[bk]['cycles'].keys())
    for j, jk in enumerate(batch2[batch2_keys[i]]['cycles'].keys()):
        batch1[bk]['cycles'][str(last_cycle + j)] = batch2[batch2_keys[i]]['cycles'][jk]

del batch2['b2c7']
del batch2['b2c8']
del batch2['b2c9']
del batch2['b2c15']
del batch2['b2c16']


'''Process batch3'''
batch3 = pickle.load(open(r'./DATA/02 raw_pkl_data/batch3.pkl', 'rb'))
# remove noisy channels from batch3
del batch3['b3c37']
del batch3['b3c2']
del batch3['b3c23']
del batch3['b3c32']
del batch3['b3c42']
del batch3['b3c43']

numBat1 = len(batch1.keys())
print('Number of batteries in batch1: ', numBat1)
with open('./DATA/02 raw_pkl_data/batch1_pro.pkl', 'wb') as fp:
    pickle.dump(batch1, fp)
numBat2 = len(batch2.keys())
print('Number of batteries in batch2: ', numBat2)
with open('./DATA/02 raw_pkl_data/batch2_pro.pkl', 'wb') as fp:
    pickle.dump(batch2, fp)
numBat3 = len(batch3.keys())
print('Number of batteries in batch3: ', numBat3)
with open('./DATA/02 raw_pkl_data/batch3_pro.pkl', 'wb') as fp:
    pickle.dump(batch3, fp)


# Plot the discharge capacity curves of the three batches
for i in batch1.keys():
    plt.plot(batch1[i]['summary']['cycle'], batch1[i]['summary']['QD'], color='b')
for i in batch2.keys():
    plt.plot(batch2[i]['summary']['cycle'], batch2[i]['summary']['QD'], color='r')
for i in batch3.keys():
    plt.plot(batch3[i]['summary']['cycle'], batch3[i]['summary']['QD'], color='g')
plt.xlabel('Cycle Number')
plt.ylabel('Discharge Capacity (Ah)')
plt.ylim(0.82, 1.105)
plt.show()


"' Delete less than 80% of capacity"
batch1 = pickle.load(open(r'./DATA/02 raw_pkl_data/batch1_pro.pkl', 'rb'))
batch2 = pickle.load(open(r'./DATA/02 raw_pkl_data/batch2_pro.pkl', 'rb'))
batch3 = pickle.load(open(r'./DATA/02 raw_pkl_data/batch3_pro.pkl', 'rb'))
bat_dict = [batch1, batch2, batch3]

# Plot the discharge capacity of the original three batches
for i in bat_dict[0].keys():
    plt.plot(bat_dict[0][i]['summary']['cycle'], bat_dict[0][i]['summary']['QD'], color='b')
for i in bat_dict[1].keys():
    plt.plot(bat_dict[1][i]['summary']['cycle'], bat_dict[1][i]['summary']['QD'], color='r')
for i in bat_dict[2].keys():
    plt.plot(bat_dict[2][i]['summary']['cycle'], bat_dict[2][i]['summary']['QD'], color='g')
plt.xlabel('Cycle Number')
plt.ylabel('Discharge Capacity (Ah)')
plt.ylim(0.82, 1.105)
plt.title('Before')
# plt.savefig('./DATA/Qd_fig/Before.png')
plt.show()

cycle_index = {}
del_cycle_index = {}

for batch_i in bat_dict:  # 每个batch
    abnormal_cycle = {}
    for key in batch_i.keys():  # 每个电池
        cycle_index[key] = []
        del_cycle_index[key] = []

        QD = batch_i[key]['summary']['QD']

        # Record cycle indices with less than 80% capacity
        for j in range(len(QD)):
            percent = QD[j] / 1.1
            if percent < 0.8:
                # del_cycle_index[key].append(j)
                del_cycle_index[key].extend(list(range(j, len(QD))))
                break
            else:
                cycle_index[key].append(j)
            # Record exception cycles
            if QD[j] > 1.3 or QD[j] < 0.5:
                abnormal_cycle[key] = []
                abnormal_cycle[key].append(j)
        # Update ‘summary’
        for key2, data2 in batch_i[key]['summary'].items():
            batch_i[key]['summary'][key2] = batch_i[key]['summary'][key2][cycle_index[key]]
        # Update ‘cycle_life’
        batch_i[key]['cycle_life'] = len(batch_i[key]['summary']['cycle'])
        # Update ’cycles‘
        for key3 in list(batch_i[key]['cycles'].keys()):
            if int(key3) in del_cycle_index[key]:
                del batch_i[key]['cycles'][key3]
        if len(del_cycle_index[key]) != 0:
            print("capacity percentage of {} before deletion：{:.2%}".format(key, QD[-1] / 1.1))
            print("capacity percentage of {} after deletion：{:.2%}".format(key, batch_i[key]['summary']['QD'][-1] / 1.1))
            print('cell_{}删除：{}'.format(key, del_cycle_index[key]))

    # Plot the discharge capacity of each battery
    for i in batch_i.keys():
        plt.figure()
        plt.plot(batch_i[i]['summary']['cycle'], batch_i[i]['summary']['QD'], color='b')
        plt.title(i)
        plt.xlabel('Cycle Number')
        plt.ylabel('Discharge Capacity (Ah)')
        # plt.savefig('./DATA/Qd_fig/{}.png'.format(i))
        plt.show()

    # Plot anomalous cycle features
    for key4 in abnormal_cycle.keys():
        if len(abnormal_cycle[key4]) != 0:
            for i in abnormal_cycle[key4]:
                fig1, axes1 = plt.subplots(2, 3, figsize=(10, 6))
                axes1 = axes1.flat
                axes1[0].plot(batch_i[key4]['cycles'][str(i)]['t'],
                              batch_i[key4]['cycles'][str(i)]['V'])
                axes1[1].plot(batch_i[key4]['cycles'][str(i)]['t'],
                              batch_i[key4]['cycles'][str(i)]['I'])
                axes1[2].plot(batch_i[key4]['cycles'][str(i)]['t'],
                              batch_i[key4]['cycles'][str(i)]['T'])
                axes1[3].plot(batch_i[key4]['cycles'][str(i)]['t'],
                              batch_i[key4]['cycles'][str(i)]['Qc'])
                axes1[4].plot(batch_i[key4]['cycles'][str(i)]['t'],
                              batch_i[key4]['cycles'][str(i)]['Qd'])
                axes1[0].set_xlabel('time')
                axes1[0].set_ylabel('V')
                axes1[1].set_xlabel('time')
                axes1[1].set_ylabel('I')
                axes1[2].set_xlabel('time')
                axes1[2].set_ylabel('T')
                axes1[3].set_xlabel('time')
                axes1[3].set_ylabel('Qc')
                axes1[4].set_xlabel('time')
                axes1[4].set_ylabel('Qd')
                fig1.suptitle('cell_{}_cycle_{}'.format(key4, i))
                # plt.savefig('./DATA/abnorm_cycle/cell_{}_cycle_{}.png'.format(key4, i))
                plt.show()

# with open('./DATA/02 raw_pkl_data/batch_all_pro.pkl', 'wb') as fp:
#     pickle.dump(bat_dict, fp)

# Plot the discharge capacity curves for the three batches with SOH greater than 80%
for i in bat_dict[0].keys():
    plt.plot(bat_dict[0][i]['summary']['cycle'], bat_dict[0][i]['summary']['QD'], color='b')
for i in bat_dict[1].keys():
    plt.plot(bat_dict[1][i]['summary']['cycle'], bat_dict[1][i]['summary']['QD'], color='r')
for i in bat_dict[2].keys():
    plt.plot(bat_dict[2][i]['summary']['cycle'], bat_dict[2][i]['summary']['QD'], color='g')
plt.xlabel('Cycle Number')
plt.ylabel('Discharge Capacity (Ah)')
plt.ylim(0.87, 1.105)
plt.title('After')
# plt.savefig('./DATA/Qd_fig/After.png')
plt.show()
