import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle
from scipy.interpolate import interp1d
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter

'''
    1. Extract the knee points and add the information to the data_knee_extract.pkl data file
    2. Plot the discharge capacity curve of each battery (sample battery b1c7), the two fitted line segments, and the inflection point
    3. Plot the discharge capacity curve of all batteries (remove the outlier version)
    4. Plot the feature map (V, I, T), sample cells b2c6 and b3c21
'''

plt.rc('font', family='Times New Roman')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 8
plt.rcParams['xtick.major.pad'] = 4
plt.rcParams['ytick.major.pad'] = 4


# Define bacon-watts formula
def bacon_watts_model(x, alpha0, alpha1, alpha2, x1):
    ''' Equation of bw_model'''
    return alpha0 + alpha1 * (x - x1) + alpha2 * (x - x1) * np.tanh((x - x1) / 1e-8)


def fit_bacon_watts(y, p0):
    ''' Function to fit Bacon-Watts model to identify knee-point in capacity fade data

    Args:
    - capacity fade data (list): cycle-to-cycle evolution of Qd capacity
    - p0 (list): initial parameter values for Bacon-Watts model

    Returns:
    - popt (int): fitted parameters
    - confint (list): 95% confidence interval for fitted knee-point
    '''

    # Define array of cycles
    x = np.arange(len(y)) + 1

    # Fit bacon-watts
    popt, pcov = curve_fit(bacon_watts_model, x, y, p0=p0)

    confint = [popt[3] - 1.96 * np.diag(pcov)[3],
               popt[3] + 1.96 * np.diag(pcov)[3]]
    return popt, confint


# 导入数据
data_raw = pickle.load(open(r'./DATA/02 raw_pkl_data/batch_all_pro.pkl', 'rb'))
# Extract knee point and store data
for batch_i in data_raw:
    for key in batch_i.keys():
        QD = batch_i[key]['summary']['QD']
        cycle = batch_i[key]['summary']['cycle']
        p0 = [1, -1e-4, -1e-4, len(QD) * .7]
        popt_kpoint, confint = fit_bacon_watts(QD, p0)
        knee_point = round(popt_kpoint[3])
        batch_i[key].update({'knee_point': popt_kpoint[3]})
        knee_class = cycle.copy()
        knee_class[knee_class < popt_kpoint[3]] = 0
        knee_class[knee_class >= popt_kpoint[3]] = 1
        batch_i[key]['summary'].update({'knee_class': knee_class})

        if key == 'b1c7':
            plt.figure(figsize=(2.8, 2.8), dpi=600)
            ax = plt.subplot(111)
            plt.plot(cycle, QD, '-', color='#3585bf', linewidth=2)
            plt.plot(cycle, bacon_watts_model(cycle, *popt_kpoint), '-', color='#fd8404', linewidth=2)
            plt.scatter([knee_point], [QD[knee_point]], color='#fd8404', s=100, marker='*', zorder=3)
            plt.vlines(popt_kpoint[3], 0, 2, color='#fd8404', linewidth=2, linestyle='--')
            # plt.vlines(confint[0], 0, 2, color='red', linewidth=1, linestyle='--')
            # plt.vlines(confint[1], 0, 2, color='red', linewidth=1, linestyle='--')
            plt.xlabel('Cycles')
            plt.ylabel('Capacity/Ah')
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.set_aspect(len(cycle) / 0.23)
            plt.ylim(.875, 1.105)
            plt.xlim(0, len(cycle))
            # plt.title('{}'.format(key), fontsize=10)
            ax.set_title('(b)', loc='left', fontsize=10, x=-0.25, pad=1, fontweight='bold')
            plt.savefig('./kneepoint_pic/(b)_{}.png'.format(key), dpi=600, bbox_inches="tight")
            plt.show()
            plt.clf()
            plt.close('all')


# # with open('DATA/03 knee_extract/data_knee_extract.pkl', 'wb') as fp:
# #     pickle.dump(data_raw, fp)

# plot the discharge capacity of all batteries (outliers removed)
# find the index of the outlier
def RemoveIndex(data, sd_size=15):
    # 3sigma
    delete_index_list = []
    for i in range(0, len(data), sd_size):
        a = data[i:min(i + sd_size, len(data))].tolist()
        mean0, std0 = np.mean(a), np.std(a)
        delete_index = np.array(
            [j for j, x in enumerate(a) if x > mean0 + 3 * std0 or x < mean0 - 3 * std0]) + i  # 3sigma法则
        delete_index_list.extend(delete_index)
    # del data[delete_index_list]
    return delete_index_list

color1 = '#08478d'
color2 = "#a8cee5"
color3 = "#3585bf"  # blue
color0 = "#fd8404"  # orange
color = [color1, color2, color3]

bat_dict = pickle.load(open(r'./DATA/03 knee_extract/data_knee_extract.pkl', 'rb'))
fig1 = plt.figure(3, figsize=(2.8, 2.8), dpi=600)
ax = plt.gca()  # Get current axes
batch_name = ['Batch1', 'Batch2', 'Batch3']
for i, batch_i in enumerate(bat_dict):
    for ii, (key, data) in enumerate(batch_i.items()):
        sample_cell = key
        QD = data['summary']['QD']
        cycle = data['summary']['cycle']
        del_index = RemoveIndex(QD)
        QD = np.delete(QD, del_index)
        cycle = np.delete(cycle, del_index)
        if len(del_index) > 0:
            print('{}cell del {} cycle'.format(key, del_index))
        knee_point = round(data['knee_point'])
        if ii == 0:
            plt.plot(cycle, QD, label=batch_name[i], color=color[i], linewidth=1)
        else:
            plt.plot(cycle, QD, color=color[i], linewidth=1)
        plt.scatter([knee_point], [QD[knee_point]], color=color0, s=25, marker='*', zorder=2)
# plt.title('All Batteries', fontsize=10)
ax.set_title('(a)', loc='left', fontsize=10, x=-0.25, pad=1, fontweight='bold')
plt.xlabel('Cycles')
plt.ylabel('Capacity/Ah')
plt.ylim(.875, 1.105)
plt.xlim(0, 2300)
ax.set_aspect(2300 / 0.23)
ax.grid(True, alpha=0.2, linestyle='--')
plt.legend()
plt.savefig('./kneepoint_pic/(a)_all_cells.png', dpi=600, bbox_inches="tight")
plt.show()


# Plot the features
def plotVIT(dict_list, l=200):
    plt.rc('font', family='Times New Roman')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.size'] = 8
    plt.rcParams['xtick.major.pad'] = 4
    plt.rcParams['ytick.major.pad'] = 4

    for key, data in dict_list.items():
        if key not in ['b3c21']:
            continue
        EOL = int(data['cycle_life'])
        cycle_list = EOL
        fig1, axes1 = plt.subplots(1, 3, figsize=(6, 2), dpi=600)
        fig1.subplots_adjust(left=0.1, right=1, top=0.8, bottom=0.2, wspace=0.4)
        axes1 = axes1.flat
        cmap = plt.cm.get_cmap('Blues', cycle_list)
        norm = plt.Normalize(0, cycle_list)

        for cycle in range(EOL):
            QD = data['summary']['QD'][cycle]
            if QD > 1.3 or QD < 0.5:
                print("Battery {} cycle {} discharge capacity abnormal, discharge capacity {:.2f}".format(key, cycle, QD))
                continue

            V = data['cycles'][str(cycle)]['V']
            I = data['cycles'][str(cycle)]['I']
            t = data['cycles'][str(cycle)]['t']
            T = data['cycles'][str(cycle)]['T']
            Qc = data['cycles'][str(cycle)]['Qc']

            if t[-1] > 100:
                print("Battery {} time abnormal cycle {}, one cycle charging time is {:.2f}".format(key, cycle, t[-1]))
                continue
            if all(item == 0 for item in V):
                continue
            # extract charging process data
            end = len(t)
            for j in range(len(t)):
                if (I[j] < 0) and (I[j + 1] < 0) and (I[j + 2] < 0) and (I[j + 3] < 0) and (I[j + 4] < 0):
                    end = j
                    break

            t, V, I, T, Qc = t[:end], V[:end], I[:end], T[:end], Qc[:end]
            if t.size == 0:
                print('battery {} cycle {} has no data'.format(key, cycle))
                continue
            elif np.diff(t).size == 0:
                print('battery {} cycle {} np.diff(t).size == 0'.format(key, cycle))
                continue
            elif np.max(np.abs(np.diff(t))) > 10:
                print("battery {} cycle {} time span is large, maximum time span{:.2f}，duration{:.2f}, remove the current cycle".format(key, cycle, np.max(np.abs(np.diff(t))),
                                                                          t[-1] - t[0]))  # Delete the current cycle if the time span is large
                continue
            if any(item < -1 for item in I):
                print('The charging current curve of battery {} cycle {}  has a negative number'.format(key, cycle))
                continue

            t1, indices = np.unique(t, return_index=True)

            if len(t) != len(t1):
                print('{}cycle{} delete:'.format(key, cycle), len(t) - len(t1))
                V = V[indices]
                I = I[indices]
                T = T[indices]
                Qc = Qc[indices]

            xnew = np.linspace(t1[0], t1[-1], num=l)
            f1 = interp1d(t1, V, kind='linear')
            Vnew = f1(xnew)
            f2 = interp1d(t1, I, kind='linear')
            Inew = f2(xnew)
            f3 = interp1d(t1, T, kind='linear')
            Tnew = f3(xnew)

            axes1[0].plot(xnew, Vnew, label=cycle, color=cmap(int(cycle)), linewidth=0.5, alpha=0.5)
            axes1[1].plot(xnew, Inew, label=cycle, color=cmap(int(cycle)), linewidth=0.5, alpha=0.5)
            axes1[2].plot(xnew, Tnew, label=cycle, color=cmap(int(cycle)), linewidth=0.5, alpha=0.5)

        y_formatter = FormatStrFormatter('%1.1f')
        axes1[0].set_xlabel('time')
        axes1[0].set_ylabel('V')
        axes1[0].tick_params(length=1)
        axes1[0].yaxis.set_major_formatter(y_formatter)
        axes1[0].set_title('(d)', loc='left', fontsize=10, x=-0.25, y=0.98)
        axes1[1].set_xlabel('time')
        axes1[1].set_ylabel('I')
        axes1[1].tick_params(length=1)
        axes1[1].yaxis.set_major_formatter(y_formatter)
        axes1[1].set_title('(e)', loc='left', fontsize=10, x=-0.25, y=0.98)
        axes1[2].set_xlabel('time')
        axes1[2].set_ylabel('T')
        axes1[2].tick_params(length=1)
        axes1[2].set_title('(f)', loc='left', fontsize=10, x=-0.25, y=0.98)
        axes1[2].yaxis.set_major_formatter(y_formatter)
        fig1.suptitle(key, fontsize=10, x=0.5, y=0.95)
        # colorbar
        cb = fig1.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=[axes1[0], axes1[1], axes1[2]], pad=0.01)
        cb.set_label(label='cycles', size=8)
        cb.ax.yaxis.set_minor_locator(AutoMinorLocator())
        # Set the tick length
        cb.ax.tick_params(which='minor', length=0.8, direction='out')
        cb.ax.tick_params(which='major', length=1.5, direction='out')

        # plt.savefig('./feature_pic/{}.png'.format(key), dpi=600, bbox_inches="tight")
        plt.show()
        plt.clf()
        plt.close('all')
        break

# for batch_i in bat_dict:
#     plotVIT(batch_i)
