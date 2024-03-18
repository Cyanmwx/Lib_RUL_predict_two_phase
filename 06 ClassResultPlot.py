import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import matplotlib.pyplot as plt
import torch
import lib
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pickle
from sklearn import metrics
from sklearn.metrics import confusion_matrix

start = time.time()
batch = 256

plt.rc('font', family='Times New Roman')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 8
plt.rcParams['xtick.major.pad'] = 4
plt.rcParams['ytick.major.pad'] = 4
plt.rcParams.update({'font.size': 8})

## GPU
device = lib.try_gpu()

test_x, test_y, _ = pickle.load(
    open('./DATA/two_phase_data_seed364/cycle3/knee_class_all/data_test_rul.pkl', 'rb'))
seq_len = 5

data_all = pickle.load(
    open('./DATA/two_stage_data_seed364_rul5/cycle3/knee_class_all/test_data_rul.pkl', 'rb'))


net = lib.ResNet_2D(seq_len=seq_len)
net = net.to(device)

class_model_path = 'modelfile/class_model_focal'
class_model_weight_path = class_model_path + '/class_model.pt'

pic_path = './result_pic/class_result_pic'
class_net_dic = torch.load(class_model_weight_path, map_location=device)
net.load_state_dict(class_net_dic)
net.eval()

predict_result = {}
true_result = {}
test_dataset = {}
test_dataloader = {}
for key, data in test_x.items():
    test_dataset[key] = TensorDataset(test_x[key], test_y[key])
    test_dataloader[key] = DataLoader(test_dataset[key], batch_size=batch, shuffle=False)
y1 = []
y2 = []

with torch.no_grad():
    for m, (key, data) in enumerate(test_x.items()):
        X = test_x[key]
        y = test_y[key]
        X, y = X.float().to(device), y.float().to(device)
        output = net(X)
        pred = torch.max(output, 1)[1]
        y1.extend(torch.flatten(pred).detach().cpu().tolist())
        y2.extend(torch.flatten(y).detach().cpu().tolist())
        predict_result[key] = pred.tolist()
        true_result[key] = y.tolist()

    y11 = torch.tensor(y1)
    y22 = torch.tensor(y2)
    test_correct = torch.sum(y11 == y22)
    test_correct_per = test_correct.cpu() / len(y11)

    cm = confusion_matrix(y22, y11)
    lib.plot_confusion_matrix(cm, classes=['class_0', 'class_1'], normalize=True, results_path=pic_path
                              , title='Result with Focal Loss'
                              )

    f1_score_test = metrics.f1_score(y2, y1, average='binary')
    precision_score_test = metrics.precision_score(y2, y1, average='binary')
    recall_score_test = metrics.recall_score(y2, y1, average='binary')

    print('The test set accuracy is ：{:.2%}'.format(test_correct_per))
    print('The test set f1_score is：{:.4}'.format(f1_score_test))
    print('The test set precision_score is：{:.4}'.format(precision_score_test))
    print('The test set recall_score is：{:.4}'.format(recall_score_test))


# find the index of the outlier
def RemoveIndex(data, sd_size=15):
    # 3sigma
    delete_index_list = []
    for i in range(0, len(data), sd_size):
        a = data[i:min(i + sd_size, len(data))].tolist()
        mean0, std0 = np.mean(a), np.std(a)
        delete_index = np.array(
            [j for j, x in enumerate(a) if x > mean0 + 3 * std0 or x < mean0 - 3 * std0]) + i
        delete_index_list.extend(delete_index)
    return delete_index_list


colors = ['#ff0000', '#60ffa4', '#fdc156', '#009afb']
x = ['(a)', '(b)', '(c)']
t = 0
for i, (key, class_index) in enumerate(predict_result.items()):

    if key not in ['b1c32', 'b2c12',  'b3c13']:
        continue
    fig1 = plt.figure(3, figsize=(2.2, 2.2), dpi=600)
    ax1 = plt.gca()
    # Unpredicted parts
    QD = data_all[key]['QD']
    cycle = data_all[key]['cycle']
    QD_init = QD[:3]
    cycle_init = cycle[:3]

    QD_01 = QD[3:]
    cycle_01 = cycle[3:]
    knee_point = round(data_all[key]['knee_point'])
    predict0_index = [m+4 for m, e in enumerate(predict_result[key]) if e == 0]
    predict1_index = [m+4 for m, e in enumerate(predict_result[key]) if e == 1]
    true0_index = [m+4 for m, e in enumerate(true_result[key]) if e == 0]
    true1_index = [m+4 for m, e in enumerate(true_result[key]) if e == 1]
    short = true1_index[0] - knee_point
    if short > 0:
        print("true1-knee_point:", short)
        predict0_index = predict0_index[:-short]
        predict1_index = [m - short for m in predict1_index]

    short2 = true1_index[-1] - len(cycle)
    if short2 > -1:
        print("true1-end:", short2)
        predict1_index = predict1_index[:short2 - 1]

    plt.scatter(cycle_init, QD_init, color=colors[1], s=8, marker='o', zorder=2, label='Initial cycles')
    plt.scatter(cycle[predict0_index], QD[predict0_index], color=colors[3], s=8, marker='o', zorder=1, label='Slow aging phase')
    plt.scatter(cycle[predict1_index], QD[predict1_index], color=colors[2], s=8, marker='o', zorder=3, label='Rapid aging phase')
    plt.axvline(x=knee_point, color=colors[0], linestyle='--', linewidth=1.5, zorder=3, label='True knee point')

    # plt.tick_params(labelsize=8)
    plt.legend(fontsize=6)
    plt.ylim(.875, 1.105)
    plt.xlim(0, len(cycle))
    plt.xlabel('Cycles')
    plt.ylabel('Capacity/Ah')
    plt.title('{}'.format(key))
    ax1.set_title(x[t], loc='left', fontsize=10, x=-0.25, pad=1, fontweight='bold')
    t = t + 1
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.9, top=0.9)
    ax1.set_aspect(len(cycle)/0.23)
    plt.tick_params(axis='both', which='both', length=1)
    plt.grid(alpha=.2, linestyle='--')
    # plt.savefig(pic_path + r'/{}_bce.png'.format(key), dpi=600, format='png', bbox_inches="tight")
    plt.show()
    plt.clf()
    plt.close('all')
