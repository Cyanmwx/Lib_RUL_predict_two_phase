import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
import math
import itertools
import torch.nn.functional as F
from scipy.interpolate import interp1d


# torch.backends.cudnn.enabled = False

def RMSE(hat, value):
    res = torch.dot(hat - value, hat - value)
    rmse = torch.sqrt(res / len(value))
    return rmse

def MAPE(hat, value):
    temp = torch.abs((hat - value) / (value))
    temp1 = torch.sum(temp)
    temp2 = len(value)
    mape = temp1 / temp2
    return mape


def MAE(hat, value):
    temp = torch.abs(hat - value)
    mae = torch.sum(temp) / len(value)
    return mae

def AE(hat, value):
    # temp = torch.abs(hat - value)
    temp = hat - value
    return temp


def MAE_max(hat, value):
    temp = torch.abs(hat - value)
    mae = torch.max(temp)
    return mae


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


# find the index of the outlier
def RemoveIndex(data, sd_size=15):
    # 3sigma
    delete_index_list = []
    for i in range(0, len(data), sd_size):
        a = data[i:min(i + sd_size, len(data))].tolist()
        mean0, std0 = np.mean(a), np.std(a)
        delete_index = np.array(
            [j for j, x in enumerate(a) if x > mean0 + 3 * std0 or x < mean0 - 3 * std0]) + i  # 3sigma
        delete_index_list.extend(delete_index)
    # del data[delete_index_list]
    return delete_index_list



# Return a set containing [V, I, T, knee_class] for each battery's charge phase
# Take 200 sample points for each cycle V, I, T and use linear interpolation
def getData(dict_list, l=200):
    res = {}
    knee_res = {}
    rul_eol_res = {}
    eol = {}
    cycle_num = {}
    cycle_num_delete = {}
    cycle_list_delete = {}
    for key, data in dict_list.items():
        # print(key)
        res[key] = []
        knee_res[key] = []
        rul_eol_res[key] = []
        EOL = int(data['cycle_life'])
        eol[key] = EOL
        QD_list = data['summary']['QD']
        cycle_list_delete[key] = []
        cycle_list_delete[key].extend(RemoveIndex(QD_list, sd_size=50))
        print('The battery {} discharge capacity curve outlier cycle{}'.format(key, cycle_list_delete[key]))

        for cycle in range(EOL):
            # Remove capacity curve outlier cycles
            if cycle in cycle_list_delete[key]:
                continue
            QD = QD_list[cycle]
            if QD > 1.3 or QD < 0.5:
                cycle_list_delete[key].append(cycle)
                print("Battery {} cycle {} discharge capacity is abnormal, discharge capacity is {:.2f}".format(key, cycle, QD))
                continue
            # Remove RUL exception cycles
            RUL = EOL - data['summary']['cycle'][cycle] + 1
            if RUL == 0 or RUL < 0:
                cycle_list_delete[key].append(cycle)
                print("RUL <= 0", key, cycle)
                continue
            # Remove knee point exception cycles
            knee_class = data['summary']['knee_class'][cycle]
            if knee_class == 0 and RUL <= 0:
                cycle_list_delete[key].append(cycle)
                print("Before knee point RUL <= 0", key, cycle)
                continue

            V = data['cycles'][str(cycle)]['V']
            I = data['cycles'][str(cycle)]['I']
            t = data['cycles'][str(cycle)]['t']
            T = data['cycles'][str(cycle)]['T']
            Qc = data['cycles'][str(cycle)]['Qc']

            # Remove the charging time anomaly loop
            if t[-1] > 100:
                print("Battery {} time abnormal cycle {}, one cycle charging time is {:.2f}".format(key, cycle, t[-1]))
                continue
            # Remove cycles with all zero voltages
            if all(item == 0 for item in V):
                continue
            # extract charging process data (current stays negative)
            end = len(t)
            for j in range(len(t)):
                if (I[j] < 0) and (I[j + 1] < 0) and (I[j + 2] < 0) and (I[j + 3] < 0) and (I[j + 4] < 0):
                    end = j
                    break
            t, V, I, T, Qc = t[:end], V[:end], I[:end], T[:end], Qc[:end]

            # Remove the sampling anomaly loop
            if t.size == 0:
                print('battery {} cycle {} has no data'.format(key, cycle))
                continue
            elif np.diff(t).size == 0:
                print('battery {} cycle {} np.diff(t).size == 0'.format(key, cycle))
                continue
            elif np.max(np.abs(np.diff(t))) > 10:
                print(
                    "battery {} cycle {} time span is large, maximum time span{:.2f}，duration{:.2f}, remove the current cycle".format(
                        key, cycle, np.max(np.abs(np.diff(t))),
                        t[-1] - t[0]))  # Delete the current cycle if the time span is large
                continue
            if any(item < -1 for item in I):
                print('The charging current curve of battery {} cycle {}  has a negative number'.format(key, cycle))
                continue

            # Sort The Times monotonically increasing, removing the same times
            t1, indices = np.unique(t, return_index=True)
            if len(t) != len(t1):
                print('{}cycle{} remove sampling {} points'.format(key, cycle, len(t) - len(t1)))
                V = V[indices]
                I = I[indices]
                T = T[indices]
                # Qc = Qc[indices]

            # Linear interpolation, take 200 sample points, l=200
            xnew = np.linspace(t1[0], t1[-1], num=l)
            f1 = interp1d(t1, V, kind='linear')
            Vnew = f1(xnew)
            f2 = interp1d(t1, I, kind='linear')
            Inew = f2(xnew)
            f3 = interp1d(t1, T, kind='linear')
            Tnew = f3(xnew)
            # f4 = interp1d(t1, Qc, kind='linear')
            # Qcnew = f4(xnew)

            RUL_EOL = torch.stack((torch.tensor(RUL), torch.tensor(EOL)))
            cycle_temp = [Vnew, Inew, Tnew]
            knee_res[key].append(knee_class)
            rul_eol_res[key].append(RUL_EOL)
            res[key].append(cycle_temp)

    VIT_result = {}
    for key, data in res.items():
        r = []  # The cycle data of a single battery is stored
        for i, x in enumerate(data):
            tempv = torch.tensor(x[0]).unsqueeze(dim=0)  # V
            tempi = torch.tensor(x[1]).unsqueeze(dim=0)  # I
            tempt = torch.tensor(x[2]).unsqueeze(dim=0)  # T
            temp = torch.cat((tempv, tempi, tempt), 0)
            r.append(temp)
        VIT_result[key] = r
        cycle_num[key] = len(r)
        cycle_num_delete[key] = eol[key] - cycle_num[key]
        print('The battery {} has a total of {} cycles'.format(key, cycle_num[key]))
        print('The {} battery removes {} cycles'.format(key, cycle_num_delete[key]))
        print('The {} battery removes the {} cycles'.format(key, cycle_list_delete[key]))

    return VIT_result, knee_res, rul_eol_res, cycle_num, cycle_num_delete

# two phase split
def classData(data_dict, knee_class, rul):
    data_dict0 = {}
    data_dict1 = {}
    class0 = {}
    class1 = {}
    rul0 = {}
    rul1 = {}
    for i, (key, data) in enumerate(data_dict.items()):
        for index, ii in enumerate(knee_class[key]):
            if ii == 0:
                if key not in data_dict0:
                    data_dict0[key] = []
                    class0[key] = []
                    rul0[key] = []
                    data_dict0[key].append(data_dict[key][index])
                    class0[key].append(knee_class[key][index])
                    rul0[key].append(rul[key][index])
                else:
                    data_dict0[key].append(data_dict[key][index])
                    class0[key].append(knee_class[key][index])
                    rul0[key].append(rul[key][index])
            else:
                if key not in data_dict1:
                    data_dict1[key] = []
                    class1[key] = []
                    rul1[key] = []
                    data_dict1[key].append(data_dict[key][index])
                    class1[key].append(knee_class[key][index])
                    rul1[key].append(rul[key][index])
                else:
                    data_dict1[key].append(data_dict[key][index])
                    class1[key].append(knee_class[key][index])
                    rul1[key].append(rul[key][index])
        data_dict0[key] = torch.stack(data_dict0[key], dim=0)
        data_dict1[key] = torch.stack(data_dict1[key], dim=0)
        class0[key] = torch.tensor(class0[key])
        class1[key] = torch.tensor(class1[key])
        rul0[key] = torch.stack(rul0[key])
        rul1[key] = torch.stack(rul1[key])
    data_dict = [data_dict0, data_dict1]
    classes = [class0, class1]
    rul = [rul0, rul1]

    return data_dict, classes, rul


# Split train validation and test set on batteries (6:2:2)
def diviseData(data_dict, knee_class, rul, divise_seed=0):  # 划分数据集
    train_data, valid_data, test_data = {}, {}, {}
    train_class, valid_class, test_class = {}, {}, {}
    train_label, valid_label, test_label = {}, {}, {}

    train_rate = 0.6
    valid_rate = 0.2
    dic_len = len(data_dict)
    np.random.seed(divise_seed)

    shuffled_index = np.random.permutation(dic_len)
    split_index1 = int(dic_len * train_rate)
    split_index2 = int(dic_len * valid_rate)
    # train_index = shuffled_index[:split_index1]
    valid_index = shuffled_index[split_index1:split_index1 + split_index2]
    test_index = shuffled_index[split_index1 + split_index2:-1]
    for i, (key, data) in enumerate(data_dict.items()):
        if i in test_index:  # test set
            test_data[key] = data
            test_class[key] = knee_class[key]
            test_label[key] = rul[key]

        elif i in valid_index:  # validation
            valid_data[key] = data
            valid_class[key] = knee_class[key]
            valid_label[key] = rul[key]

        else:  # train
            train_data[key] = data
            train_class[key] = knee_class[key]
            train_label[key] = rul[key]
    # print('test:', test_data.keys())
    # print('valid:', valid_data.keys())
    return train_data, valid_data, test_data, train_class, valid_class, test_class, train_label, valid_label, test_label


# 序列划分
def DataToSeq(data, knee_class, rul, seq_len=5, interval=1):
    data_X = torch.tensor(0)
    data_Class = torch.tensor(0)
    data_Y = torch.tensor(0)

    data_last_index = len(data)
    if data_last_index <= 0:
        print('Insufficient battery data')

    X = []
    Knee_Class = []
    Y = []

    for i in range(seq_len, data_last_index, interval):
        temp = data[i - seq_len + 1: i + 1]
        data_x = torch.stack(temp, dim=0)
        data_class = knee_class[i]
        data_y = rul[i]
        X.append(data_x)
        Knee_Class.append(data_class)
        Y.append(data_y)

    data_x1 = torch.stack(X, dim=0)
    data_class1 = torch.tensor(Knee_Class)
    data_y1 = torch.stack(Y)

    if data_X.numel() <= 1:
        data_X = data_x1
        data_Class = data_class1
        data_Y = data_y1
    else:
        data_X = torch.cat((data_X, data_x1), 0)
        data_Class = torch.cat((data_Class, data_class1), 0)
        data_Y = torch.cat((data_Y, data_y1), 0)
    return data_X, data_Class, data_Y


# V, I, T feature normalization, training set, test set, validation set
def MaxMinNorm(train_data, valid_data, test_data_dic):
    max0, _ = torch.max(train_data, dim=0, keepdim=True)
    max0, _ = torch.max(max0, dim=1, keepdim=True)
    max0, _ = torch.max(max0, dim=3, keepdim=True)
    min0, _ = torch.min(train_data, dim=0, keepdim=True)
    min0, _ = torch.min(min0, dim=1, keepdim=True)
    min0, _ = torch.min(min0, dim=3, keepdim=True)

    train_data_norm = (train_data - min0) / (max0 - min0)
    valid_data_norm = (valid_data - min0) / (max0 - min0)

    for key3, data3 in test_data_dic.items():
        test_data_dic[key3] = (test_data_dic[key3] - min0) / (max0 - min0)
    print('[Vmax, Imax, Tmax]:', max0.squeeze())
    print('[Vmin, Imin, Tmin]:', min0.squeeze())
    return train_data_norm, valid_data_norm, test_data_dic, max0.squeeze(), min0.squeeze()



# Plot the confusion matrix
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap='Blues',
                          results_path='./modelfile'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : The value of the calculated confusion matrix
    - classes : The column corresponding to each row and each column in the confusion matrix
    - normalize : True: Displays the percentage; False: displays the count
    The ordinate is the true value and the abscissa is the predicted value
    """
    fontsize = 8
    matplotlib.rc('xtick', labelsize=fontsize)
    matplotlib.rc('ytick', labelsize=fontsize)
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    if normalize:
        matrix = cm
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.figure()

    figsize = (2.8, 2.8)
    figure, ax = plt.subplots(figsize=figsize, dpi=600)

    cb = figure.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), fraction=0.045, pad=0.06)
    cb.ax.tick_params(labelsize=8)
    cb.set_label(label='percent', size=8)
    # cb.ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(np.arange(0, 1.1, 0.2)))
    cb.ax.set_yticklabels(['{:.0f}%'.format(x) for x in np.arange(0, 101, 20)])

    cb.ax.tick_params(which='minor', length=0.8, direction='out')  # 设置小刻度线的长度
    cb.ax.tick_params(which='major', length=1.5, direction='out')  # 设置大刻度线的长度

    cb.ax.tick_params(length=1)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    font_title = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': fontsize + 2,
                  }
    # plt.title(title, fontdict=font_title, pad=10)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel('Predict class', fontsize=fontsize, labelpad=5)
    ax.set_ylabel('True class', fontsize=fontsize, labelpad=5)

    plt.tick_params(labelsize=fontsize)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    print(labels)
    [label.set_fontname('Times New Roman') for label in labels]
    if normalize:
        fm_int = 'd'
        fm_float = '.2%'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, '{:.2%}'.format(cm[i, j]),
                     horizontalalignment="center", verticalalignment='center', family="Times New Roman",
                     weight="normal", size=fontsize,
                     color="white" if cm[i, j] > thresh else "black")
            # plt.text(j, i, '{}\n({:.2%})'.format(matrix[i, j], cm[i, j]),
            #          horizontalalignment="center", verticalalignment='center', family="Times New Roman",
            #          weight="normal", size=fontsize,
            #          color="white" if cm[i, j] > thresh else "black")
            # plt.text(j, i, format(matrix[i, j], fm_int),
            #          horizontalalignment="center", verticalalignment='top', family="Times New Roman", weight="normal",
            #          size=15,
            #          color="white" if cm[i, j] > thresh else "black")
    else:
        fm_int = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fm_int),
                     horizontalalignment="center", verticalalignment='Times New Roman',
                     color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    # font_lable = {'family': 'Times New Roman',
    #               'weight': 'normal',
    #               'size': 15,
    #               }
    # plt.ylabel('True label', font_lable)
    # plt.xlabel('Predicted label', font_lable)

    ax.set_title('(b)', loc='left', fontsize=10, x=-0.25, pad=1, fontweight='bold')
    plt.savefig(results_path + r'/{}1.png'.format(title), dpi=600, format='png', bbox_inches="tight")
    plt.show()


# Classification
class Bottlrneck2d(torch.nn.Module):
    def __init__(self, In_channel, Med_channel, Out_channel, downsample=False):
        super(Bottlrneck2d, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(In_channel, Med_channel, 1, self.stride),
            torch.nn.BatchNorm2d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Med_channel, Med_channel, 3, padding=1),
            torch.nn.BatchNorm2d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Med_channel, Out_channel, 1),
            torch.nn.BatchNorm2d(Out_channel),
            torch.nn.ReLU(),
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv2d(In_channel, Out_channel, 1, self.stride)
        else:
            self.res_layer = None

    def forward(self, x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        y = self.layer(x)
        y = y + residual
        return y
class ResNet_2D(nn.Module):
    def __init__(self, seq_len=5):
        super(ResNet_2D, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(seq_len, 16, (1, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((1, 3)),
        )
        self.res_layer1 = Bottlrneck2d(16, 32, 32, False)
        self.model3 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1, 3)),
        )
        self.res_layer2 = Bottlrneck2d(32, 32, 32, False)
        self.model5 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1, 3)),
        )
        self.res_layer3 = Bottlrneck2d(32, 32, 16, False)

        self.flat = nn.AdaptiveAvgPool2d((None, 1))

        self.model7 = nn.Sequential(
            nn.Linear(288, 2),
            nn.Sigmoid(),
        )

    def forward(self, input):  # [256, 5, 3, 200]
        x = self.model1(input)  # torch.Size([256, 16, 3, 66])
        x = self.res_layer1(x)  # torch.Size([256, 32, 3, 66])

        x = self.model3(x)  # torch.Size([256, 32, 3, 31])
        x = self.res_layer2(x)  # torch.Size([256, 32, 3, 21])

        x = self.model5(x)  # torch.Size([256, 32, 3, 6])
        x = self.res_layer3(x)  # torch.Size([256, 16, 3, 6])

        x = x.view(x.size(0), -1)  # torch.Size([256, 288])
        x = self.model7(x)  # torch.Size([256, 2])
        return x


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


# @save
def transpose_output(X, num_heads):
    """Reversing the operation of the transpose qkv function"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """The softmax operation is performed by masking the element on the last axis"""
    # X:3D tensor, valid lens:1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # The masked element on the last axis is replaced with a very large negative value so that its softmax output is 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)



class DotProductAttention(nn.Module):
    """Scaling dot product attention"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, number of queries, d)
    # shape of keys: (batch_size, number of key-value pairs, d)
    # Shape of values: (batch_size, number of key-value pairs, dimension of values)
    # shape of valid_lens :(batch_size,) or (batch_size, number of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # set transpose_b=True to swap the last two dimensions of the keys
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    """MultiHeadAttention"""

    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens=None):
        # Shape of queries, keys, values:
        # (batch_size, number of queries or key-value pairs, num_hiddens)
        # shape of valid_lens :(batch_size,) or (batch_size, number of queries)
        # Shape of queries, keys, values output after transformation:
        # (batch_size*num_heads, number of queries or key-value pairs, num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) num_heads times,
            # Then copy the second item like this, and so on.
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # Shape of output :(batch_size*num_heads, number of queries, num_hiddens/num_heads)
        # output = self.attention(queries, keys, values, valid_lens)
        x = self.attention(queries, keys, values, valid_lens)
        # x = self.layer_norm(attention + queries)

        # Shape of output_concat :(batch_size, number of queries, num_hiddens)
        output_concat = transpose_output(x, self.num_heads)
        return self.W_o(output_concat)


# GRA
class GRA(nn.Module):
    """1d-cnn + gate + 2d-res + MSA"""
    def __init__(self, seq_len=13, feature_num=64, feature_size=64,
                 nhead=4, num_layers=2, num_hidden=128,
                 dropout=0.05):
        super(GRA, self).__init__()
        self.inplanes = 8
        self.model_type = 'GRU_Transformer7'

        self.model0 = nn.Sequential(
            nn.Conv1d(2, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(64, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self. model0_conv1 = nn.Conv1d(2, 64, 1, stride=3)

        self.model1 = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(64, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.model1_conv1 = nn.Conv1d(1, 64, 1, stride=3, padding=1)

        self.gate = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
        )
        # CNN
        self.model2 = nn.Sequential(
            nn.Conv2d(seq_len, 16, (3, 3),
                      ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(3),
        )
        self.model3 = nn.Sequential(
            nn.Conv2d(16, 64, (3, 3),
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.BatchNorm2d(64),
        )
        self.model3_conv1 = nn.Conv2d(seq_len, 64, (1, 1), stride=4)

        self.model4 = nn.Sequential(
            nn.Conv2d(64, feature_num, (3, 3)),
            nn.BatchNorm2d(feature_num),
        )
        self.model4_conv1 = nn.Conv2d(seq_len, feature_num, (1, 1), stride=5, padding=1)

        self.model5 = nn.Sequential(
            nn.AdaptiveMaxPool2d((None, 1)),
        )
        self.model6 = nn.Sequential(
            nn.Linear(14, feature_size),
            nn.ReLU(),
            nn.BatchNorm1d(feature_num),
        )
        self.MSA = MultiHeadAttention(feature_size, feature_size, feature_size, num_hidden, nhead, dropout)
        self.decoder = nn.Sequential(nn.Dropout(dropout),
                                     nn.Linear(num_hidden*feature_num, 1))


    def forward(self, input):
        X = input.permute(1, 0, 2, 3)
        x5 = []
        for x in X:
            x1 = self.model0(x[:, :-1, :])
            x2 = self.model1(x[:, -1:, :])
            x3 = x2 * self.gate(x2)
            x4 = x1 + x3
            x5.append(x4)
        src0 = torch.stack(x5, dim=0).permute(1, 0, 2, 3)
        src1 = self.model2(src0)
        src2 = F.relu(self.model3(src1) + self.model3_conv1(src0))
        src3 = F.relu(self.model4(src2) + self.model4_conv1(src0))

        src = self.model5(src3).squeeze(dim=3)
        src = self.model6(src)

        output = self.MSA(src, src, src)

        output = output.view(output.shape[0], -1)
        output = self.decoder(output)
        output = output.squeeze()
        return output


# Compare residual networks
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, seq_len=13, output_size=1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(seq_len, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(64, 64, 2, stride=1)
        self.layer2 = self.make_layer(64, 128, 2, stride=2)
        self.layer3 = self.make_layer(128, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256, output_size)

    def make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


# Compare MLP
class MLP(nn.Module):
    def __init__(self, seq_len=13):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(seq_len * 3 * 200, 512), nn.ReLU(),
                                    nn.Linear(512, 256), nn.ReLU(),
                                    nn.Linear(256, 128), nn.ReLU(),
                                    nn.Linear(128, 1)
                                    )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        out = self.layer1(x)
        return out


# Compare LSTM
class LSTM(nn.Module):
    def __init__(self, seq_len=13, feature_num=600, num_layers=2):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(feature_num, 512, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)  # Flatten the input
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


class no_gate_GRA(nn.Module):
    """1d-cnn + 2d-res + MSA"""
    def __init__(self, seq_len=13, feature_num=64, feature_size=64,
                 nhead=4, num_layers=2, num_hidden=128,
                 dropout=0.05):
        super(no_gate_GRA, self).__init__()
        self.inplanes = 8
        self.model_type = 'no_gate_GRA'

        # self.model0 = nn.Sequential(
        #     nn.Conv1d(2, 64, 3, padding=1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.MaxPool1d(3),
        #
        #     nn.Conv1d(64, 64, 3),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        # )
        # self. model0_conv1 = nn.Conv1d(2, 64, 1, stride=3)
        #
        # self.model1 = nn.Sequential(
        #     nn.Conv1d(1, 64, 3, padding=1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.MaxPool1d(3),
        #
        #     nn.Conv1d(64, 64, 3),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        # )
        #
        # self.model1_conv1 = nn.Conv1d(1, 64, 1, stride=3, padding=1)
        #
        # self.gate = nn.Sequential(
        #     nn.Linear(64, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.Sigmoid(),
        # )

        self.model0 = nn.Sequential(
            nn.Conv1d(3, 64, 3, padding=1),

            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 3),
            nn.BatchNorm1d(64),
        )

        # CNN
        self.model2 = nn.Sequential(
            nn.Conv2d(seq_len, 16, (3, 3),
                      ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(3),
        )
        self.model3 = nn.Sequential(
            nn.Conv2d(16, 64, (3, 3),
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.BatchNorm2d(64),
        )
        self.model3_conv1 = nn.Conv2d(seq_len, 64, (1, 1), stride=4)

        self.model4 = nn.Sequential(
            nn.Conv2d(64, feature_num, (3, 3)),
            nn.BatchNorm2d(feature_num),
        )
        self.model4_conv1 = nn.Conv2d(seq_len, feature_num, (1, 1), stride=5, padding=1)

        self.model5 = nn.Sequential(
            nn.AdaptiveMaxPool2d((None, 1)),
        )
        self.model6 = nn.Sequential(
            nn.Linear(14, feature_size),
            nn.ReLU(),
            nn.BatchNorm1d(feature_num),
        )

        self.MSA = MultiHeadAttention(feature_size, feature_size, feature_size, num_hidden, nhead, dropout)
        self.decoder2 = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(num_hidden * feature_num, 1))

    def forward(self, input):
        X = input.permute(1, 0, 2, 3)
        x4 = []
        for x in X:
            x1 = self.model0(x)
            x4.append(x1)
        src0 = torch.stack(x4, dim=0).permute(1, 0, 2, 3)
        src1 = self.model2(src0)  # [256, 16, 20, 20]
        src2 = F.relu(self.model3(src1) + self.model3_conv1(src0))
        src3 = F.relu(self.model4(src2) + self.model4_conv1(src0))

        src = self.model5(src3).squeeze(dim=3)
        src = self.model6(src)

        output = self.MSA(src, src, src)

        output = output.view(output.shape[0], -1)
        output = self.decoder2(output)
        output = output.squeeze()
        return output


class no_ResNet_GRA(nn.Module):
    """1d-cnn + gate + MSA"""

    def __init__(self, seq_len=13, feature_num=64, feature_size=64,
                 nhead=4, num_layers=2, num_hidden=128,
                 dropout=0.05):
        super(no_ResNet_GRA, self).__init__()
        self.inplanes = 8
        self.model_type = 'no_ResNet_GRA'

        self.model0 = nn.Sequential(
            nn.Conv1d(2, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(64, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.model0_conv1 = nn.Conv1d(2, 64, 1, stride=3)

        self.model1 = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(64, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.model1_conv1 = nn.Conv1d(1, 64, 1, stride=3, padding=1)

        self.gate = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
        )

        self.linear = nn.Sequential(
            nn.Linear(seq_len, 1),
            nn.ReLU()
        )

        # # CNN
        # self.model2 = nn.Sequential(
        #     nn.Conv2d(seq_len, 16, (3, 3),
        #               # padding=1,
        #               # stride=3, padding=3,
        #               # stride=(1, 3), padding=3
        #               ),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(3),
        # )
        # self.model3 = nn.Sequential(
        #     nn.Conv2d(16, 64, (3, 3),
        #               # padding=1,
        #               # stride=(1, 3), padding=3
        #               ),
        #
        #     # nn.MaxPool2d(3),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, (3, 3)),
        #     nn.BatchNorm2d(64),
        # )
        # self.model3_conv1 = nn.Conv2d(seq_len, 64, (1, 1), stride=4)
        #
        # self.model4 = nn.Sequential(
        #     nn.Conv2d(64, feature_num, (3, 3)),
        #     # nn.ReLU(),
        #     # nn.MaxPool2d(3),
        #     nn.BatchNorm2d(feature_num),
        # )
        # self.model4_conv1 = nn.Conv2d(seq_len, feature_num, (1, 1), stride=5, padding=1)
        #
        # self.model5 = nn.Sequential(
        #     # nn.Linear(16, 1),
        #     nn.AdaptiveMaxPool2d((None, 1)),
        #
        #     # nn.ReLU(),
        #     # nn.BatchNorm2d(feature_num),
        # )
        # self.model6 = nn.Sequential(
        #     # nn.AdaptiveAvgPool2d((None, feature_size)),
        #     nn.Linear(14, feature_size),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(feature_num),
        # )

        self.MSA = MultiHeadAttention(feature_size, feature_size, feature_size, num_hidden, nhead, dropout)
        self.decoder2 = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(num_hidden * feature_num, 1))

    def forward(self, input):
        X = input.permute(1, 0, 2, 3)
        x5 = []
        for x in X:
            x1 = self.model0(x[:, :-1, :])
            x2 = self.model1(x[:, -1:, :])
            x3 = x2 * self.gate(x2)
            x4 = x1 + x3
            x5.append(x4)
        src0 = torch.stack(x5, dim=0).permute(1, 2, 3, 0)
        src = self.linear(src0).squeeze(dim=3)
        # src1 = self.model2(src0)  # [256, 16, 20, 20]
        # src2 = F.relu(self.model3(src1) + self.model3_conv1(src0))
        # src3 = F.relu(self.model4(src2) + self.model4_conv1(src0))
        #
        # src = self.model5(src3).squeeze(dim=3)
        # src = self.model6(src)

        output = self.MSA(src, src, src)

        output = output.view(output.shape[0], -1)
        output = self.decoder2(output)
        output = output.squeeze()
        return output


class no_MSA_GRA(nn.Module):
    """1d-cnn + gate + MSA"""
    def __init__(self, seq_len=13, feature_num=64, feature_size=64,
                 nhead=4, num_layers=2, num_hidden=128,
                 dropout=0.05):
        super(no_MSA_GRA, self).__init__()
        self.inplanes = 8
        self.model_type = 'no_MSA_GRA'

        self.model0 = nn.Sequential(
            nn.Conv1d(2, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(64, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.model0_conv1 = nn.Conv1d(2, 64, 1, stride=3)

        self.model1 = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(64, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.model1_conv1 = nn.Conv1d(1, 64, 1, stride=3, padding=1)

        self.gate = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
        )

        self.linear = nn.Sequential(
            nn.Linear(seq_len, 1),
            nn.ReLU()
        )

        # CNN
        self.model2 = nn.Sequential(
            nn.Conv2d(seq_len, 16, (3, 3),
                      ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(3),
        )
        self.model3 = nn.Sequential(
            nn.Conv2d(16, 64, (3, 3),
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.BatchNorm2d(64),
        )
        self.model3_conv1 = nn.Conv2d(seq_len, 64, (1, 1), stride=4)

        self.model4 = nn.Sequential(
            nn.Conv2d(64, feature_num, (3, 3)),
            nn.BatchNorm2d(feature_num),
        )
        self.model4_conv1 = nn.Conv2d(seq_len, feature_num, (1, 1), stride=5, padding=1)

        self.model5 = nn.Sequential(
            nn.AdaptiveMaxPool2d((None, 1)),
        )
        self.model6 = nn.Sequential(
            nn.Linear(14, feature_size),
            nn.ReLU(),
            nn.BatchNorm1d(feature_num),
        )

        self.linear1 = nn.Sequential(
            nn.Linear(64, 1),
            nn.ReLU()
        )

        self.linear2 = nn.Sequential(
            nn.Linear(64, 1),
            nn.ReLU()
        )

        self.MSA = MultiHeadAttention(feature_size, feature_size, feature_size, num_hidden, nhead, dropout)
        self.decoder = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(num_hidden * feature_num, 1))

    def forward(self, input):
        X = input.permute(1, 0, 2, 3)
        x5 = []
        for x in X:
            x1 = self.model0(x[:, :-1, :])
            x2 = self.model1(x[:, -1:, :])
            x3 = x2 * self.gate(x2)
            x4 = x1 + x3
            x5.append(x4)
        src0 = torch.stack(x5, dim=0).permute(1, 0, 2, 3)
        src1 = self.model2(src0)
        src2 = F.relu(self.model3(src1) + self.model3_conv1(src0))
        src3 = F.relu(self.model4(src2) + self.model4_conv1(src0))

        src = self.model5(src3).squeeze(dim=3)
        src = self.model6(src)

        output = self.linear1(src).squeeze()
        output = self.linear2(output).squeeze()

        # output = self.MSA(src, src, src)
        # output = output.view(output.shape[0], -1)
        # output = self.decoder(output)
        # output = output.squeeze()
        return output



