import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import lib
import pickle
import numpy as np
import torch

''' Perform data cleaning on features (clear abnormal cycles, linear interpolation of VIT curves)'''
# # Importing the dataset [bat_dict1, bat_dict2, bat_dict3]
# bat_dict = pickle.load(open(r'DATA/knee_extract/data_knee_extract.pkl', 'rb'))
#
# # Returns the dictionary containing the [V, I, T] values for each battery during its lifetime charging phase: knee_class, rul
# l = 200
# data_dics = []
# knee_classes = []
# rul_eols = []
# for i in range(len(bat_dict)):  # 3个batch
#     data_dic, knee_class, rul_eol, \
#     cycle_num, cycle_num_delete = lib.getData(bat_dict[i], l=l)     # data is a dictionary that stores the characteristics (voltage, current, temperature) and labels for each cell
#     print('cycle_num:', cycle_num)
#     print('cycle_delete:', cycle_num_delete)
#     data_dics.append(data_dic)
#     knee_classes.append(knee_class)
#     rul_eols.append(rul_eol)
#
# with open('./DATA/knee_extract/data_knee_rul.pkl', 'wb') as fp:
#     pickle.dump([data_dics, knee_classes, rul_eols, l], fp)

seq_len_list = [3]
divise_seed_list = [364]
interval = 1
for s1 in divise_seed_list:
    for s2 in seq_len_list:
        seq_len = s2
        divise_seed = s1

        data_dicts, knee_classes, rul_eols, l = pickle.load(
            open(r'./DATA/03 knee_extract/data_knee_rul.pkl', 'rb'))

        train_data, valid_data, test_data, \
        train_class, valid_class, test_class, \
        train_label, valid_label, test_label = {}, {}, {}, {}, {}, {}, {}, {}, {}
        for i in range(len(data_dicts)):
            train_data_temp, valid_data_temp, test_data_temp, \
            train_class_temp, valid_class_temp, test_class_temp, \
            train_label_temp, valid_label_temp, test_label_temp = lib.diviseData(
                data_dicts[i], knee_classes[i], rul_eols[i], divise_seed=divise_seed)
            train_data.update(train_data_temp)
            valid_data.update(valid_data_temp)
            test_data.update(test_data_temp)
            train_class.update(train_class_temp)
            valid_class.update(valid_class_temp)
            test_class.update(test_class_temp)
            train_label.update(train_label_temp)
            valid_label.update(valid_label_temp)
            test_label.update(test_label_temp)

        test_X, valid_X, train_X = {}, {}, {}
        test_Class, valid_Class, train_Class = {}, {}, {}
        test_Y, valid_Y, train_Y = {}, {}, {}

        # Split the sequence and store the training dataset as a dictionary
        for ii in train_data.keys():
            train_X[ii], train_Class[ii], train_Y[ii] = lib.DataToSeq(
                train_data[ii], train_class[ii], train_label[ii], seq_len=seq_len, interval=interval)

        # Split sequence and store validation in dictionary form
        for ii in valid_data.keys():
            valid_X[ii], valid_Class[ii], valid_Y[ii] = lib.DataToSeq(
                valid_data[ii], valid_class[ii], valid_label[ii], seq_len=seq_len, interval=interval)

        # Split sequence and store test dataset in dictionary form
        for ii in test_data.keys():
            test_X[ii], test_Class[ii], test_Y[ii] = lib.DataToSeq(
                test_data[ii], test_class[ii], test_label[ii], seq_len=seq_len, interval=interval)

        train_xs, train_Classes, train_ys = lib.classData(train_X, train_Class, train_Y)
        valid_xs, valid_Classes, valid_ys = lib.classData(valid_X, valid_Class, valid_Y)
        test_xs, test_Classes, test_ys = lib.classData(test_X, test_Class, test_Y)
        for m in range(3):
            # save the whole process data
            if m == 2:
                # continue
                print('knee_class_all', '*' * 100)
                # Concatenate the training data into a tensor with the battery number
                train_data_X = torch.tensor(0)
                train_data_Class = torch.tensor(0)
                train_data_Y = torch.tensor(0)
                train_min_label, train_max_label = torch.tensor(1000), torch.tensor(0)
                for index, data in train_X.items():
                    if train_data_X.numel() <= 1:
                        train_data_X = train_X[index]
                        train_data_Class = train_Class[index]
                        train_data_Y = train_Y[index]
                        train_max_label = torch.maximum(torch.max(train_Y[index][:, 0]), train_max_label)
                        train_min_label = torch.minimum(torch.max(train_Y[index][:, 0]), train_min_label)
                    else:
                        train_data_X = torch.cat((train_data_X, train_X[index]), 0)
                        train_data_Class = torch.cat((train_data_Class, train_Class[index]), 0)
                        train_data_Y = torch.cat((train_data_Y, train_Y[index]), 0)
                        train_max_label = torch.maximum(torch.max(train_Y[index][:, 0]), train_max_label)
                        train_min_label = torch.minimum(torch.max(train_Y[index][:, 0]), train_min_label)

                # Concatenate the validation set into a tensor by battery number
                valid_data_X = torch.tensor(0)
                valid_data_Class = torch.tensor(0)
                valid_data_Y = torch.tensor(0)
                valid_min_label, valid_max_label = torch.tensor(1000), torch.tensor(0)
                for index, data in valid_X.items():
                    if valid_data_X.numel() <= 1:
                        valid_data_X = valid_X[index]
                        valid_data_Class = valid_Class[index]
                        valid_data_Y = valid_Y[index]
                        valid_max_label = torch.maximum(torch.max(valid_Y[index][:, 0]), valid_max_label)
                        valid_min_label = torch.minimum(torch.max(valid_Y[index][:, 0]), valid_min_label)
                    else:
                        valid_data_X = torch.cat((valid_data_X, valid_X[index]), 0)
                        valid_data_Class = torch.cat((valid_data_Class, valid_Class[index]), 0)
                        valid_data_Y = torch.cat((valid_data_Y, valid_Y[index]), 0)
                        valid_max_label = torch.maximum(torch.max(valid_Y[index][:, 0]), valid_max_label)
                        valid_min_label = torch.minimum(torch.max(valid_Y[index][:, 0]), valid_min_label)
                # train_data_X normalization
                print('begin normalization', '-' * 10)
                train_data_norm, valid_data_norm, test_data_norm, max_data, min_data = lib.MaxMinNorm(
                    train_data_X, valid_data_X, test_X)
                print('finish normalization', '-' * 10)

                # train_max_label, _ = torch.max(train_data_Y, dim=0)
                # train_max_label = train_max_label[0]
                # train_min_label, _ = torch.min(train_data_Y, dim=0)
                # train_min_label = train_min_label[0]
                test_min_label, test_max_label = torch.tensor(1000), torch.tensor(0)
                for i2 in test_Y.keys():
                    test_max_label = torch.maximum(torch.max(test_Y[i2][:, 0]), test_max_label)
                    test_min_label = torch.minimum(torch.max(test_Y[i2][:, 0]), test_min_label)
                print('train_max_label, train_min_label:', train_max_label, train_min_label)
                print('valid_max_label, valid_min_label:', valid_max_label, valid_min_label)
                print('test_max_label, test_min_label:', test_max_label, test_min_label)

                data_path = './DATA/two_phase_data_seed{}/cycle{}/knee_class_all'.format(divise_seed, seq_len)
                if not os.path.exists(data_path):
                    os.makedirs(data_path, exist_ok=True)

                with open(data_path + r'/cell_info.txt', "w") as f:
                    f.write('train:\n{}:\n'
                            'valid:\n{}:\n'
                            'test:\n{}\n'
                            '[train_max_label, valid_max_label, test_max_label]:\n{}\n'
                            '[train_min_label, valid_min_label, test_min_label]:\n{}\n'
                            '[train_max_data]: {}\n'
                            '[train_min_data]: {}\n'
                            .format(train_data.keys(), valid_data.keys(), test_data.keys(),
                                    [train_max_label, valid_max_label, test_max_label],
                                    [train_min_label, valid_min_label, test_min_label],
                                    max_data, min_data))

                with open(data_path + '/data_train_rul.pkl', 'wb') as fp:
                    pickle.dump([train_data_norm, train_data_Class, train_data_Y], fp)

                with open(data_path + '/data_valid_rul.pkl', 'wb') as fp:
                    pickle.dump([valid_data_norm, valid_data_Class, valid_data_Y], fp)

                with open(data_path + '/data_test_rul.pkl', 'wb') as fp:
                    pickle.dump([test_data_norm, test_Class, test_Y], fp)

            # Save data in two phases
            else:
                print('knee_class{}'.format(m), '*' * 100)
                # Concatenate the training data into a tensor with the battery number
                train_data_X = torch.tensor(0)
                train_data_Class = torch.tensor(0)
                train_data_Y = torch.tensor(0)
                print('The training set starts to be concatenated', '-' * 10)
                train_min_label, train_max_label = torch.tensor(1000), torch.tensor(0)
                for index, data in train_xs[m].items():
                    if train_data_X.numel() <= 1:
                        train_data_X = train_xs[m][index]
                        train_data_Class = train_Classes[m][index]
                        train_data_Y = train_ys[m][index]
                        train_max_label = torch.maximum(torch.max(train_ys[m][index][:, 0]), train_max_label)
                        train_min_label = torch.minimum(torch.max(train_ys[m][index][:, 0]), train_min_label)
                        # print('train_min_label:', train_min_label)
                    else:
                        train_data_X = torch.cat((train_data_X, train_xs[m][index]), 0)
                        train_data_Class = torch.cat((train_data_Class, train_Classes[m][index]), 0)
                        train_data_Y = torch.cat((train_data_Y, train_ys[m][index]), 0)
                        train_max_label = torch.maximum(torch.max(train_ys[m][index][:, 0]), train_max_label)
                        train_min_label = torch.minimum(torch.max(train_ys[m][index][:, 0]), train_min_label)
                        # print('train_min_label:', train_min_label)

                # Concatenate the validation set into a tensor by battery number
                valid_data_X = torch.tensor(0)
                valid_data_Class = torch.tensor(0)
                valid_data_Y = torch.tensor(0)
                valid_min_label, valid_max_label = torch.tensor(1000), torch.tensor(0)
                for index, data in valid_xs[m].items():
                    if valid_data_X.numel() <= 1:
                        valid_data_X = valid_xs[m][index]
                        valid_data_Class = valid_Classes[m][index]
                        valid_data_Y = valid_ys[m][index]
                        valid_max_label = torch.maximum(torch.max(valid_ys[m][index][:, 0]), valid_max_label)
                        valid_min_label = torch.minimum(torch.max(valid_ys[m][index][:, 0]), valid_min_label)
                    else:
                        valid_data_X = torch.cat((valid_data_X, valid_xs[m][index]), 0)
                        valid_data_Class = torch.cat((valid_data_Class, valid_Classes[m][index]), 0)
                        valid_data_Y = torch.cat((valid_data_Y, valid_ys[m][index]), 0)
                        valid_max_label = torch.maximum(torch.max(valid_ys[m][index][:, 0]), valid_max_label)
                        valid_min_label = torch.minimum(torch.max(valid_ys[m][index][:, 0]), valid_min_label)
                print('The training set is concatenated', '-' * 10)
                print('Start normalization', '-' * 10)
                train_data_X_norm, valid_data_X_norm, test_xs_norm, max_data, min_data = lib.MaxMinNorm(
                    train_data_X, valid_data_X, test_xs[m])
                print('Finish normalization', '-' * 10)

                # train_max_label, _ = torch.max(train_data_Y, dim=0)
                # train_max_label = train_max_label[0]
                # train_min_label, _ = torch.min(train_data_Y, dim=0)
                # train_min_label = train_min_label[0]
                test_min_label, test_max_label = torch.tensor(1000), torch.tensor(0)
                for i2 in test_ys[m].keys():
                    test_max_label = torch.maximum(torch.max(test_ys[m][i2][:, 0]), test_max_label)
                    test_min_label = torch.minimum(torch.max(test_ys[m][i2][:, 0]), test_min_label)
                print('train_max_label, train_min_label:', train_max_label, train_min_label)
                print('valid_max_label, valid_min_label:', valid_max_label, valid_min_label)
                print('test_max_label, test_min_label:', test_max_label, test_min_label)

                data_path = './DATA/two_phase_data_seed{}/cycle{}/knee_class{}'.format(divise_seed, seq_len, m)
                if not os.path.exists(data_path):
                    os.makedirs(data_path, exist_ok=True)

                with open(data_path + r'/cell_info.txt', "w") as f:
                    f.write('train:\n{}:\n'
                            'valid:\n{}:\n'
                            'test:\n{}\n'
                            '[train_max_label, valid_max_label, test_max_label]:\n{}\n'
                            '[train_min_label, valid_min_label, test_min_label]:\n{}\n'
                            '[train_max_data]: {}\n'
                            '[train_min_data]: {}\n'
                            .format(train_data.keys(), valid_data.keys(), test_data.keys(),
                                    [train_max_label, valid_max_label, test_max_label],
                                    [train_min_label, valid_min_label, test_min_label],
                                    max_data, min_data))  # 自带文件关闭功能，不需要再写f.close()

                print('Start saving', '-' * 10)
                with open(data_path + '/data_train_rul.pkl', 'wb') as fp:
                    pickle.dump([train_data_X_norm, train_data_Class, train_data_Y], fp)

                with open(data_path + '/data_valid_rul.pkl', 'wb') as fp:
                    pickle.dump([valid_data_X_norm, valid_data_Class, valid_data_Y], fp)

                with open(data_path + '/data_test_rul.pkl', 'wb') as fp:
                    pickle.dump([test_xs_norm, test_Classes[m], test_ys[m]], fp)
                print('Finish saving', '-' * 10)