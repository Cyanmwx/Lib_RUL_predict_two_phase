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
import pandas as pd
import d2l



start = time.time()
batch = 128
lr = 5e-4
weight_decay = 5e-4
epochs = 50
## GPU
device = lib.try_gpu()
delta = 5

## Training process
def train(net, train_dataloader, valid_dataloader,
          num_epochs, lr, weight_decay, device, delta=5):
    train_losses = []
    valid_losses = []
    train_MAEs = []
    train_RMSEs = []
    valid_MAEs = []
    valid_RMSEs = []
    best_net_dic = 0

    net.train()
    # Adam
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs*0.1, gamma=0.7)
    criterion = torch.nn.HuberLoss(reduction='mean', delta=delta)
    for epoch in range(num_epochs):
        temp_train_loss = []
        temp_train_MAE = []
        temp_train_RMSE = []
        net.train()
        for X, y in train_dataloader:
            X, y = X.float().to(device), y.float().to(device)
            y1 = y[:, 0]
            optimizer.zero_grad()
            output = net(X)
            # loss
            if output.shape == y1.shape:
                l = criterion(output, y1)
                mae = lib.MAE(output, y1)
                rmse = lib.RMSE(output, y1)
            else:
                output = torch.squeeze(output).reshape(y1.shape)
                l = criterion(output, y1)
                mae = lib.MAE(output, y1)
                rmse = lib.RMSE(output, y1)
            temp_train_loss.append(l.detach().cpu().numpy())
            temp_train_MAE.append(mae.detach().cpu().numpy())
            temp_train_RMSE.append(rmse.detach().cpu().numpy())
            l.backward()
            optimizer.step()

        scheduler.step()
        train_loss = np.mean(temp_train_loss)
        train_MAE = np.mean(temp_train_MAE)
        train_RMSE = np.mean(temp_train_RMSE)


        train_losses.append(train_loss)
        train_MAEs.append(train_MAE)
        train_RMSEs.append(train_RMSE)

        print('Train Epoch: {}\t'
              'Average Loss: {:.4f}\t'
              'MAE: {:.2f}\t'
              'RMSE: {:.4f}'.format(epoch, train_loss, train_MAE, train_RMSE))
        if valid_dataloader is not None:
            temp_valid_loss = []
            temp_valid_MAE = []
            temp_valid_RMSE = []
            net.eval()
            with torch.no_grad():
                for X, y in valid_dataloader:
                    X, y = X.float().to(device), y.float().to(device)
                    y1 = y[:, 0]
                    output = net(X)

                    # loss
                    if output.shape == y.shape:
                        mae = lib.MAE(output, y1)
                        rmse = lib.RMSE(output, y1)
                        l = criterion(output, y1)
                    else:
                        output = torch.squeeze(output).reshape(y1.shape)
                        l = criterion(output, y1)
                        mae = lib.MAE(output, y1)
                    temp_valid_loss.append(l.detach().cpu().numpy())
                    temp_valid_MAE.append(mae.detach().cpu().numpy())
                    temp_valid_RMSE.append(rmse.detach().cpu().numpy())
            valid_loss = np.mean(temp_valid_loss)
            valid_MAE = np.mean(temp_valid_MAE)
            valid_RMSE = np.mean(temp_valid_RMSE)

            if (valid_losses == []) or (valid_loss < valid_losses[-1]):
                best_net_dic = net.state_dict()
                # torch.save(net.state_dict(), './modelfile/model_{}.pt'.format(time0))

            valid_losses.append(valid_loss)
            valid_MAEs.append(valid_MAE)
            valid_RMSEs.append(valid_RMSE)
            print('Test Epoch: {}\t'
                  'Average Loss: {:.4f}\t'
                  'MAE: {:.2f}\t'
                  'RMSE: {:.4}'.format(epoch, valid_loss, valid_MAE, valid_RMSE))
    return train_losses, train_MAEs, train_RMSEs, valid_losses, valid_MAEs, valid_RMSEs, best_net_dic


for s1 in [5]:

    model_path0 = './modelfile/contrast_test/cycle{}'.format(s1)
    MAE_10 = []
    RMSE_10 = []

    data_path = './DATA/two_phase_data_seed364/cycle{}/knee_class1'.format(s1)
    for s3 in range(5):
        train_x, _, train_y = pickle.load(open(data_path + '/data_train_rul.pkl', 'rb'))
        valid_x, _, valid_y = pickle.load(open(data_path + '/data_valid_rul.pkl', 'rb'))
        test_x, _, test_y = pickle.load(open(data_path + '/data_test_rul.pkl', 'rb'))
        seq_len = s1

        train_dataset = TensorDataset(train_x, train_y)
        train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

        valid_dataset = TensorDataset(valid_x, valid_y)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch, shuffle=True)

        # 模型
        net = lib.GRA(seq_len=seq_len)

        # # Comparison of models
        # net = lib.ResNet(seq_len=seq_len)
        # net = lib.LSTM(seq_len=seq_len)
        # net = lib.MLP(seq_len=seq_len)
        # # Ablation experiment
        # net = lib.no_gate_GRA(seq_len=seq_len)
        # net = lib.no_ResNet_GRA(seq_len=seq_len)
        # net = lib.no_MSA_GRA(seq_len=seq_len)
        net = net.to(device)


        train_l_sum, valid_l_sum = 0, 0
        train_l = []
        best_net_dic1 = 0
        train_losses, train_MAEs, train_RMSEs, \
        valid_losses, valid_MAEs, valid_RMSEs, \
        best_net_dic = train(net, train_dataloader, valid_dataloader, num_epochs=epochs, lr=lr,
                             weight_decay=weight_decay, device=device)

        time0 = time.strftime('%m%d_%H%M', time.localtime(time.time()))

        model_path = model_path0 + '/cycle{}_rul_model_{}_num{}'.format(s1, time0, s3)
        pic_path = model_path + '/test_rul_pic'
        if not os.path.exists(model_path0):
            os.makedirs(model_path0, exist_ok=True)
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
        if not os.path.exists(pic_path):
            os.makedirs(pic_path, exist_ok=True)
        torch.save(best_net_dic, model_path + './rul_model_{}.pt'.format(time0))

        train_l_sum += train_losses[-1]
        train_l.append(train_losses[-1])
        valid_l_sum += valid_losses[-1]
        d2l.plot(list(range(1, epochs + 1)), [train_losses, valid_losses],
                 xlabel='epoch', ylabel='mse', xlim=[1, epochs],
                 legend=['train', 'valid']
                 )
        plt.show()
        d2l.plot(list(range(1, epochs + 1)), [train_MAEs, valid_MAEs],
                 xlabel='epoch', ylabel='MAE', xlim=[1, epochs],
                 legend=['train', 'valid']
                 )
        plt.show()
        print(f'train loss：{float(train_losses[-1]):4f}, '
              f'validation loss：{float(valid_losses[-1]):4f}')

        end = time.time()
        print('Program run time ：{}'.format(end - start))

        MAEs = {}
        RMSEs = {}

        MAEs_all = []
        RMSEs_all = []

        test_dataset = {}
        test_dataloader = {}
        for key, data in test_x.items():
            test_dataset[key] = TensorDataset(test_x[key], test_y[key])
            test_dataloader[key] = DataLoader(test_dataset[key], batch_size=batch, shuffle=False)

        net.load_state_dict(best_net_dic)
        net.eval()
        with torch.no_grad():
            for m, (key, data) in enumerate(test_dataloader.items()):
                y1 = []
                y2 = []
                MAEs[key] = []
                RMSEs[key] = []
                for X, y in test_dataloader[key]:
                    X, y = X.float().to(device), y.float().to(device)
                    y_rul = y[:, 0]
                    y_eol = y[:, 1]
                    output = net(X)
                    # loss
                    if output.shape == y_rul.shape:
                        # calculate metric
                        MAE = lib.MAE(output.detach().cpu(), y_rul.detach().cpu())
                        RMSE = lib.RMSE(output.detach().cpu(), y_rul.detach().cpu())
                    else:
                        output = torch.squeeze(output).reshape(y_rul.shape)
                        MAE = lib.MAE(output.detach().cpu(), y_rul.detach().cpu())
                        RMSE = lib.RMSE(output.detach().cpu(), y_rul.detach().cpu())
                    y1.extend(torch.flatten(output).detach().cpu().tolist())
                    y2.extend(torch.flatten(y_rul).detach().cpu().tolist())
                    MAEs[key].append(MAE)
                    RMSEs[key].append(RMSE)
                y11 = pd.DataFrame(y1)
                y22 = pd.DataFrame(y2)
                fig = plt.figure(2, figsize=(8, 8))
                ax = fig.add_subplot(111)
                plt.scatter(y22, y11, s=5, c='red', label='predict')
                yy = range(round(min([min(y2), min(y1)])), round(max([max(y2), max(y1)])))
                plt.plot(yy, yy, color="dodgerblue", linewidth=3, linestyle="-", label="real")
                ax.set_aspect('equal', adjustable='box')
                plt.legend(loc="upper right")
                plt.title('{}result'.format(key))
                plt.xlabel('n')
                plt.ylabel('rul')
                plt.gca()
                plt.savefig(pic_path + '/rul_{}.png'.format(key))
                # plt.show()
                plt.clf()
                plt.close('all')

                print('test dataset{}_MAE is：{:.4}'.format(key, np.mean(MAEs[key])))
                print('test dataset{}_RMSE is：{:.4}'.format(key, np.mean(RMSEs[key])))

                with open(model_path + r'/test_result.txt', "a") as f:
                    f.write('test dataset{}:\n'
                            'MAE：{:.4}\t'
                            'RMSE：{:.4}\n\n'
                            .format(key, np.mean(MAEs[key]), np.mean(RMSEs[key]))
                            )
                MAEs_all.append(np.mean(MAEs[key]))
                RMSEs_all.append(np.mean(RMSEs[key]))
            print("test dataset max_MAE：{:.4}, max_RMSE：{:.4}"
                  .format(np.max(MAEs_all), np.max(RMSEs_all)))
            print("test dataset min_MAE：{:.4}, min_RMSE：{:.4}"
                  .format(np.min(MAEs_all), np.min(RMSEs_all)))
            print("test dataset mean_MAE：{:.4}, mean_RMSE：{:.4}"
                  .format(np.mean(MAEs_all), np.mean(RMSEs_all)))
            print("train_x length：", len(train_x))

            with open(model_path + r'/test_result.txt', "a") as f:
                f.write('test dataset max_MAE：{:.4}, max_RMSE：{:.4}\n'
                        'test dataset min_MAE：{:.4}, min_RMSE：{:.4}\n'
                        'test dataset mean_MAE：{:.4}, mean_RMSE：{:.4}\n'
                        'train_x length：{}'
                        .format(np.max(MAEs_all), np.max(RMSEs_all),
                                np.min(MAEs_all), np.min(RMSEs_all),
                                np.mean(MAEs_all), np.mean(RMSEs_all),
                                len(train_x)
                                )
                        )
        MAE_10.append(np.mean(MAEs_all))
        RMSE_10.append(np.mean(RMSEs_all))
    with open(model_path0 + r'/contrast_test_result.txt', "a") as f:
        f.write('cycle{}:\n'
                'MAE:\t{}\n'
                'RMSE:\t{}\n'
                'mean_MAE:\t{:.4}\n'
                'mean_RMSE:\t{:.4}\n'.format(s1, MAE_10, RMSE_10, np.mean(MAE_10), np.mean(RMSE_10)))
