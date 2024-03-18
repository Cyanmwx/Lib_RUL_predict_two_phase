import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import matplotlib.pyplot as plt
import torch
import lib
import time
from torch.utils.data import DataLoader, TensorDataset
import pickle
import pandas as pd

## GPU
device = lib.try_gpu()

test_x, _, test_y = pickle.load(
    open('./DATA/two_phase_data_seed364/cycle{}/knee_class1/data_test_rul.pkl', 'rb'))
seq_len = 5

net = lib.GRA(seq_len=seq_len)
# net = lib.ResNet(seq_len=seq_len)
# net = lib.LSTM(seq_len=seq_len)
# net = lib.MLP(seq_len=seq_len)
# net = lib.no_gate_GRA(seq_len=seq_len)
# net = lib.no_ResNet_GRA(seq_len=seq_len)
# net = lib.no_MSA_GRA(seq_len=seq_len)
net = net.to(device)

rul_model_path = './modelfile/contrast_test/cycle3/cycle3_rul_model_resnet'
rul_model_weight_path = rul_model_path + '/rul_model.pt'
best_net_dic = torch.load(rul_model_weight_path, map_location=device)

result_path = rul_model_path + '/test_rul_result'
if not os.path.exists(result_path):
    os.makedirs(result_path, exist_ok=True)

test_dataset = {}
test_dataloader = {}
for key, data in test_x.items():
    test_dataset[key] = TensorDataset(test_x[key], test_y[key])
    test_dataloader[key] = DataLoader(test_dataset[key], batch_size=128, shuffle=False)

net.load_state_dict(best_net_dic)
net.eval()
with torch.no_grad():
    for m, (key, data) in enumerate(test_dataloader.items()):
        ymin = 100
        ymax = 0

        y1 = []
        y2 = []
        y3 = []
        for X, y in test_dataloader[key]:
            X, y = X.float().to(device), y.float().to(device)
            y_rul = y[:, 0]
            y_eol = y[:, 1]
            output = net(X)
            y1.extend(torch.flatten(output).detach().cpu().tolist())
            y2.extend(torch.flatten(y_rul).detach().cpu().tolist())
            y3.extend(torch.flatten(y_eol).detach().cpu().tolist())

        y11 = pd.DataFrame(y1)  # predicted value
        y22 = pd.DataFrame(y2)  # true value
        y33 = pd.DataFrame(y3)  # eol
        result = pd.DataFrame({'predict': y1, 'ground truth': y2, 'eol': y3})
        result.to_csv(result_path + "/{}_predict_result.csv".format(key), mode="w", encoding="utf-8")
        print("{}write data successfully".format(key))
