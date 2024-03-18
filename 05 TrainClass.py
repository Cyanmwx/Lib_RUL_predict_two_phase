import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import matplotlib.pyplot as plt
import torch
from torch import nn
import lib
import time
import visdom
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pickle
import pandas as pd
from sklearn import metrics
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

start = time.time()
batch = 256
lr = 5e-4
epochs = 20

seq_len = 3
best_net_dic = 0

# # Store model parameters
# time0=time.strftime('%m%d_%H%M',time.localtime(time.time()))
# model_path = './modelfile/class_model_{}'.format(time0)
# if not os.path.exists(model_path):
#     os.makedirs(model_path, exist_ok=True)

## GPU
device = lib.try_gpu()

train_x, train_y, _ = pickle.load(
    open('./DATA/two_phase_data_seed364/cycle3/knee_class_all/data_train_rul.pkl', 'rb'))
valid_x, valid_y, _ = pickle.load(
    open('./DATA/two_phase_data_seed364/cycle3/knee_class_all/data_valid_rul.pkl', 'rb'))


train_dataset = TensorDataset(train_x, train_y)
train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

valid_dataset = TensorDataset(valid_x, valid_y)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch, shuffle=True)


test_x, test_y, _ = pickle.load(
    open('./DATA/two_phase_data_seed364/cycle3/knee_class_all/data_test_rul.pkl', 'rb'))


net = lib.ResNet_2D(seq_len=seq_len)
net = net.to(device)

# Loss function and optimizer
class Focal_Loss(nn.Module):
    """
    Focal Loss
    """
    def __init__(self, alpha=0.38, gamma=5.5):
        super(Focal_Loss, self).__init__()
        self.alpha = torch.tensor(alpha).to(device)
        self.gamma = torch.tensor(gamma).to(device)
    def forward(self, preds, labels):
        """
        preds:sigmoid
        labels
        """
        eps = 1e-7
        loss_1 = -1 * self.alpha * torch.pow((1 - preds), self.gamma) * torch.log(preds + eps) * labels
        loss_0 = -1 * (1 - self.alpha) * torch.pow(preds, self.gamma) * torch.log(1 - preds + eps) * (1 - labels)
        loss = loss_0 + loss_1
        return torch.mean(loss)

criterion = nn.BCELoss()
# criterion = Focal_Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

# 训练过程
train_losses = []
valid_losses = []
train_correct_pers = []
valid_correct_pers = []
train_f1_scores = []
valid_f1_scores = []

for epoch in range(epochs):
    temp_train_loss = []
    train_correct = 0
    valid_correct = 0
    prob_all = []
    label_all = []

    net.train()
    for X, y in train_dataloader:  # X [batch,seq,3,200] y [batch,0/1]
        X, y = X.to(torch.float32).to(device), y.to(torch.float32).to(device)
        optimizer.zero_grad()
        output = net(X)

        # f1-score
        prob = output.cpu().detach().numpy()
        prob_all.extend(np.argmax(prob, axis=1))
        label = y.cpu().detach().numpy()
        label_all.extend(label)

        onehot_y = torch.eye(2)[y.long(), :].to(device)
        l = criterion(output, onehot_y)
        temp_train_loss.append(l.detach().cpu().numpy())
        l.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        train_correct += pred.eq(y.data.view_as(pred)).cpu().sum()  # Calculate the correct prediction number

    scheduler.step()
    train_loss = np.mean(temp_train_loss)
    # if (train_losses == []) or (train_loss < train_losses[-1]):
    #     best_net_dic = net.state_dict()
    train_losses.append(train_loss)
    train_correct_per = train_correct / float(len(train_dataloader.dataset))
    train_correct_pers.append(train_correct_per)
    train_f1_score = metrics.f1_score(label_all, prob_all, average='macro')
    train_f1_scores.append(train_f1_score)
    print('Train Epoch: {}\t'
          'F1 Score: {:.4f}\t'
          'Average Loss: {:.4f}\t'
          'accuracy: {:.2%}'.format(epoch, train_f1_score, train_loss, train_correct_per))


    # evaluate on the validation dataset
    temp_valid_loss = []
    prob_all = []
    label_all = []

    net.eval()
    with torch.no_grad():
        for X, y in valid_dataloader:
            X, y = X.to(torch.float32).to(device), y.to(torch.float32).to(device)
            output = net(X)

            # f1-score
            prob = output.cpu().detach().numpy()
            prob_all.extend(np.argmax(prob, axis=1))
            label = y.cpu().detach().numpy()
            label_all.extend(label)

            onehot_y = torch.eye(2)[y.long(), :].to(device)
            l = criterion(output, onehot_y)
            temp_valid_loss.append(l.detach().cpu().numpy())

            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            valid_correct += pred.eq(y.data.view_as(pred)).cpu().sum()
    valid_loss = np.mean(temp_valid_loss)
    if (valid_losses == []) or (valid_loss < valid_losses[-1]):
        best_net_dic = net.state_dict()
    valid_losses.append(valid_loss)
    valid_correct_per = valid_correct / float(len(valid_dataloader.dataset))
    valid_correct_pers.append(valid_correct_per)
    valid_f1_score = metrics.f1_score(label_all, prob_all, average='macro')
    valid_f1_scores.append(valid_f1_score)
    print('Test Epoch: {}'
          '\tF1 Score: {:.4f}'
          '\tAverage Loss: {:.4f}'
          '\taccuracy: {:.2%}'.format(epoch, valid_f1_score, valid_loss, valid_correct_per))

end = time.time()
print('The training and validation time is：{}'.format(end - start))

# save model parameters
time0 = time.strftime('%m%d_%H%M', time.localtime(time.time()))
model_path = './modelfile/class_model_{}'.format(time0)
pic_path = model_path + '/test_class_pic'
if not os.path.exists(model_path):
    os.makedirs(model_path, exist_ok=True)
if not os.path.exists(pic_path):
    os.makedirs(pic_path, exist_ok=True)
# torch.save(best_net_dic, model_path + './class_model_{}.pt'.format(time0))

## Plot
trainloss_df = pd.DataFrame(train_losses)
validloss_df = pd.DataFrame(valid_losses)
train_correct_per_df = pd.DataFrame(train_correct_pers)
valid_correct_per_df = pd.DataFrame(valid_correct_pers)
train_f1_score_df = pd.DataFrame(train_f1_scores)
valid_f1_score_df = pd.DataFrame(valid_f1_scores)

plt.figure(0, figsize=(8, 4))
plt.plot(trainloss_df, color="dodgerblue", linewidth=1.0, linestyle="-", label="train loss")
plt.plot(validloss_df, color="darkorange", linewidth=1.0, linestyle="-", label="valid loss")
plt.legend(loc="upper right")
plt.title('Train and Valid Loss_Epoch{}'.format(epochs))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.gca()
# plt.savefig(pic_path + r'\loss_{}.png'.format(time0))
plt.show()

plt.figure(1, figsize=(8, 4))
plt.plot(train_correct_per_df, color="dodgerblue", linewidth=1.0, linestyle="-", label="train_correct_per")
plt.plot(valid_correct_per_df, color="darkorange", linewidth=1.0, linestyle="-", label="valid_correct_per")
plt.legend(loc="upper right")
plt.title('correct_per_Epoch{}'.format(epochs))
plt.xlabel('Epoch')
plt.ylabel('per')
plt.gca()
# plt.savefig(pic_path + r'\correct_per_{}.png'.format(time0))
plt.show()

plt.figure(2, figsize=(8, 4))
plt.plot(train_f1_score_df, color="dodgerblue", linewidth=1.0, linestyle="-", label="train_f1_score")
plt.plot(valid_f1_score_df, color="darkorange", linewidth=1.0, linestyle="-", label="valid_f1_score")
plt.legend(loc="upper right")
plt.title('f1_score_Epoch{}'.format(epochs))
plt.xlabel('Epoch')
plt.ylabel('f1_score')
plt.gca()
# plt.savefig(pic_path + r'\f1_score_{}.png'.format(time0))
plt.show()

# class_model_weight_path = 'modelfile/class_model_focal/class_model.pt'
# best_net_dic = torch.load(class_model_weight_path, map_location=device)
net.load_state_dict(best_net_dic)
net.eval()

test_dataset = {}
test_dataloader = {}
for key, data in test_x.items():
    test_dataset[key] = TensorDataset(test_x[key], test_y[key])
    test_dataloader[key] = DataLoader(test_dataset[key], batch_size=batch, shuffle=False)
y1 = []
y2 = []
with torch.no_grad():
    for m, (key, data) in enumerate(test_dataloader.items()):
        for X, y in test_dataloader[key]:
            X, y = X.float().to(device), y.float().to(device)
            output = net(X)
            pred = torch.max(output, 1)[1]
            y1.extend(torch.flatten(pred).detach().cpu().tolist())
            y2.extend(torch.flatten(y).detach().cpu().tolist())

    # calculate accuracy
    y11 = torch.tensor(y1)
    y22 = torch.tensor(y2)
    test_correct = torch.sum(y11 == y22)
    test_correct_per = test_correct.cpu() / len(y11)

    # calculate the confusion matrix
    cm = confusion_matrix(y22, y11)
    lib.plot_confusion_matrix(cm, classes=['class_0', 'class_1'], normalize=True, results_path=pic_path)

    # f1-score
    f1_score_test = metrics.f1_score(y2, y1, average='binary')
    precision_score_test = metrics.precision_score(y2, y1, average='binary')
    recall_score_test = metrics.recall_score(y2, y1, average='binary')

    print('The test set accuracy is ：{:.2%}'.format(test_correct_per))
    print('The test set f1_score is：{:.4}'.format(f1_score_test))
    print('The test set precision_score is：{:.4}'.format(precision_score_test))
    print('The test set recall_score is：{:.4}'.format(recall_score_test))
