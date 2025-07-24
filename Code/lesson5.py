"""
テーマ：CNNでより高精度なCIFAR10の分類器を実装する
"""
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm_notebook as tqdm
from PIL import Image
from sklearn.model_selection import train_test_split


class train_dataset(torch.utils.data.Dataset):
    def __init__(self, x_train, t_train):
        data = x_train.astype('float32')
        self.x_train = []
        for i in range(data.shape[0]):
            self.x_train.append(Image.fromarray(np.uint8(data[i])))
        self.t_train = t_train
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.transform(self.x_train[idx]), torch.tensor(self.t_train[idx], dtype=torch.long)

class test_dataset(torch.utils.data.Dataset):
    def __init__(self, x_test):
        data = x_test.astype('float32')
        self.x_test = []
        for i in range(data.shape[0]):
            self.x_test.append(Image.fromarray(np.uint8(data[i])))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.x_test)

    def __getitem__(self, idx):
        return self.transform(self.x_test[idx])

def fix_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class gcn():
    def __init__(self):
        pass

    def __call__(self, x):
        mean = torch.mean(x)
        std = torch.std(x)
        return (x - mean)/(std + 10**(-6))

class ZCAWhitening():
    def __init__(self, epsilon=1e-4, device="cuda"):
        self.epsilon = epsilon
        self.device = device

    def fit(self, images):
        x = images[0][0].reshape(1, -1)
        self.mean = torch.zeros([1, x.size()[1]]).to(self.device)
        con_matrix = torch.zeros([x.size()[1], x.size()[1]]).to(self.device)
        for i in range(len(images)):
            x = images[i][0].reshape(1, -1).to(self.device)
            self.mean += x / len(images)
            con_matrix += torch.mm(x.t(), x) / len(images)
            if i % 10000 == 0:
                print("{0}/{1}".format(i, len(images)))
        self.E, self.V = torch.linalg.eigh(con_matrix)
        self.E = torch.max(self.E, torch.zeros_like(self.E))
        self.ZCA_matrix = torch.mm(torch.mm(self.V, torch.diag((self.E.squeeze()+self.epsilon)**(-0.5))), self.V.t())
        print("completed!")

    def __call__(self, x):
        size = x.size()
        x = x.reshape(1, -1).to(self.device)
        x -= self.mean
        x = torch.mm(x, self.ZCA_matrix.t())
        x = x.reshape(tuple(size))
        x = x.to("cpu")
        return x

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.0)

def train_model(model, dataloader_train, dataloader_valid, optimizer, loss_function, n_epochs, device):
    """
    モデルの学習を実行する関数
    """
    model.to(device)
    
    for epoch in range(n_epochs):
        losses_train = []
        losses_valid = []

        model.train()
        n_train = 0
        acc_train = 0
        for x, t in dataloader_train:
            x, t = x.to(device), t.to(device)
            y = model(x)
            loss = loss_function(y, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = y.argmax(dim=1)
            n_train += t.size(0)
            acc_train += (pred == t).float().sum().item()
            losses_train.append(loss.tolist())

        model.eval()
        n_val = 0
        acc_val = 0
        with torch.no_grad():
           for x, t in dataloader_valid:
            x, t = x.to(device), t.to(device)
            y = model(x)
            loss = loss_function(y, t)
            pred = y.argmax(dim=1)
            n_val += t.size(0)
            acc_val += (pred == t).float().sum().item()
            losses_valid.append(loss.tolist())

        print(
            f'EPOCH: {epoch}, '
            f'Train [Loss: {np.mean(losses_train):.3f}, Accuracy: {acc_train/n_train:.3f}], '
            f'Valid [Loss: {np.mean(losses_valid):.3f}, Accuracy: {acc_val/n_val:.3f}]'
        )


if __name__ == '__main__':
    fix_seed(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # パス設定
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'Data', 'Lecture05_dataset')

    # 学習データの読み込み
    x_train = np.load(os.path.join(data_path, 'x_train.npy'))
    t_train = np.load(os.path.join(data_path, 't_train.npy'))

    # テストデータの読み込み
    x_test = np.load(os.path.join(data_path, 'x_test.npy'))

    trainval_data = train_dataset(x_train, t_train)
    test_data = test_dataset(x_test)

    # データの前処理
    zca = ZCAWhitening(device=device)
    zca.fit(trainval_data)

    val_size = 3000
    train_data, val_data = torch.utils.data.random_split(trainval_data, [len(trainval_data)-val_size, val_size])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: zca(x)),
    ])
    transform_valid_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: zca(x)),
    ])

    train_data.dataset.transform = transform_train
    val_data.dataset.transform = transform_valid_test
    test_data.transform = transform_valid_test

    batch_size = 64
    dataloader_train = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    dataloader_valid = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    dataloader_test = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    conv_net = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout(0.2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout(0.3),
        nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256), nn.Dropout(0.5),
        nn.Linear(256, 10)
    )

    conv_net.apply(init_weights)

    n_epochs = 5
    lr = 0.01
    optimizer = optim.SGD(conv_net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loss_function = nn.CrossEntropyLoss()

    # 学習関数の呼び出し
    train_model(
        model=conv_net,
        dataloader_train=dataloader_train,
        dataloader_valid=dataloader_valid,
        optimizer=optimizer,
        loss_function=loss_function,
        n_epochs=n_epochs,
        device=device
    )

# 動作確認済み。実際に動かすときはgoogle colabを使わないと処理が終わらない