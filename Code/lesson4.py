"""
テーマ：MNISTのファッション版 (Fashion MNIST，クラス数10) を多層パーセプトロンによって分類する
"""
import torch._dynamo
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import inspect


class train_dataset(torch.utils.data.Dataset):
    def __init__(self, x_train, t_train):
        self.x_train = x_train.reshape(-1, 784).astype('float32') / 255
        self.t_train = t_train

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.x_train[idx], dtype=torch.float), torch.tensor(self.t_train[idx], dtype=torch.long)

class test_dataset(torch.utils.data.Dataset):
    def __init__(self, x_test):
        self.x_test = x_test.reshape(-1, 784).astype('float32') / 255

    def __len__(self):
        return self.x_test.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.x_test[idx], dtype=torch.float)

def relu(x):
    return torch.where(x > 0, x, torch.zeros_like(x))

def softmax(x):
    x_exp = torch.exp(x - x.max(dim=1, keepdim=True)[0])
    return x_exp / x_exp.sum(dim=1, keepdim=True)

class Dense(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        W = torch.randn(out_dim, in_dim) * (1.0 / in_dim)**0.5
        b = torch.zeros(out_dim)
        self.W = nn.Parameter(W)
        self.b = nn.Parameter(b)
    def forward(self, x):
        return x @ self.W.t() + self.b

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.l1 = Dense(in_dim, hid_dim)
        self.l2 = Dense(hid_dim, out_dim)
    def forward(self, x):
        z1 = self.l1(x)
        a1 = relu(z1)
        z2 = self.l2(a1)
        return softmax(z2)

def train_model(mlp, dataloader_train, dataloader_valid, optimizer, n_epochs, device):
    for epoch in range(n_epochs):
        losses_train = []
        losses_valid = []
        train_num = 0
        train_true_num = 0
        valid_num = 0
        valid_true_num = 0

        mlp.train()
        for x, t in dataloader_train:
            # データをデバイスに送る
            x = x.to(device)
            t = t.to(device)

            # 順伝播と損失計算
            y = mlp(x)
            prob = y[torch.arange(y.size(0)), t]
            loss = -torch.log(prob).mean()
            
            # 逆伝播とパラメータ更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 訓練データの損失と精度を記録
            pred = y.argmax(dim=1)
            losses_train.append(loss.tolist())
            acc = torch.where(pred.cpu() == t.cpu(), 1, 0)
            train_num += acc.size(0)
            train_true_num += acc.sum().item()

        mlp.eval() # モデルを評価モードに設定
        for x, t in dataloader_valid:
            # データをデバイスに送る
            x = x.to(device)
            t = t.to(device)

            # 順伝播と損失計算
            y = mlp(x)
            prob = y[torch.arange(y.size(0)), t]
            loss = -torch.log(prob).mean()
            
            # 検証データの損失と精度を記録
            pred = y.argmax(dim=1)
            losses_valid.append(loss.tolist())
            acc = torch.where(pred.cpu() == t.cpu(), 1, 0)
            valid_num += acc.size(0)
            valid_true_num += acc.sum().item()

        # エポックごとの結果を表示
        print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]'.format(
            epoch,
            np.mean(losses_train),
            train_true_num/train_num,
            np.mean(losses_valid),
            valid_true_num/valid_num
        ))


if __name__ == '__main__':
    torch._dynamo.disable()
    
    nn_except = ["Module", "Parameter", "Sequential"]
    for m in inspect.getmembers(nn):
        if not m[0] in nn_except and m[0][0:2] != "__":
            delattr(nn, m[0])

    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'Data', 'Lecture04_dataset')

    #学習データ
    x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
    t_train = np.load(os.path.join(data_dir, 't_train.npy'))

    #テストデータ
    x_test = np.load(os.path.join(data_dir, 'x_test.npy'))

    # テストデータの読み込み
    trainval_data = train_dataset(x_train, t_train)
    test_data = test_dataset(x_test)

    # データの分割
    batch_size = 32
    val_size = 10000
    train_size = len(trainval_data) - val_size

    train_data, val_data = torch.utils.data.random_split(trainval_data, [train_size, val_size])

    dataloader_train = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    dataloader_valid = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True
    )

    dataloader_test = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

    # ハイパーパラメータとモデルの初期化
    in_dim = 784
    hid_dim = 200
    out_dim = 10
    lr = 0.001
    n_epochs = 10

    mlp = MLP(in_dim, hid_dim, out_dim).to(device)
    optimizer = optim.Adam(mlp.parameters(), lr=lr)

    # モデルの学習
    train_model(mlp, dataloader_train, dataloader_valid, optimizer, n_epochs, device)

# 動作確認済み