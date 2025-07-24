import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Tuple
import csv

# --- 1. Dataset ---

class dataset(torch.utils.data.Dataset):
    def __init__(self, x_data):
        self.x_data = x_data.reshape(-1, 784).astype('float32') / 255

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.x_data[idx], dtype=torch.float)

def torch_log(x):
    return torch.log(torch.clamp(x, min=1e-10))

# --- 2. Model ---

class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.enc_fc1 = nn.Linear(784, 200)
        self.enc_fc2 = nn.Linear(200, 200)
        self.enc_fc_mean = nn.Linear(200, z_dim)
        self.enc_fc_logvar = nn.Linear(200, z_dim)

        self.dec_fc1 = nn.Linear(z_dim, 200)
        self.dec_fc2 = nn.Linear(200, 200)
        self.dec_fc3 = nn.Linear(200, 784)

    def encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.enc_fc1(x))
        x = F.relu(self.enc_fc2(x))
        mean = self.enc_fc_mean(x)
        log_var = self.enc_fc_logvar(x)
        return mean, log_var

    def sample_z(self, mean: torch.Tensor, log_var: torch.Tensor, device: str) -> torch.Tensor:
        epsilon = torch.randn(mean.shape).to(device)
        return mean + torch.exp(0.5 * log_var) * epsilon

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.dec_fc1(z))
        z = F.relu(self.dec_fc2(z))
        y = torch.sigmoid(self.dec_fc3(z))
        return y

    def forward(self, x: torch.Tensor, device: str) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        mean, log_var = self.encoder(x)
        z = self.sample_z(mean, log_var, device)
        y = self.decoder(z)
        return y, (mean, log_var)


# --- 3. Train and Vaild ---

def train_model(model, train_dataloader, valid_dataloader, optimizer, n_epochs, device):
    for epoch in range(n_epochs):
        losses = []
        KL_losses = []
        reconstruction_losses = []
        losses_val = []

        # --- 訓練フェーズ ---
        model.train()
        for x in train_dataloader:
            x = x.to(device)
            y, (mean, log_var) = model(x, device)

            reconstruction_loss = F.binary_cross_entropy(y, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))
            loss = reconstruction_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.cpu().detach().numpy())
            KL_losses.append(kl_loss.cpu().detach().numpy())
            reconstruction_losses.append(reconstruction_loss.cpu().detach().numpy())

        # --- 検証フェーズ ---
        model.eval()
        with torch.no_grad():
            for x in valid_dataloader:
                x = x.to(device)
                y, (mean, log_var) = model(x, device)

                reconstruction_loss = F.binary_cross_entropy(y, x, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))
                loss = reconstruction_loss + kl_loss
                
                losses_val.append(loss.cpu().detach().numpy())

        print('EPOCH:%d, Train Lower Bound:%.3f, (KL:%.3f, Recon:%.3f), Valid Lower Bound:%.3f' %
            (epoch+1, np.average(losses), np.average(KL_losses), np.average(reconstruction_losses), np.average(losses_val)))


if __name__ == '__main__':
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)

    # パスの設定
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'Data', 'Lecture13_dataset')

    # データセットの取り込み
    x_train = np.load(os.path.join(data_path, 'x_train.npy'), allow_pickle=True)
    x_test = np.load(os.path.join(data_path, 'x_test.npy'), allow_pickle=True)

    # データの前処理
    trainval_data = dataset(x_train)
    test_data = dataset(x_test)

    # ハイパーパラメータ
    z_dim = 10
    n_epochs = 15
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # モデル、損失関数、最適化アルゴリズムの定義
    val_size = 10000
    model = VAE(z_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_size = len(trainval_data) - val_size

    train_data, val_data = torch.utils.data.random_split(trainval_data, [train_size, val_size])

    # データローダ
    dataloader_train = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    dataloader_valid = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False
    )

    dataloader_test = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

    # 学習
    train_model(
        model=model,
        train_dataloader=dataloader_train,
        valid_dataloader=dataloader_valid,
        optimizer=optimizer,
        n_epochs=n_epochs,
        device=device
    )

# 動作確認済み。実際に動かすときはgoogle colabを使わないと処理が終わらない
