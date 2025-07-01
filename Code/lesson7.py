import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
import string
import re
from typing import List, Union


def text_transform(text: List[int], max_length=256):
    # <BOS>は1, <EOS>は2とする
    text = text[:max_length - 1] + [2]
    return text, len(text)

def collate_batch(batch):
    label_list, text_list, len_seq_list = [], [], []

    for sample in batch:
        if isinstance(sample, tuple):
            label, text = sample
            label_list.append(label)
        else:
            text = sample.copy()

        text, len_seq = text_transform(text)
        text_list.append(torch.tensor(text))
        len_seq_list.append(len_seq)

    # <PAD>は3とする
    if not label_list: 
        return None, pad_sequence(text_list, padding_value=3).T, torch.tensor(len_seq_list)

    return torch.tensor(label_list), pad_sequence(text_list, padding_value=3).T, torch.tensor(len_seq_list)

class Embedding(nn.Module):
    def __init__(self, vocab_size, emb_dim, padding_idx=3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)

    def forward(self, x):
        return self.emb(x)

class SequenceTaggingNet(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        outputs, _ = self.rnn(emb)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, self.rnn.hidden_size)
        h_final = outputs.gather(1, idx).squeeze(1)
        return self.fc(h_final)


def train_model(net, train_dataloader, valid_dataloader, criterion, optimizer, n_epochs, device):
    for epoch in range(n_epochs):
        losses_train, losses_valid, all_t_valid, all_y_pred = [], [], [], []

        # 訓練フェーズ
        net.train()
        for label, line, len_seq in train_dataloader:
            x = line.to(device)
            t = label.to(device).float()
            len_seq_idx = len_seq.to(device)

            optimizer.zero_grad()
            logits = net(x, len_seq_idx).squeeze(1)
            loss = criterion(logits, t)
            loss.backward()
            optimizer.step()
            losses_train.append(loss.item())

        # 検証フェーズ
        net.eval()
        with torch.no_grad():
            for label, line, len_seq in valid_dataloader:
                x = line.to(device)
                t = label.to(device).float()
                len_seq_idx = len_seq.to(device)

                logits = net(x, len_seq_idx).squeeze(1)
                loss = criterion(logits, t)
                losses_valid.append(loss.item())

                preds = torch.sigmoid(logits).round()
                all_t_valid.extend(t.cpu().tolist())
                all_y_pred.extend(preds.cpu().tolist())

        f1 = f1_score(all_t_valid, all_y_pred, average='macro', zero_division=0)
        print('EPOCH: {}, Train Loss: {:.3f}, Valid Loss: {:.3f}, Validation F1: {:.3f}'.format(
            epoch + 1,
            np.mean(losses_train),
            np.mean(losses_valid),
            f1
        ))


if __name__ == '__main__':
    # シードの固定
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # データパスの構築
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'Data', 'Lecture07_dataset')

    x_train_all = np.load(os.path.join(data_path, 'x_train.npy'), allow_pickle=True)
    t_train_all = np.load(os.path.join(data_path, 't_train.npy'), allow_pickle=True)
    x_test = np.load(os.path.join(data_path, 'x_test.npy'), allow_pickle=True)



    # 学習データと検証データに分割
    x_train, x_valid, t_train, t_valid = train_test_split(x_train_all, t_train_all, test_size=0.2, random_state=seed)

    # 語彙数の計算
    word_num = np.concatenate(np.concatenate((x_train, x_test))).max() + 1
    print(f"単語種数: {word_num}")


    # ハイパーパラメータの設定
    batch_size = 128
    emb_dim = 100
    hid_dim = 50
    n_epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用デバイス: {device}")


    train_dataloader = DataLoader(
        list(zip(t_train, x_train)),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    valid_dataloader = DataLoader(
        list(zip(t_valid, x_valid)),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_dataloader = DataLoader(
        list(x_test),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )


    # 学習ループ
    net = SequenceTaggingNet(word_num, emb_dim, hid_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters())

    # 学習の実行
    train_model(
        net=net,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        n_epochs=n_epochs,
        device=device
    )

# 実際に動かすときはGoogleColabでやった方がいい（処理が終わらない）