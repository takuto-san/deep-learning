"""
FCNでより高性能なVOC2011データセットのセグメンテーションモデルを実装する
"""
import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split


class train_dataset(torch.utils.data.Dataset):
    def __init__(self, x_train, t_train):
        self.x_train = []
        self.t_train = []
        for i in range(x_train.shape[0]):
            self.x_train.append(transforms.Resize((224, 224))(Image.fromarray(np.uint8(x_train[i]))))
            self.t_train.append(transforms.Resize((224, 224))(Image.fromarray(np.uint8(t_train[i]))))
        self.transform = transforms.ToTensor()
        self.target_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.transform(self.x_train[idx]), self.target_transform(self.t_train[idx])

class test_dataset(torch.utils.data.Dataset):
    def __init__(self, x_test):
        self.x_test = []
        for i in range(x_test.shape[0]):
            self.x_test.append(transforms.Resize((224, 224))(Image.fromarray(np.uint8(x_test[i]))))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.x_test)

    def __getitem__(self, idx):
        return self.transform(self.x_test[idx])

def fix_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def TargetToTensor(target, num_classes=21):
    target = np.array(target)
    target[target > 20] = 0
    target = torch.from_numpy(target).type(torch.long)
    target = F.one_hot(target, num_classes=num_classes).permute(2,0,1).type(torch.float)
    return target

class FCN(nn.Module):
    def __init__(self, backbone, num_classes=21):
        super(FCN, self).__init__()
        self.backbone = backbone
        self.FCNhead = nn.Conv2d(2048, num_classes, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        x = self.FCNhead(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

class mIoUScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        hist = self.confusion_matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iou = np.nanmean(iou)
        return mean_iou

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

def train_model(model, dataloader_train, dataloader_valid, loss_fn, metrics, optimizer, n_epochs, device):
    model.to(device)

    for epoch in range(n_epochs):
        train_losses = []
        valid_losses = []
        metrics.reset()

        model.train()
        with tqdm(total=len(dataloader_train), unit="batch") as pbar:
            pbar.set_description(f"[train] Epoch {epoch+1}/{n_epochs}")
            for image, target in dataloader_train:
                optimizer.zero_grad()
                image, target = image.to(device), target.to(device)
                output = model(image)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                pbar.set_postfix(loss=np.array(train_losses).mean())
                pbar.update(1)

        model.eval()
        with tqdm(total=len(dataloader_valid), unit="batch") as pbar:
            pbar.set_description(f"[valid] Epoch {epoch+1}/{n_epochs}")
            for image, target in dataloader_valid:
                image, target = image.to(device), target.to(device)
                with torch.no_grad():
                    output = model(image)
                loss = loss_fn(output, target)
                valid_losses.append(loss.item())
                metrics.update(target.argmax(1).cpu().numpy(), output.argmax(1).cpu().numpy())
                pbar.set_postfix(loss=np.array(valid_losses).mean(), mIoU=metrics.get_scores())
                pbar.update(1)


if __name__ == '__main__':
    fix_seed(seed=42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'Data', 'Lecture06_dataset')

    x_train = np.load(os.path.join(data_path, 'x_train.npy'), allow_pickle=True)
    t_train = np.load(os.path.join(data_path, 't_train.npy'), allow_pickle=True)
    x_test = np.load(os.path.join(data_path, 'x_test.npy'), allow_pickle=True)

    trainval_data = train_dataset(x_train, t_train)
    test_data = test_dataset(x_test)

    val_size = 100
    train_data, val_data = torch.utils.data.random_split(trainval_data, [len(trainval_data)-val_size, val_size])

    num_classes = 21
    image_transform = transforms.ToTensor()
    target_transform = transforms.Compose([
        transforms.Lambda(lambda target: TargetToTensor(target, num_classes=num_classes))
    ])

    trainval_data.transform = image_transform
    trainval_data.target_transform = target_transform
    test_data.transform = image_transform

    batch_size = 16
    dataloader_train = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    dataloader_valid = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    dataloader_test = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    backbone = nn.Sequential(*list(backbone.children())[:-2])
    model = FCN(backbone=backbone, num_classes=num_classes)
    
    loss_fn = nn.BCEWithLogitsLoss()
    metrics = mIoUScore(num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    n_epochs = 10
    
    train_model(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_valid=dataloader_valid,
        loss_fn=loss_fn,
        metrics=metrics,
        optimizer=optimizer,
        n_epochs=n_epochs,
        device=device
    )

# 実際に動かすときはGoogleColabでやった方がいい（処理が終わらない）