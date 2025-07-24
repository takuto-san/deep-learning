"""
テーマ：
NYUv2 セマンティックセグメンテーション
RGB画像から、画像内の各ピクセルがどのクラスに属するかを予測するセマンティックセグメンテーションタスク
"""
import os
import time
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import VisionDataset
from torchvision import transforms, models
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    Lambda,
    InterpolationMode
)
from torch.cuda.amp import autocast, GradScaler
from dataclasses import dataclass
import random

# PyTorchの環境設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def set_seed(seed):
    """乱数シードを固定する"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --- 1. DataLoader and IoU ---

def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    cmap = cmap/255 if normalized else cmap
    return cmap

class NYUv2(VisionDataset):
    cmap = colormap()
    def __init__(self, root, split='train', include_depth=False, transform=None, target_transform=None):
        super(NYUv2, self).__init__(root, transform=transform, target_transform=target_transform)
        assert(split in ('train', 'test'))
        self.root = root
        self.split = split
        self.include_depth = include_depth
        img_names = os.listdir(os.path.join(self.root, self.split, 'image'))
        img_names.sort()
        images_dir = os.path.join(self.root, self.split, 'image')
        self.images = [os.path.join(images_dir, name) for name in img_names]
        if self.split == 'train':
            label_dir = os.path.join(self.root, self.split, 'label')
            self.labels = [os.path.join(label_dir, name) for name in img_names]
            self.targets = self.labels
        depth_dir = os.path.join(self.root, self.split, 'depth')
        self.depths = [os.path.join(depth_dir, name) for name in img_names]

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        depth = Image.open(self.depths[idx]).convert('L')
        if self.transform is not None:
            image = self.transform(image)
            depth = self.transform(depth)
        if self.split == 'test':
            return (image, depth) if self.include_depth else image
        target = Image.open(self.targets[idx])
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (image, depth, target) if self.include_depth else (image, target)

    def __len__(self):
        return len(self.images)

# Mean IoUを計算する
def mean_iou(pred, label, num_classes):
    iou_list = []
    for c in range(num_classes):
        pred_c = (pred == c) & (label != 255)
        label_c = (label == c) & (label != 255)
        intersection = torch.logical_and(pred_c, label_c).sum()
        union = torch.logical_or(pred_c, label_c).sum()
        iou = (intersection + 1e-6) / (union + 1e-6)
        iou_list.append(iou.item())
    return np.mean(iou_list)

# --- 2. Model ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class PretrainedUNet(nn.Module):
    def __init__(self, num_classes, in_channels=4):
        super().__init__()
        base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        base_weights = base_model.conv1.weight.clone()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight[:, :3, :, :] = base_weights
            self.conv1.weight[:, 3, :, :] = base_weights.mean(dim=1)
        self.encoder1 = nn.Sequential(self.conv1, base_model.bn1, base_model.relu)
        self.pool = base_model.maxpool
        self.encoder2 = base_model.layer1
        self.encoder3 = base_model.layer2
        self.encoder4 = base_model.layer3
        self.encoder5 = base_model.layer4
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = DoubleConv(512 + 256, 256)
        self.dec3 = DoubleConv(256 + 128, 128)
        self.dec2 = DoubleConv(128 + 64, 64)
        self.dec1 = DoubleConv(64 + 64, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        up5 = F.interpolate(e5, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([up5, e4], dim=1))

        up4 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([up4, e3], dim=1))

        up3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([up3, e2], dim=1))

        up2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([up2, e1], dim=1))

        out = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.final_conv(out)

# --- 3. Train and Vaild ---

@dataclass
# 学習用データの設定
class TrainingConfig:
    dataset_root: str = ""    
    batch_size: int = 16
    num_workers: int = 0 # default: 4
    in_channels: int = 4
    num_classes: int = 13
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    train_val_split_ratio: float = 0.8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# データセットの準備
def setup_dataloaders(config: TrainingConfig):
    transform = Compose([Resize((240, 320), interpolation=InterpolationMode.BILINEAR), ToTensor()])
    target_transform = Compose([Resize((240, 320), interpolation=InterpolationMode.NEAREST), Lambda(lambda lbl: torch.from_numpy(np.array(lbl)).long())])

    full_train_dataset = NYUv2(root=config.dataset_root, split='train', include_depth=True, transform=transform, target_transform=target_transform)
    test_dataset = NYUv2(root=config.dataset_root, split='test', include_depth=True, transform=transform)

    train_size = int(config.train_val_split_ratio * len(full_train_dataset))
    valid_size = len(full_train_dataset) - train_size
    train_subset, valid_subset = random_split(full_train_dataset, [train_size, valid_size])

    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    valid_loader = DataLoader(valid_subset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)

    print(f"Train data: {len(train_subset)} samples, Valid data: {len(valid_subset)} samples")
    return train_loader, valid_loader, test_loader

# モデルの学習と検証
def train_model(net, train_dataloader, valid_dataloader, criterion, optimizer, n_epochs, device, num_classes):
    scaler = GradScaler()
    for epoch in range(n_epochs):
        # --- 訓練フェーズ ---
        net.train()
        train_loss = 0.0
        for image, depth, label in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]"):
            image, depth, label = image.to(device), depth.to(device), label.to(device)
            optimizer.zero_grad()
            with autocast():
                x = torch.cat((image, depth), dim=1)
                pred = net(x)
                loss = criterion(pred, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # --- 検証フェーズ ---
        net.eval()
        valid_loss = 0.0
        total_iou = 0.0
        with torch.no_grad():
            for image, depth, label in tqdm(valid_dataloader, desc=f"Epoch {epoch+1}/{n_epochs} [Valid]"):
                image, depth, label = image.to(device), depth.to(device), label.to(device)
                x = torch.cat((image, depth), dim=1)
                pred_logits = net(x)
                loss = criterion(pred_logits, label)
                valid_loss += loss.item()
                pred_labels = pred_logits.argmax(dim=1)
                total_iou += mean_iou(pred_labels, label, num_classes)

        avg_train_loss = train_loss / len(train_dataloader)
        avg_valid_loss = valid_loss / len(valid_dataloader)
        avg_iou = total_iou / len(valid_dataloader)

        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Valid mIoU: {avg_iou:.4f}')

    return net # 学習済みモデルを返す

# def run_inference(model, test_loader, device):
#     """テストデータで推論を行い、提出ファイルを作成する"""
#     model.eval()
#     predictions = []
#     with torch.no_grad():
#         print("Generating predictions for submission...")
#         for image, depth in tqdm(test_loader):
#             image, depth = image.to(device), depth.to(device)
#             x = torch.cat((image, depth), dim=1)
#             output = model(x)
#             pred = output.argmax(dim=1)
#             predictions.append(pred.cpu())
#     predictions = torch.cat(predictions, dim=0).numpy()
#     np.save('submission.npy', predictions)
#     print("Predictions saved to submission.npy")


# --- 4. Execution ---

if __name__ == '__main__':
    # パスの設定
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'Data', 'final_submission_dataset')
    
    # 設定の初期化
    config = TrainingConfig(dataset_root=data_path)
    set_seed(42)

    # データローダーの準備
    train_loader, valid_loader, test_loader = setup_dataloaders(config)
    print(f"Using device: {config.device}")

    # モデル、損失関数、最適化手法の定義
    model = PretrainedUNet(num_classes=config.num_classes, in_channels=config.in_channels).to(config.device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # 学習の実行
    trained_model = train_model(
        net=model,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        n_epochs=config.epochs,
        device=config.device,
        num_classes=config.num_classes
    )

    # モデルの保存
    model_path = "final_model.pt"
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # # テストデータでの推論
    # run_inference(trained_model, test_loader, config.device)

# 動作確認済み。実際に動かすときはgoogle colabを使わないと処理が終わらない