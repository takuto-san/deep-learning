"""
ViTによる画像分類を実装する
"""
import os
import sys
import random
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import h5py
from os.path import join
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import logging
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F


# 要素にドットでアクセスできる辞書クラス
class Args(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

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

class SelfAttention(nn.Module):
  def __init__(self, dim, heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
      super().__init__()
      self.heads = heads
      head_dim = dim // heads
      self.scale = head_dim ** -0.5
      self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
      self.attn_drop = nn.Dropout(attn_drop)
      self.proj = nn.Linear(dim, dim)
      self.proj_drop = nn.Dropout(proj_drop)

  def forward(self, x):
      B, N, C = x.shape
      qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
      q, k, v = qkv[0], qkv[1], qkv[2]
      attn = (q @ k.transpose(-2, -1)) * self.scale
      attn = attn.softmax(dim=-1)
      attn = self.attn_drop(attn)
      x = (attn @ v).transpose(1, 2).reshape(B, N, C)
      x = self.proj(x)
      x = self.proj_drop(x)
      return x

class Block(nn.Module):
  def __init__(self, dim, heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
    super().__init__()
    self.norm1 = nn.LayerNorm(dim)
    self.attn = SelfAttention(dim, heads=heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
    self.norm2 = nn.LayerNorm(dim)
    mlp_hidden_dim = int(dim * mlp_ratio)
    self.mlp = nn.Sequential(
        nn.Linear(dim, mlp_hidden_dim),
        nn.GELU(),
        nn.Dropout(drop),
        nn.Linear(mlp_hidden_dim, dim),
        nn.Dropout(drop)
    )

  def forward(self, x):
    x = x + self.attn(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x


class PatchEmbedding(nn.Module):
  def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
      super().__init__()
      self.img_size = img_size
      self.patch_size = patch_size
      self.grid_size = img_size // patch_size
      self.num_patches = self.grid_size * self.grid_size
      self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

  def forward(self, x):
      B, C, H, W = x.shape
      x = self.proj(x).flatten(2).transpose(1, 2)
      return x

class ViT(nn.Module):
  def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
               embed_dim=768, depth=12, heads=12, mlp_ratio=4., qkv_bias=True,
               drop_rate=0., attn_drop_rate=0.):
      super().__init__()
      self.num_classes = num_classes
      self.num_features = self.embed_dim = embed_dim
      self.patch_embed = PatchEmbedding(
          img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
      num_patches = self.patch_embed.num_patches
      self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
      self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
      self.pos_drop = nn.Dropout(p=drop_rate)
      self.blocks = nn.ModuleList([
          Block(
              dim=embed_dim, heads=heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
              drop=drop_rate, attn_drop=attn_drop_rate)
          for _ in range(depth)])
      self.norm = nn.LayerNorm(embed_dim)
      self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
      torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
      torch.nn.init.trunc_normal_(self.cls_token, std=.02)
      self.apply(self._init_weights)

  def _init_weights(self, m):
      if isinstance(m, nn.Linear):
          torch.nn.init.trunc_normal_(m.weight, std=.02)
          if isinstance(m, nn.Linear) and m.bias is not None:
              nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.LayerNorm):
          nn.init.constant_(m.bias, 0)
          nn.init.constant_(m.weight, 1.0)

  def configure_optimizers(self, train_config):
      decay = set()
      no_decay = set()
      whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
      blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
      for mn, m in self.named_modules():
          for pn, p in m.named_parameters():
              fpn = '%s.%s' % (mn, pn) if mn else pn
              if pn.endswith('bias'):
                  no_decay.add(fpn)
              elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                  decay.add(fpn)
              elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                  no_decay.add(fpn)
      no_decay.add('pos_embed')
      no_decay.add('cls_token')
      param_dict = {pn: p for pn, p in self.named_parameters()}
      inter_params = decay & no_decay
      union_params = decay | no_decay
      assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
      assert len(param_dict.keys() - union_params) == 0, f"parameters {param_dict.keys() - union_params} were not separated into either decay/no_decay set!"
      optim_groups = [
          {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
          {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
      ]
      optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
      return optimizer

  def forward(self, x, targets=None):
      B = x.shape[0]
      x = self.patch_embed(x)
      cls_tokens = self.cls_token.expand(B, -1, -1)
      x = torch.cat((cls_tokens, x), dim=1)
      x = x + self.pos_embed
      x = self.pos_drop(x)
      for blk in self.blocks:
          x = blk(x)
      x = self.norm(x)
      logits = self.head(x[:, 0])
      loss = None
      if targets is not None:
          loss = F.cross_entropy(logits.view(-1, self.num_classes), targets.view(-1))
      return logits, loss

logger = logging.getLogger(__name__)

class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = False
    warmup_tokens = 375e6
    final_tokens = 260e9
    ckpt_path = None
    num_workers = 0

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        if self.config.ckpt_path:
            torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            shuffle = is_train
            loader = DataLoader(data, shuffle=shuffle, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y) in pbar:
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y)
                    loss = loss.mean()
                    losses.append(loss.item())

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()
                        if self.tokens < config.warmup_tokens:
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    pbar.set_description(f"epoch {epoch_num+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        self.tokens = 0
        for epoch in range(config.max_epochs):
            run_epoch('train', epoch_num=epoch)
            if self.test_dataset is not None:
                test_loss = run_epoch('test')

            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()


if __name__ == '__main__':
    seed=42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.set_printoptions(edgeitems=1e3)

    # パス設定
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'Data', 'Lecture08_dataset')

    # 学習データの読み込み
    x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
    t_train = np.load(os.path.join(data_dir, 't_train.npy'))
    x_test = np.load(os.path.join(data_dir, 'x_test.npy'))

    # テストデータの読み込み
    trainval_data = train_dataset(x_train, t_train)
    test_data = test_dataset(x_test)

    # データの分割
    val_size = 3000
    train_data, valid_data = torch.utils.data.random_split(trainval_data, [len(trainval_data) - val_size, val_size])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
    )
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
    )

    train_data.dataset.transform = train_transform
    valid_data.dataset.transform = test_transform
    test_data.transform = test_transform

    block_size = 256
    args = Args({
        'img_size': 32,
        'patch_size': 4,
        'num_classes': 10,
        'embed_dim': 512,
        'depth': 6,
        'heads': 8,
        'mlp_ratio': 4.0,
        'qkv_bias': True,
        'drop_rate': 0.1,
        'attn_drop_rate': 0.1,
    })
    model = ViT(**vars(args))

    max_epochs=10
    model_path = os.path.join(script_dir, '..', 'Data', 'models', 'trained_vision_model.pth')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    num_patches = (args.img_size // args.patch_size)**2
    final_tokens_estimate = max_epochs * len(train_data) * num_patches

    tconf = TrainerConfig(
        max_epochs=max_epochs,
        batch_size=128,
        learning_rate=5e-4,
        lr_decay=True,
        warmup_tokens=512*10,
        final_tokens=final_tokens_estimate,
        grad_norm_clip=1.0,
        weight_decay=0.1,
        ckpt_path=model_path,
        num_workers=4
    )

    trainer = Trainer(model, train_data, valid_data, tconf)
    trainer.train()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        exit()

    model.to(device)
    model.eval()

    train_dataloader = DataLoader(train_data, shuffle=True, pin_memory=True,
                                  batch_size=tconf.batch_size, num_workers=tconf.num_workers)
    valid_dataloader = DataLoader(valid_data, shuffle=False, pin_memory=True,
                                 batch_size=tconf.batch_size, num_workers=tconf.num_workers)

    train_acc, valid_acc = 0., 0.
    with torch.no_grad():
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x, y)
            acc = (torch.argmax(logits, dim=1) == y).float().sum().cpu()
            train_acc += acc

        for x, y in valid_dataloader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x, y)
            acc = (torch.argmax(logits, dim=1) == y).float().sum().cpu()
            valid_acc += acc

    print(f"Train Acc.: {(train_acc / len(train_data)):.4f}")
    print(f"Valid Acc. : {(valid_acc / len(valid_data)):.4f}")

# 実際に動かすときはGoogleColabでやった方がいい（処理が終わらない）