"""
自己教師あり学習を用いて事前学習を行い、得られた表現をLinear probingで評価する
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def take_indexes(sequences, indexes):
    return torch.gather(sequences, dim=1, index=indexes.unsqueeze(2).repeat(1, 1, sequences.shape[-1]))

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x_data, t_data, transform=None):
        self.x_data = [Image.fromarray(np.uint8(img)) for img in x_data]
        self.t_data = t_data 
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        img = self.x_data[idx]
        label = torch.tensor(self.t_data[idx], dtype=torch.long) 
        if self.transform:
            img = self.transform(img)
        return img, label

class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, mlp_dim, dropout=dropout)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        self.patcher = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
    def forward(self, img):
        x = self.patcher(img)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embedding
        return x

class PatchShuffle(nn.Module):
    def __init__(self, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
    def forward(self, x):
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = take_indexes(x, ids_keep)
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore).bool()
        return x_masked, mask, ids_restore

class MAE_Encoder(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3,
                 embed_dim=256, depth=4, heads=8, mlp_dim=512, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, heads, mlp_dim, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        if len(x.shape) == 4:
            x = self.patch_embed(x)
        
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

class MAE_Decoder(nn.Module):
    def __init__(self, embed_dim=256, decoder_embed_dim=128, decoder_depth=2, 
                 decoder_heads=4, mlp_dim=256, patch_size=4, in_channels=3):
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_embed_dim))
        num_patches = (32 // patch_size)**2 
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, num_patches, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_heads, mlp_dim) for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_channels, bias=True)

    def forward(self, x, restore_indices):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], restore_indices.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = take_indexes(x_, restore_indices)
        x = x_ + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x

class MAE_ViT(nn.Module):
    def __init__(self, encoder, decoder, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.shuffler = PatchShuffle(mask_ratio)
    
    def forward(self, img):
        x = self.encoder.patch_embed(img) 
        x_masked, mask, ids_restore = self.shuffler(x)
        
        latent = self.encoder(x_masked)
        
        pred = self.decoder(latent, ids_restore)
        target = self.patchify(img)
        loss = (pred[mask] - target[mask]) ** 2
        return loss.mean()
    
    def patchify(self, imgs):
        c = imgs.shape[1] 
        p = self.encoder.patch_embed.patch_size
        h, w = imgs.shape[2] // p, imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x

class Classifier(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        pooled_features = features.mean(dim=1)
        output = self.head(pooled_features)
        return output

def run_pretraining(model, dataloader, optimizer, n_epochs, device):
    print("\n--- Starting Pre-training ---")
    model.to(device)
    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        with tqdm(total=len(dataloader), unit="batch") as pbar:
            pbar.set_description(f"[Pre-train] Epoch {epoch+1}/{n_epochs}")
            for images, _ in dataloader:
                optimizer.zero_grad()
                images = images.to(device)
                loss = model(images)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                pbar.set_postfix(loss=np.mean(train_losses))
                pbar.update(1)
    print("Pre-training finished.")

def run_linear_probing(model, dataloader_train, dataloader_valid, criterion, optimizer, n_epochs, device):
    print("\n--- Starting Linear Probing ---")
    model.to(device)
    for epoch in range(n_epochs):
        model.train()
        train_losses, train_accs = [], []
        with tqdm(total=len(dataloader_train), unit="batch") as pbar:
            pbar.set_description(f"[Train] Epoch {epoch+1}/{n_epochs}")
            for images, labels in dataloader_train:
                optimizer.zero_grad()
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                train_accs.append((outputs.argmax(1) == labels).float().mean().item())
                pbar.set_postfix(loss=np.mean(train_losses), acc=np.mean(train_accs))
                pbar.update(1)

        model.eval()
        valid_losses, valid_accs = [], []
        with tqdm(total=len(dataloader_valid), unit="batch") as pbar:
            pbar.set_description(f"[Valid] Epoch {epoch+1}/{n_epochs}")
            for images, labels in dataloader_valid:
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = model(images)
                loss = criterion(outputs, labels)
                valid_losses.append(loss.item())
                valid_accs.append((outputs.argmax(1) == labels).float().mean().item())
                pbar.set_postfix(loss=np.mean(valid_losses), acc=np.mean(valid_accs))
                pbar.update(1)
    print("Linear probing finished.")

if __name__ == '__main__':
    fix_seed(seed=42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    data_path = os.path.join(script_dir, '..', 'Data', 'Lecture10_dataset')

    x_train_all = np.load(os.path.join(data_path, 'x_train.npy'))
    t_train_all = np.load(os.path.join(data_path, 't_train.npy'))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    x_train, x_val, t_train, t_val = train_test_split(
        x_train_all, t_train_all, test_size=3000, random_state=42
    )

    train_data = CustomDataset(x_train, t_train, transform=transform)
    valid_data = CustomDataset(x_val, t_val, transform=transform)

    dataloader_train = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    dataloader_valid = torch.utils.data.DataLoader(valid_data, batch_size=128, shuffle=False)

    pretrain_encoder = MAE_Encoder(image_size=32, in_channels=3, embed_dim=256, depth=4, heads=8, mlp_dim=512)
    pretrain_decoder = MAE_Decoder(embed_dim=256, decoder_embed_dim=128, decoder_depth=2, decoder_heads=4, mlp_dim=256)
    pretrain_model = MAE_ViT(pretrain_encoder, pretrain_decoder)
    optimizer_pretrain = optim.AdamW(pretrain_model.parameters(), lr=1e-3)
    
    run_pretraining(
        model=pretrain_model,
        dataloader=dataloader_train,
        optimizer=optimizer_pretrain,
        n_epochs=20,
        device=device
    )

    classifier_model = Classifier(pretrain_model.encoder)
    optimizer_probe = optim.AdamW(classifier_model.head.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    run_linear_probing(
        model=classifier_model,
        dataloader_train=dataloader_train,
        dataloader_valid=dataloader_valid,
        criterion=criterion,
        optimizer=optimizer_probe,
        n_epochs=10,
        device=device
    )