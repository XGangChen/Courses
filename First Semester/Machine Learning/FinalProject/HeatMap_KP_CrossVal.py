import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut,
    TimeSeriesSplit, GroupKFold
)
import matplotlib.pyplot as plt
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

class KeypointHeatmapDataset(Dataset):
    def __init__(self, ann_file, img_dir, img_size=(256,256), heatmap_size=(64,64), sigma=2):
        with open(ann_file) as f:
            data = json.load(f)
        self.img_dir      = img_dir
        self.img_size     = img_size
        self.heatmap_size = heatmap_size
        self.sigma        = sigma
        self.num_kp       = len(data['annotations'][0]['keypoints']) // 3
        self.imgs         = [img['file_name'] for img in data['images']]
        self.kps          = [np.array(ann['keypoints']).reshape(-1,3)[:,:2] / np.array(img_size)[None]
                             for ann in data['annotations']]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.imgs[idx])
        img = cv2.imread(path)
        img = cv2.resize(img, self.img_size)
        img = img[...,::-1].transpose(2,0,1) / 255.0
        hm  = make_gaussian_heatmaps(self.kps[idx], self.heatmap_size, self.sigma)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(hm, dtype=torch.float32)

class SimpleHeatmapNet(nn.Module):
    def __init__(self, num_kp=7):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 256 -> 128

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 128 -> 64

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),  # 64 -> 64
        )
        self.head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, num_kp, 1),   # 64x64 -> K heatmaps
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x  # B x K x 64 x 64

def make_gaussian_heatmaps(kps, size, sigma=2):
    K = kps.shape[0]
    H, W = size
    hm = np.zeros((K,H,W), dtype=np.float32)
    for i,(x,y) in enumerate(kps):
        cx, cy = int(x*W), int(y*H)
        th = np.zeros((H,W), dtype=np.float32)
        cv2.circle(th,(cx,cy), sigma*3, 1, -1)
        th = cv2.GaussianBlur(th, (0,0), sigma)
        if th.max()>0:
            hm[i] = th / th.max()
    return hm

def train_one_epoch(model, loader, optim, loss_fn):
    model.train()
    total_loss = 0
    for imgs, tgs in loader:
        imgs, tgs = imgs.to(device), tgs.to(device)
        preds = model(imgs)
        loss = loss_fn(preds, tgs)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def eval_one_fold(model, loader, loss_fn):
    model.eval()
    total, count = 0, 0
    with torch.no_grad():
        for imgs, tgs in loader:
            imgs, tgs = imgs.to(device), tgs.to(device)
            preds = model(imgs)
            total += loss_fn(preds, tgs).item() * imgs.size(0)
            count += imgs.size(0)
    return total / count

def compare_model_cv(model_cls, ds_args, cv, n_epochs=10):
    ds = KeypointHeatmapDataset(**ds_args)
    indices = list(range(len(ds)))

    labels = np.zeros(len(ds), dtype=int)      # for StratifiedKFold
    groups = np.arange(len(ds))                # for GroupKFold

    if isinstance(cv, StratifiedKFold):
        split_gen = cv.split(indices, labels)
    elif isinstance(cv, GroupKFold):
        split_gen = cv.split(indices, groups=groups)
    else:
        split_gen = cv.split(indices)

    train_losses_folds = []
    val_losses_folds   = []

    for fold, (train_idx, val_idx) in enumerate(split_gen):
        train_dl = DataLoader(Subset(ds, train_idx), batch_size=8, shuffle=True, num_workers=4)
        val_dl   = DataLoader(Subset(ds, val_idx), batch_size=8, shuffle=False, num_workers=4)

        model = model_cls(num_kp=ds.num_kp).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        train_losses = []
        val_losses   = []
        for epoch in range(1, n_epochs+1):
            tr_loss = train_one_epoch(model, train_dl, optim, loss_fn)
            val_loss = eval_one_fold(model, val_dl, loss_fn)
            train_losses.append(tr_loss)
            val_losses.append(val_loss)
            print(f"Fold {fold+1}, Epoch {epoch}/{n_epochs} — Train: {tr_loss:.4f}, Val: {val_loss:.4f}")

        train_losses_folds.append(train_losses)
        val_losses_folds.append(val_losses)

    avg_train = np.mean(train_losses_folds, axis=0)
    avg_val   = np.mean(val_losses_folds, axis=0)
    return avg_train, avg_val

if __name__ == "__main__":
    ann_file = '/home/xgang/WinShared_D/Graduation/First_Year/Machine-Learning/FinalProject/TM-pose.v1-20250518-ver1.coco/train/_annotations.coco.json'
    img_dir  = '/home/xgang/WinShared_D/Graduation/First_Year/Machine-Learning/FinalProject/TM-pose.v1-20250518-ver1.coco/train'
    ds_args = {'ann_file': ann_file, 'img_dir': img_dir,
               'img_size': (256,256), 'heatmap_size': (64,64), 'sigma': 2}

    cv_strategies = {
        'K-Fold': KFold(n_splits=5, shuffle=True, random_state=0),
        'Stratified K-Fold': StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        'Leave-One-Out': LeaveOneOut(),
        'Time Series': TimeSeriesSplit(n_splits=5),
        'Group K-Fold': GroupKFold(n_splits=5)
    }

    n_epochs = 10
    results = {}

    model_cls = SimpleHeatmapNet
    for name, cv in cv_strategies.items():
        print(f"Running {name}...")
        train_curve, val_curve = compare_model_cv(model_cls, ds_args, cv, n_epochs)
        results[name] = (train_curve, val_curve)
    
    for name, (train_curve, val_curve) in results.items():
        plt.figure()
        plt.plot(range(1, n_epochs+1), train_curve, label='Train')
        plt.plot(range(1, n_epochs+1), val_curve,   label='Validation')
        plt.title(f'{name} Convergence')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    
