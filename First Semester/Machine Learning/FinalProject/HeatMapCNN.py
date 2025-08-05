import os
import json
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Utility: make Gaussian heatmaps from normalized keypoints
# -----------------------------------------------------------------------------
def make_gaussian_heatmaps(keypoints, heatmap_size, sigma=2):
    """
    keypoints:  ndarray, shape (K,3) with (x_norm, y_norm, v) in [0,1]/visibility
    heatmap_size: (H, W)
    returns: np.array of shape (K, H, W)
    """
    K = keypoints.shape[0]
    H, W = heatmap_size
    heatmaps = np.zeros((K, H, W), dtype=np.float32)

    ys = np.arange(H).reshape(H,1)
    xs = np.arange(W).reshape(1,W)

    for i, (x_n, y_n, v) in enumerate(keypoints):
        if v < 1: 
            continue  # skip invisible
        x = x_n * W
        y = y_n * H
        d2 = (xs - x)**2 + (ys - y)**2
        heatmaps[i] = np.exp(-d2 / (2 * sigma**2))
    return heatmaps

def draw_keypoints(img_tensor, coords, point_size=40):
    """
    img_tensor: torch.Tensor, shape (3, H, W), values in [0,1]
    coords: list of (x_norm, y_norm) in [0,1]
    """
    # 1) to H×W×3 NumPy uint8
    img = img_tensor.permute(1,2,0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    
    H, W = img.shape[:2]
    xs = [x * W for x, _ in coords]
    ys = [y * H for _, y in coords]

    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.scatter(xs, ys, s=point_size)      # default color
    plt.axis('off')
    plt.tight_layout()
    return plt

# -----------------------------------------------------------------------------
# 2. Dataset that returns (image_tensor, heatmap_tensor)
# -----------------------------------------------------------------------------
class KeypointHeatmapDataset(Dataset):
    def __init__(self, ann_file, img_dir,
                 img_size=(256,256), heatmap_size=(64,64), sigma=2):
        with open(ann_file) as f:
            coco = json.load(f)
        self.images = {img['id']: img for img in coco['images']}
        self.anns   = [a for a in coco['annotations'] if a.get('keypoints')]
        self.img_dir      = img_dir
        self.img_size     = img_size
        self.heatmap_size = heatmap_size
        self.sigma        = sigma

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann  = self.anns[idx]
        info = self.images[ann['image_id']]
        path = os.path.join(self.img_dir, info['file_name'])

        # --- load & preprocess image ---
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size) / 255.0
        img = torch.from_numpy(img).permute(2,0,1).float()

        # --- prepare keypoints & heatmaps ---
        kps = np.array(ann['keypoints'], dtype=np.float32).reshape(-1,3)
        # normalize from original image size
        kps[:,0] /= info['width']
        kps[:,1] /= info['height']
        hm = make_gaussian_heatmaps(kps, self.heatmap_size, sigma=self.sigma)
        hm = torch.from_numpy(hm)  # (K, H_hm, W_hm)

        return img, hm

# -----------------------------------------------------------------------------
# 3. Simple ConvNet → heatmaps
# -----------------------------------------------------------------------------
class HeatmapKPNet(nn.Module):
    def __init__(self, num_keypoints, heatmap_size=(64,64)):
        super().__init__()
        # three conv+pool blocks
        self.conv1 = nn.Conv2d(3,  16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2,2)
        # final 1×1 conv to predict K heatmaps
        self.head  = nn.Conv2d(64, num_keypoints, 1)
        self.heatmap_size = heatmap_size

    def forward(self, x):
        # x: (B,3,H,W)
        x = F.relu(self.conv1(x)); x = self.pool(x)  # H/2×W/2
        x = F.relu(self.conv2(x)); x = self.pool(x)  # H/4×W/4
        x = F.relu(self.conv3(x)); x = self.pool(x)  # H/8×W/8
        x = self.head(x)                             # (B,K,H/8,W/8)
        # upsample to desired heatmap resolution
        x = F.interpolate(x, size=self.heatmap_size,
                          mode='bilinear', align_corners=False)
        return x  # (B, K, H_hm, W_hm)

# -----------------------------------------------------------------------------
# 4. Training & quick inference demo
# -----------------------------------------------------------------------------
def main():
    # --- paths to your unzipped RoboFlow export ---
    train_ann = '/home/xgang/WinShared_D/Graduation/First_Year/Machine-Learning/FinalProject/TM-pose.v1-20250518-ver1.coco/train/_annotations.coco.json'
    train_imgs= '/home/xgang/WinShared_D/Graduation/First_Year/Machine-Learning/FinalProject/TM-pose.v1-20250518-ver1.coco/train'

    # 1) prepare data
    ds     = KeypointHeatmapDataset(
                 ann_file=train_ann,
                 img_dir =train_imgs,
                 img_size=(256,256),
                 heatmap_size=(64,64),
                 sigma=2)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)

    # 2) model, optimizer, loss
    K     = ds[0][1].shape[0]
    model = HeatmapKPNet(num_keypoints=K, heatmap_size=(64,64))
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # 3) one training step
    model.train()
    imgs, hms = next(iter(loader))   # grab first batch
    preds     = model(imgs)          # (B,K,64,64)
    loss      = loss_fn(preds, hms)
    loss.backward()
    opt.step()
    print(f"[Train] batch loss = {loss.item():.4f}")

    # 4) quick inference on one sample
    model.eval()
    with torch.no_grad():
        img, gt_hm = ds[0]
        out = model(img.unsqueeze(0))      # (1,K,64,64)
        coords = []
        H,W = out.shape[-2:]
        for k in range(K):
            flat = out[0,k].view(-1)
            idx  = torch.argmax(flat).item()
            y, x = divmod(idx, W)
            coords.append((x/W, y/H))     # normalized coords

    # out is (1,K,H_hm,W_hm), coords = [(x,y),...]
    vis = draw_keypoints(img, coords)
    vis.show()  # for interactive
    # or to save:
    vis.savefig("pred_keypoints.png", bbox_inches='tight', pad_inches=0)
    print("Saved prediction overlay to pred_keypoints.png") 
    # print(f"[Inference] normalized keypoints: {coords}")

if __name__ == '__main__':
    main()
