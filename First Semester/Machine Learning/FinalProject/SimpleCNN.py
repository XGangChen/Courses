import os, json
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class KeypointDataset(Dataset):
    def __init__(self, ann_file, img_dir, img_size=(256,256)):
        with open(ann_file) as f:
            coco = json.load(f)
        # map image_id → filename/size
        self.images = {i['id']:i for i in coco['images']}
        # only keep anns with keypoints
        self.anns = [a for a in coco['annotations'] if a.get('keypoints')]
        self.img_dir  = img_dir
        self.img_size = img_size  # e.g. (H, W)
    
    def __len__(self):
        return len(self.anns)
    
    def __getitem__(self, idx):
        a = self.anns[idx]
        im_info = self.images[a['image_id']]
        path = os.path.join(self.img_dir, im_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # resize + normalize to [0,1]
        img = cv2.resize(img, self.img_size) / 255.0

        # "torch.from_numpy()" Converts the NumPy array to a PyTorch tensor. 
        # "permute(2,0,1)" changes the shape from (H,W,C) to (C,H,W), which is the expected format for PyTorch models.
        img = torch.from_numpy(img).permute(2,0,1).float()

        # keypoints: [x1,y1,vis1, x2,y2,vis2, ...]
        kps = torch.tensor(a['keypoints'], dtype=torch.float32).view(-1,3)
        # normalize coords by original size
        h0, w0 = im_info['height'], im_info['width']
        kps[:,0] /= w0
        kps[:,1] /= h0
        # keep only x,y and flatten → (2*K,)
        kps = kps[:,:2].reshape(-1)
        return img, kps

class SimpleKPNet(nn.Module):
    def __init__(self, num_keypoints, img_size=(256,256)):
        super().__init__()
        H,W = img_size
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2,2)
        # three pools → spatial dims H/8 × W/8
        feat_dim   = 64 * (H//8) * (W//8)
        self.fc    = nn.Linear(feat_dim, num_keypoints*2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# usage example
if __name__ == '__main__':
    train_ds = KeypointDataset(
        ann_file='/home/xgang/WinShared_D/Graduation/First_Year/Machine-Learning/FinalProject/TM-pose.v1-20250518-ver1.coco/train/_annotations.coco.json',
        img_dir ='/home/xgang/WinShared_D/Graduation/First_Year/Machine-Learning/FinalProject/TM-pose.v1-20250518-ver1.coco/train',
        img_size=(256,256)
    )
    loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    
    # assume each annotation has K keypoints
    K = 7
    model = SimpleKPNet(num_keypoints=K, img_size=(256,256))
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    for imgs, targets in loader:
        preds = model(imgs)               # (B, 2K)
        loss  = F.mse_loss(preds, targets)
        loss.backward(); opt.step(); opt.zero_grad()
        print(f"train loss: {loss.item():.4f}")
        break
