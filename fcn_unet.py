import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Synthetic segmentation dataset
def draw_random_shape(h, w):
    img = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.float32)

    # random background noise (makes it non-trivial)
    img += 0.1 * np.random.randn(h, w).astype(np.float32)

    shape_type = np.random.choice(["circle", "rect"])
    cy, cx = np.random.randint(h//4, 3*h//4), np.random.randint(w//4, 3*w//4)

    if shape_type == "circle":
        r = np.random.randint(min(h, w)//10, min(h, w)//5)
        yy, xx = np.ogrid[:h, :w]
        region = (yy - cy)**2 + (xx - cx)**2 <= r**2
        mask[region] = 1.0
        img[region] += 1.0
    else:
        rh = np.random.randint(h//10, h//4)
        rw = np.random.randint(w//10, w//4)
        y0 = np.clip(cy - rh//2, 0, h-1)
        y1 = np.clip(cy + rh//2, 0, h)
        x0 = np.clip(cx - rw//2, 0, w-1)
        x1 = np.clip(cx + rw//2, 0, w)
        mask[y0:y1, x0:x1] = 1.0
        img[y0:y1, x0:x1] += 1.0

    # normalize-ish
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return img, mask


class ShapesSegmentation(Dataset):
    def __init__(self, n=2000, h=128, w=128):
        self.n = n
        self.h = h
        self.w = w

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img, mask = draw_random_shape(self.h, self.w)
        # (C,H,W)
        img_t = torch.from_numpy(img).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return img_t, mask_t


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
    )


# FCN-ish (Long et al.) minimal version
# - encoder downsamples
# - create class "score maps" at different scales
# - upsample + add scores (additive fusion)
class FCNish(nn.Module):
    def __init__(self, in_ch=1, n_classes=1, base=32):
        super().__init__()
        # Encoder
        self.enc1 = conv_block(in_ch, base)         # 128
        self.enc2 = conv_block(base, base*2)        # 64
        self.enc3 = conv_block(base*2, base*4)      # 32

        self.pool = nn.MaxPool2d(2)

        # Convert features to score maps (1x1 conv = per-pixel classifier at that scale)
        self.score3 = nn.Conv2d(base*4, n_classes, 1)  # 32x32 score
        self.score2 = nn.Conv2d(base*2, n_classes, 1)  # 64x64 score
        self.score1 = nn.Conv2d(base,   n_classes, 1)  # 128x128 score

    def forward(self, x):
        f1 = self.enc1(x)        # 128x128
        x = self.pool(f1)        # 64x64
        f2 = self.enc2(x)        # 64x64
        x = self.pool(f2)        # 32x32
        f3 = self.enc3(x)        # 32x32

        s3 = self.score3(f3)     # 32x32
        s3_up = F.interpolate(s3, scale_factor=2, mode="bilinear", align_corners=False)  # 64x64

        s2 = self.score2(f2)     # 64x64
        fuse_64 = s3_up + s2     # add scores

        fuse_128 = F.interpolate(fuse_64, scale_factor=2, mode="bilinear", align_corners=False)  # 128x128

        s1 = self.score1(f1)     # 128x128
        out = fuse_128 + s1      # add scores at full res

        return out


# U-Net minimal version
# - encoder downsamples
# - decoder upsamples
# - concatenate encoder features (feature-space memory)
class UNet(nn.Module):
    def __init__(self, in_ch=1, n_classes=1, base=32):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)       # 128
        self.enc2 = conv_block(base, base*2)      # 64
        self.enc3 = conv_block(base*2, base*4)    # 32

        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(base*4, base*2, kernel_size=2, stride=2)  # 32->64
        self.dec2 = conv_block(base*4, base*2)  # concat: (base*2 from up) + (base*2 skip)

        self.up1 = nn.ConvTranspose2d(base*2, base, kernel_size=2, stride=2)    # 64->128
        self.dec1 = conv_block(base*2, base)    # concat: base + base

        self.out = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        f1 = self.enc1(x)          # 128
        x = self.pool(f1)          # 64
        f2 = self.enc2(x)          # 64
        x = self.pool(f2)          # 32
        f3 = self.enc3(x)          # 32

        x = self.up2(f3)           # 64
        x = torch.cat([x, f2], dim=1)  # feature memory
        x = self.dec2(x)

        x = self.up1(x)            # 128
        x = torch.cat([x, f1], dim=1)
        x = self.dec1(x)

        return self.out(x)


# Training utilities
def iou_score(pred_logits, target, thresh=0.5):
    pred = (torch.sigmoid(pred_logits) > thresh).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = ((pred + target) > 0).float().sum(dim=(1,2,3))
    return (inter / (union + 1e-6)).mean().item()

def train_one(model, loader, device, steps=400, lr=1e-3):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    it = iter(loader)
    losses = []
    ious = []

    for step in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)

        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        if (step+1) % 50 == 0:
            with torch.no_grad():
                iou = iou_score(logits, y)
            losses.append(loss.item())
            ious.append(iou)
            print(f"step {step+1:4d} | loss {loss.item():.4f} | IoU {iou:.3f}")

    return losses, ious

@torch.no_grad()
def visualize(model, device, n=3, h=128, w=128, title=""):
    model.eval()
    fig, axes = plt.subplots(n, 3, figsize=(9, 3*n))
    for i in range(n):
        img, mask = draw_random_shape(h, w)
        x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
        logits = model(x)
        pred = torch.sigmoid(logits).cpu().squeeze().numpy()

        axes[i,0].imshow(img, cmap="gray")
        axes[i,0].set_title("input")
        axes[i,0].axis("off")

        axes[i,1].imshow(mask, cmap="gray")
        axes[i,1].set_title("gt mask")
        axes[i,1].axis("off")

        axes[i,2].imshow(pred, cmap="gray")
        axes[i,2].set_title("pred")
        axes[i,2].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    ds = ShapesSegmentation(n=2000, h=128, w=128)
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)

    fcn = FCNish()
    unet = UNet()

    print("\nTraining FCN-ish...")
    train_one(fcn, loader, device, steps=400)

    print("\nTraining U-Net...")
    train_one(unet, loader, device, steps=400)

    visualize(fcn, device, title="FCN-ish (additive score skips)")
    visualize(unet, device, title="U-Net (concat feature skips)")


if __name__ == "__main__":
    main()
