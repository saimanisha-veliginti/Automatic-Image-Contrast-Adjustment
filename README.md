# Import all the necessary libs
import os
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from PIL import Image
from google.colab import files
from skimage.color import rgb2lab
from piq import ssim
import cv2
class ConvBlock(nn.Module):
    def _init_(self, in_ch, out_ch):
        super()._init_()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.block(x)
class UNet(nn.Module):
    def _init_(self, ch=32):
        super()._init_()
        self.down1 = ConvBlock(3, ch)
        self.down2 = ConvBlock(ch, ch*2)
        self.up1   = ConvBlock(ch*2, ch)
        self.final = nn.Conv2d(ch, 3, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        u1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        return torch.sigmoid(self.final(self.up1(u1)))
class LambdaMLP(nn.Module):
    def _init_(self):
        super()._init_()
        self.net = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()   # returns >0
        )

    def forward(self, x):
        return self.net(x)
mlp = LambdaMLP().cuda().eval()  # NO TRAINING, just inference
name = "/content/images/images/sample.png"
img = np.array(Image.open(name).convert("RGB")).astype(np.float32) / 255.
H, W, _ = img.shape
img_t = torch.tensor(img).permute(2,0,1)[None].cuda()
lab = rgb2lab(img)
L = lab[:,:,0] / 100.0
meanL = L.mean()
varL  = L.var()
meanR = img[:,:,0].mean()
meanG = img[:,:,1].mean()
meanB = img[:,:,2].mean()
gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)/255.
gx = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
gy = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
ent = (gx*2+gy*2).var()

stats = torch.tensor([meanL,varL,meanR,meanG,meanB,ent], dtype=torch.float32).cuda()
lambda_weight = float(mlp(stats).item())
print("λ predicted =", lambda_weight)

net = UNet().cuda()
z   = torch.randn_like(img_t)
opt = torch.optim.Adam(net.parameters(), lr=3e-4)
mse = nn.MSELoss()
def contrast_loss(x):
    x_np = x[0].permute(1,2,0).detach().cpu().numpy()
    L = rgb2lab(x_np)[:,:,0] / 100.0
    return 1.0 - torch.tensor(L).var().to(x.device)
best_img = None
best_var = -999
for i in range(800):  # original value is 1800
    z_noisy = z + 0.03 * torch.randn_like(z)
    out     = net(z_noisy)
    loss = mse(out, img_t) + lambda_weight * contrast_loss(out)
    opt.zero_grad()
    loss.backward()
    opt.step()
    x_np = out[0].permute(1,2,0).detach().cpu().numpy()
    L = rgb2lab(x_np)[:,:,0] / 100.0
    if L.var() > best_var:
        best_var = float(L.var())
        best_img = (x_np*255).clip(0,255).astype(np.uint8)

    if i % 200 == 0:
        print("iter", i, "loss", float(loss))
out_img = Image.fromarray(best_img)
display(out_img)
out_img.save("enhanced.png")
print("saved as enhanced.png")
import os
if not os.path.exists('/content/images/'):
    os.makedirs('/content/images/')

# Assuming 'name' variable holds the uploaded zip file name
# from google.colab import files
# files.upload()     # choose images.zip (already done)
!unzip "{name}" -d /content/images/



[Dataset.zip](https://github.com/user-attachments/files/23492933/Dataset.zip)
