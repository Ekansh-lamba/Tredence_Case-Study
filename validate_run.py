import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt


class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pw = self.weight * gates
        return F.linear(x, pw, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity(self):
        g = self.get_gates()
        return (g < 1e-2).float().mean().item()


class SelfPruningNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = PrunableLinear(256 * 4, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, num_classes)
        self.drop = nn.Dropout(0.4)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x

    def sparsity_loss(self):
        sl = 0.0
        for layer in [self.fc1, self.fc2, self.fc3]:
            sl += torch.sigmoid(layer.gate_scores).abs().sum()
        return sl

    def overall_sparsity(self):
        total, pruned = 0, 0
        for layer in [self.fc1, self.fc2, self.fc3]:
            g = layer.get_gates()
            pruned += (g < 1e-2).float().sum().item()
            total += g.numel()
        return pruned / total

    def all_gate_values(self):
        parts = [layer.get_gates().cpu().numpy().flatten() for layer in [self.fc1, self.fc2, self.fc3]]
        return np.concatenate(parts)


print("=== SMOKE TEST — 2 epochs, 1000 train / 500 test, lambda=5e-4 ===", flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}", flush=True)

mean = (0.4914, 0.4822, 0.4465)
std  = (0.2023, 0.1994, 0.2010)
tr = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
te = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tr)
test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=te)

train_sub = Subset(train_ds, list(range(1000)))
test_sub  = Subset(test_ds,  list(range(500)))

train_loader = DataLoader(train_sub, batch_size=64, shuffle=True,  num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_sub,  batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
print(f"loaders ready — train batches={len(train_loader)}, test batches={len(test_loader)}", flush=True)

model = SelfPruningNet().to(device)
lam   = 5e-4
opt   = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
sch   = CosineAnnealingLR(opt, T_max=2)

for ep in range(1, 3):
    model.train()
    tot_loss, cls_sum = 0.0, 0.0
    nb = len(train_loader)
    for i, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        cls = F.cross_entropy(logits, yb)
        sp  = model.sparsity_loss()
        loss = cls + lam * sp
        loss.backward()
        opt.step()
        tot_loss += loss.item()
        cls_sum  += cls.item()
        print(f"  ep={ep} batch={i+1}/{nb} loss={loss.item():.4f} cls={cls.item():.4f}", flush=True)
    sch.step()
    sp_pct = model.overall_sparsity() * 100
    print(f"  >> ep={ep}/2 done | tot={tot_loss/nb:.4f} | cls={cls_sum/nb:.4f} | sparsity={sp_pct:.2f}%\n", flush=True)

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        correct += (model(xb).argmax(1) == yb).sum().item()
        total   += yb.size(0)
acc = correct / total
gv  = model.all_gate_values()
print(f"test acc={acc*100:.2f}% | sparsity={model.overall_sparsity()*100:.2f}%", flush=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(gv, bins=80, color="#4c72b0", edgecolor="white", linewidth=0.3)
ax.axvline(x=1e-2, color="red", linestyle="--", linewidth=1.5, label="Pruning threshold (1e-2)")
ax.set_yscale("log")
ax.set_xlabel("Gate Value")
ax.set_ylabel("Count (log scale)")
ax.set_title(f"Gate Distribution — λ=5e-4 [SMOKE TEST]")
ax.legend()
plt.tight_layout()
plt.savefig("gate_dist_smoke_test.png", dpi=150)
plt.close()
print("gate plot saved -> gate_dist_smoke_test.png", flush=True)

print("\n=== VALIDATION COMPLETE ===", flush=True)
print(f"  PrunableLinear   OK  (weight, bias, gate_scores as nn.Parameter)", flush=True)
print(f"  SelfPruningNet   OK  (conv blocks + AdaptiveAvgPool + PrunableLinear stack)", flush=True)
print(f"  sparsity_loss    OK  (L1 sum of sigmoid gates)", flush=True)
print(f"  overall_sparsity OK  ({model.overall_sparsity()*100:.2f}%)", flush=True)
print(f"  all_gate_values  OK  (shape={gv.shape})", flush=True)
print(f"  forward pass     OK  (acc={acc*100:.2f}% on 500 samples, 2 epochs)", flush=True)
print(f"  plot saved       OK  (gate_dist_smoke_test.png)", flush=True)
