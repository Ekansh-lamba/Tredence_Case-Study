import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
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


def get_cifar10_loaders():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
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
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=te)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, test_loader


def train_one_epoch(model, loader, opt, device, lam, ep):
    model.train()
    tot_loss, cls_loss_sum = 0.0, 0.0
    n_batches = len(loader)
    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        cls = F.cross_entropy(logits, yb)
        sp = model.sparsity_loss()
        loss = cls + lam * sp
        loss.backward()
        opt.step()
        tot_loss += loss.item()
        cls_loss_sum += cls.item()
        if (i + 1) % 50 == 0 or (i + 1) == n_batches:
            print(f"  ep={ep} batch={i+1}/{n_batches} loss={loss.item():.4f} cls={cls.item():.4f}", flush=True)
    return tot_loss / n_batches, cls_loss_sum / n_batches


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total


def run_experiment(lam):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- lambda={lam} | device={device} ---")
    train_loader, test_loader = get_cifar10_loaders()
    model = SelfPruningNet().to(device)
    opt = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = CosineAnnealingLR(opt, T_max=40)
    for ep in range(1, 41):
        tot, cls = train_one_epoch(model, train_loader, opt, device, lam, ep)
        sch.step()
        sp = model.overall_sparsity() * 100
        print(f"  >> ep={ep}/40 done | tot={tot:.4f} | cls={cls:.4f} | sparsity={sp:.2f}%", flush=True)
    acc = evaluate(model, test_loader, device)
    sp = model.overall_sparsity()
    gv = model.all_gate_values()
    print(f"  -> final acc={acc*100:.2f}% | sparsity={sp*100:.2f}%")
    return {"lambda_val": lam, "test_accuracy": acc, "sparsity_level": sp, "gate_values": gv}


def plot_gate_distribution(gate_values, lam):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gate_values, bins=80, color="#4c72b0", edgecolor="white", linewidth=0.3)
    ax.axvline(x=1e-2, color="red", linestyle="--", linewidth=1.5, label="Pruning threshold (1e-2)")
    ax.set_yscale("log")
    ax.set_xlabel("Gate Value", fontsize=12)
    ax.set_ylabel("Count (log scale)", fontsize=12)
    ax.set_title(f"Gate Distribution — λ={lam}", fontsize=13)
    ax.legend()
    fname = f"gate_dist_lambda_{lam}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"saved {fname}")


def main():
    lambdas = [1e-5, 5e-4, 5e-3]
    results = []
    for lam in lambdas:
        res = run_experiment(lam)
        results.append(res)
        plot_gate_distribution(res["gate_values"], lam)

    print(f"\n{'Lambda':<12} {'Test Accuracy (%)':<22} {'Sparsity Level (%)':<20}")
    print("-" * 54)
    for r in results:
        print(f"{r['lambda_val']:<12} {r['test_accuracy']*100:<22.2f} {r['sparsity_level']*100:<20.2f}")


main()
