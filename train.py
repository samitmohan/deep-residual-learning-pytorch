"""
Reproduces CIFAR-10 experiments from Section 4.2 of
"Deep Residual Learning for Image Recognition" (He et al., 2015).

Trains plain and residual networks at depths 20/32/44/56/110 with:
- SGD with momentum 0.9, weight decay 1e-4
- LR 0.1 (0.01 warmup for n >= 56), decayed by 10x at 32k and 48k iterations
- Data augmentation: 4-pixel padding + random crop, horizontal flip, per-channel normalization
- Batch size 128, 160 epochs (~64k iterations)
"""
import torch
import os
import numpy as np
from torchvision import transforms
from models import CifarResNet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets.cifar import CIFAR10
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from datetime import datetime
from tqdm import tqdm
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int, choices=(20, 32, 44, 56, 110))
parser.add_argument('-r', '--residual', action='store_true')
parser.add_argument('-o', '--option', type=str, choices=('A', 'B'), default=None)
parser.add_argument('--quick', action='store_true', help='Run only 5 epochs for debugging')
args = parser.parse_args()
large = args.n >= 56

writer = SummaryWriter()
now = datetime.now()

model = CifarResNet(args.n, residual=args.residual, option=args.option).to(device)

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model: CifarResNet-{args.n} | Residual: {args.residual} | Option: {args.option}')
print(f'Parameters: {param_count:,} ({param_count/1e6:.2f}M)')

loss_fn = CrossEntropyLoss()
optimizer = SGD(
    model.parameters(), lr=0.01 if large else 0.1, weight_decay=0.0001, momentum=0.9
)

# LR decayed by 10x at 32k and 48k iterations (Section 4.2)
scheduler = MultiStepLR(optimizer, milestones=(32_000, 48_000), gamma=0.1)

# Data augmentation: 4-pixel padding + 32x32 random crop, horizontal flip (Section 4.2)
train_data = CIFAR10(
    root='data', train=True, download=True,
    transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
)

test_data = CIFAR10(
    root='data', train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128)

# Create folders
model_name = f'CifarResNet-{args.n}-{"R" if args.residual else "P"}-{args.option or "N"}'
root = os.path.join(
    'models',
    model_name,
    now.strftime('%m_%d_%Y'),
    now.strftime('%H_%M_%S')
)
weight_dir = os.path.join(root, 'weights')
if not os.path.isdir(weight_dir):
    os.makedirs(weight_dir)

train_losses = np.empty((2, 0))
test_losses = np.empty((2, 0))
train_errors = np.empty((2, 0))
test_errors = np.empty((2, 0))
best_test_error = 1.0


def save():
    np.save(os.path.join(root, 'train_losses'), train_losses)
    np.save(os.path.join(root, 'test_losses'), test_losses)
    np.save(os.path.join(root, 'train_errors'), train_errors)
    np.save(os.path.join(root, 'test_errors'), test_errors)


for epoch in tqdm(range(5 if args.quick else 160), desc='Epoch'):
    if large and epoch == 1:
        for g in optimizer.param_groups:
            g['lr'] = 0.1

    model.train()
    train_loss, accuracy = 0, 0
    for images, labels in tqdm(train_loader, desc='Train', leave=False):
        images, labels = images.to(device), labels.to(device)

        # forward pass
        pred = model.forward(images)
        loss = loss_fn(pred, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item() / len(train_loader)
        accuracy += labels.eq(torch.argmax(pred, 1)).sum().item() / len(train_data)
        del images, labels

    train_losses = np.append(train_losses, [[epoch], [train_loss]], axis=1)
    train_errors = np.append(train_errors, [[epoch], [1 - accuracy]], axis=1)
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Error/train', 1 - accuracy, epoch)
    writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

    if epoch % 4 == 0:
        model.eval()
        with torch.no_grad():
            test_loss, accuracy = 0.0, 0.0
            for images, labels in tqdm(test_loader, desc='Test', leave=False):
                images, labels = images.to(device), labels.to(device)

                pred = model.forward(images)
                loss = loss_fn(pred, labels)

                test_loss += loss.item() / len(test_loader)
                accuracy += labels.eq(torch.argmax(pred, 1)).sum().item() / len(test_data)
                del images, labels

        test_error = 1 - accuracy
        test_losses = np.append(test_losses, [[epoch], [test_loss]], axis=1)
        test_errors = np.append(test_errors, [[epoch], [test_error]], axis=1)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Error/test', test_error, epoch)

        if test_error < best_test_error:
            best_test_error = test_error
            torch.save(model.state_dict(), os.path.join(weight_dir, 'best.pt'))

    save()
    if epoch % 20 == 0:
        torch.save(model.state_dict(), os.path.join(weight_dir, f'cp_{epoch}'))
save()
torch.save(model.state_dict(), os.path.join(weight_dir, 'final'))
print(f'Best test error: {best_test_error * 100:.2f}%')
