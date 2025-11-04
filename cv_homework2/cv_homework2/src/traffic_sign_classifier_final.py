import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.resnet import BasicBlock, ResNet
from PIL import Image
import json
import argparse
from pathlib import Path


class TrafficSignDataset(Dataset):
    """Custom Dataset for Traffic Sign Classification"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((str(img_path), class_idx))
        
        print(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ResNetCustom(ResNet):
    """Custom ResNet for Traffic Sign Classification"""
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def resnet18(num_classes=10):
    return ResNetCustom(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    return {
        'loss': total_loss / len(train_loader),
        'accuracy': 100. * correct / total
    }


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return {
        'loss': total_loss / len(test_loader),
        'accuracy': 100. * correct / total
    }


def main():
    parser = argparse.ArgumentParser(description='Traffic Sign Classification - Optimized V1')
    parser.add_argument('--data-dir', type=str, default='data/Traffic_sign')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--output', type=str, default='report/traffic_sign_final')
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Optimized data augmentation
    transform_train = transforms.Compose([
        transforms.Resize(72),
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(72),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = TrafficSignDataset(
        root_dir=os.path.join(args.data_dir, 'train_dataset'),
        transform=transform_train
    )
    test_dataset = TrafficSignDataset(
        root_dir=os.path.join(args.data_dir, 'test_dataset'),
        transform=transform_test
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    
    print(f'\nDataset Statistics:')
    print(f'  Classes: {len(train_dataset.classes)}')
    print(f'  Train samples: {len(train_dataset)}')
    print(f'  Test samples: {len(test_dataset)}')
    print(f'  Class names: {train_dataset.classes}')
    
    model = resnet18(num_classes=len(train_dataset.classes)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel: ResNet18')
    print(f'Total parameters: {total_params:,}')
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.999), weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    print(f'\nTraining Configuration:')
    print(f'  Epochs: {args.epochs}')
    print(f'  Batch size: {args.batch_size}')
    print(f'  Learning rate: {args.lr}')
    print(f'  Optimizer: AdamW (weight_decay=0.01)')
    print(f'  Scheduler: CosineAnnealingWarmRestarts (T_0=20)')
    print(f'  Dropout: 0.4')
    print(f'  Label smoothing: 0.1')
    
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
        'learning_rate': []
    }
    
    best_acc = 0
    t0 = time.time()
    
    print(f'\n{"="*70}')
    print(f'Starting Training')
    print(f'{"="*70}\n')
    
    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(lr)
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['test_loss'].append(test_metrics['loss'])
        history['test_accuracy'].append(test_metrics['accuracy'])
        
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            torch.save(model.state_dict(), f'{args.output}_best.pth')
        
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch [{epoch}/{args.epochs}] LR: {lr:.6f}')
            print(f'  Train - Loss: {train_metrics["loss"]:.4f}, Acc: {train_metrics["accuracy"]:.2f}%')
            print(f'  Test  - Loss: {test_metrics["loss"]:.4f}, Acc: {test_metrics["accuracy"]:.2f}%')
            print(f'  Best Acc: {best_acc:.2f}%')
        
        scheduler.step()
    
    t1 = time.time()
    
    torch.save(model.state_dict(), f'{args.output}_final.pth')
    
    results = {
        'dataset': 'Traffic Sign',
        'num_classes': len(train_dataset.classes),
        'class_names': train_dataset.classes,
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'best_accuracy': best_acc,
        'final_accuracy': history['test_accuracy'][-1],
        'training_time': t1 - t0,
        'config': {
            'model': 'resnet18',
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingWarmRestarts',
            'label_smoothing': 0.1,
            'dropout': 0.4,
            'weight_decay': 0.01
        },
        'history': history
    }
    
    with open(f'{args.output}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\n{"="*70}')
    print(f'Training Complete!')
    print(f'  Best Test Accuracy: {best_acc:.2f}%')
    print(f'  Final Test Accuracy: {history["test_accuracy"][-1]:.2f}%')
    print(f'  Training Time: {(t1-t0)/60:.2f} min')
    print(f'  Results saved to: {args.output}_results.json')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
