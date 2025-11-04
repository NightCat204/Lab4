import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
import json
import argparse


class ResNetCustom(ResNet):
    def __init__(self, block, layers, num_classes=1000):
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
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


def resnet18(**kwargs):
    return ResNetCustom(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet50(**kwargs):
    return ResNetCustom(Bottleneck, [3, 4, 6, 3], **kwargs)


class DistillationLoss(nn.Module):
    """
    Combined loss: hard target + soft target + logits matching
    """
    def __init__(self, temperature=4.0, alpha=0.3, beta=0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for soft target loss
        self.beta = beta    # Weight for logits matching loss
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()

    def forward(self, student_logits, teacher_logits, targets):
        # Hard target loss (standard cross-entropy)
        hard_loss = self.ce_loss(student_logits, targets)
        
        # Soft target loss (KL divergence with temperature)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # Logits matching loss (MSE on raw logits)
        logits_loss = self.mse_loss(student_logits, teacher_logits)
        
        # Combined loss
        total_loss = (1 - self.alpha - self.beta) * hard_loss + \
                     self.alpha * soft_loss + \
                     self.beta * logits_loss
        
        return total_loss, hard_loss, soft_loss, logits_loss


def train_epoch(student, teacher, train_loader, optimizer, criterion, device):
    """Train one epoch with distillation"""
    student.train()
    teacher.eval()
    
    total_loss = 0
    total_hard = 0
    total_soft = 0
    total_logits = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Get teacher predictions (no gradient)
        with torch.no_grad():
            teacher_logits = teacher(data)
        
        # Get student predictions
        student_logits = student(data)
        
        # Calculate combined loss
        loss, hard_loss, soft_loss, logits_loss = criterion(student_logits, teacher_logits, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_hard += hard_loss.item()
        total_soft += soft_loss.item()
        total_logits += logits_loss.item()
        
        # Calculate accuracy
        _, predicted = student_logits.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    return {
        'loss': total_loss / len(train_loader),
        'hard_loss': total_hard / len(train_loader),
        'soft_loss': total_soft / len(train_loader),
        'logits_loss': total_logits / len(train_loader),
        'accuracy': 100. * correct / total
    }


def evaluate(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return {
        'loss': test_loss / len(test_loader),
        'accuracy': 100. * correct / total
    }


def main():
    parser = argparse.ArgumentParser(description='Knowledge Distillation: ResNet50 -> ResNet18')
    parser.add_argument('--teacher-checkpoint', type=str, required=True, 
                        help='Path to teacher model checkpoint')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--alpha', type=float, default=0.3, help='Weight for soft target loss')
    parser.add_argument('--beta', type=float, default=0.3, help='Weight for logits loss')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--output', type=str, default='report/distilled_resnet18')
    args = parser.parse_args()
    
    # Device setup
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data preparation with enhanced augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    train_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                              download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               shuffle=True, num_workers=8)
    
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                             download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                              shuffle=False, num_workers=8)
    
    # Load teacher model (ResNet50)
    print('\nLoading teacher model (ResNet50)...')
    teacher = resnet50(num_classes=100).to(device)
    checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
    teacher.load_state_dict(checkpoint)
    teacher.eval()
    
    # Evaluate teacher
    teacher_metrics = evaluate(teacher, test_loader, device)
    print(f'Teacher accuracy: {teacher_metrics["accuracy"]:.2f}%')
    
    # Create student model (ResNet18)
    print('\nCreating student model (ResNet18)...')
    student = resnet18(num_classes=100).to(device)
    
    # Distillation loss
    criterion = DistillationLoss(
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
    
    # Training
    print(f'\n{"="*70}')
    print(f'Knowledge Distillation Training')
    print(f'Teacher: ResNet50 ({teacher_metrics["accuracy"]:.2f}%)')
    print(f'Student: ResNet18')
    print(f'Temperature: {args.temperature}, Alpha: {args.alpha}, Beta: {args.beta}')
    print(f'{"="*70}\n')
    
    history = {
        'train_loss': [],
        'train_hard_loss': [],
        'train_soft_loss': [],
        'train_logits_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
        'learning_rate': []
    }
    
    best_acc = 0
    t0 = time.time()
    
    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(lr)
        
        # Train
        train_metrics = train_epoch(student, teacher, train_loader, optimizer, criterion, device)
        
        # Evaluate
        test_metrics = evaluate(student, test_loader, device)
        
        # Record metrics
        history['train_loss'].append(train_metrics['loss'])
        history['train_hard_loss'].append(train_metrics['hard_loss'])
        history['train_soft_loss'].append(train_metrics['soft_loss'])
        history['train_logits_loss'].append(train_metrics['logits_loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['test_loss'].append(test_metrics['loss'])
        history['test_accuracy'].append(test_metrics['accuracy'])
        
        # Update best accuracy
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            # Save best model
            torch.save(student.state_dict(), f'{args.output}_best.pth')
        
        # Print progress
        print(f'Epoch [{epoch}/{args.epochs}] LR: {lr:.6f}')
        print(f'  Train - Loss: {train_metrics["loss"]:.4f} (Hard: {train_metrics["hard_loss"]:.4f}, '
              f'Soft: {train_metrics["soft_loss"]:.4f}, Logits: {train_metrics["logits_loss"]:.4f}), '
              f'Acc: {train_metrics["accuracy"]:.2f}%')
        print(f'  Test  - Loss: {test_metrics["loss"]:.4f}, Acc: {test_metrics["accuracy"]:.2f}%')
        print()
        
        scheduler.step()
    
    t1 = time.time()
    
    # Save final model
    torch.save(student.state_dict(), f'{args.output}_final.pth')
    
    # Save results
    results = {
        'teacher_accuracy': teacher_metrics['accuracy'],
        'student_best_accuracy': best_acc,
        'student_final_accuracy': history['test_accuracy'][-1],
        'training_time': t1 - t0,
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'temperature': args.temperature,
            'alpha': args.alpha,
            'beta': args.beta
        },
        'history': history
    }
    
    with open(f'{args.output}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\n{"="*70}')
    print(f'Training Complete!')
    print(f'  Teacher Accuracy: {teacher_metrics["accuracy"]:.2f}%')
    print(f'  Student Best Accuracy: {best_acc:.2f}%')
    print(f'  Student Final Accuracy: {history["test_accuracy"][-1]:.2f}%')
    print(f'  Training Time: {(t1-t0)/60:.2f} min')
    print(f'  Results saved to: {args.output}_results.json')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
