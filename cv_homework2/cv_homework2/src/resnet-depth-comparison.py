import time
import configargparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.contrib.handlers import ProgressBar
import json
import os


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


# ResNet10: [1,1,1,1] with BasicBlock
def resnet10(**kwargs):
    return ResNetCustom(BasicBlock, [1, 1, 1, 1], **kwargs)


# ResNet18: [2,2,2,2] with BasicBlock
def resnet18(**kwargs):
    return ResNetCustom(BasicBlock, [2, 2, 2, 2], **kwargs)


# ResNet50: [3,4,6,3] with Bottleneck
def resnet50(**kwargs):
    return ResNetCustom(Bottleneck, [3, 4, 6, 3], **kwargs)


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()

    parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file')
    parser.add_argument('--root', required=False, type=str, default='./data', help='data root path')
    parser.add_argument('--workers', required=False, type=int, default=8, help='number of workers for data loader')
    parser.add_argument('--bsize', required=False, type=int, default=256, help='batch size')
    parser.add_argument('--epochs', required=False, type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', required=False, type=float, default=0.1, help='learning rate')
    parser.add_argument('--device', required=False, default=None, type=str, help='CUDA device')
    parser.add_argument('--model', required=False, type=str, default='resnet10', 
                        choices=['resnet10', 'resnet18', 'resnet50'], help='model architecture')
    parser.add_argument('--output-name', required=False, type=str, default='training_result', 
                        help='output filename prefix')
    options = parser.parse_args()

    # Device setup
    if options.device is None:
        device = torch.device('cpu')
        gpu_ids = []
    else:
        if options.device.lower() == 'all':
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            gpu_ids = [int(d.strip()) for d in options.device.split(',')]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        print(f'Using GPU: cuda:{gpu_ids[0]}')

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

    train_set = torchvision.datasets.CIFAR100(root=options.root, train=True, 
                                              download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=options.bsize,
                                               shuffle=True, num_workers=options.workers)

    test_set = torchvision.datasets.CIFAR100(root=options.root, train=False,
                                             download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=options.bsize,
                                              shuffle=False, num_workers=options.workers)

    # Model creation
    model_dict = {
        'resnet10': resnet10,
        'resnet18': resnet18,
        'resnet50': resnet50
    }
    
    model = model_dict[options.model](num_classes=100)
    model = model.to(device)
    
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\nModel: {options.model.upper()}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer and loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

    # Training and evaluation functions
    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def eval_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            return y_pred, y

    trainer = Engine(train_step)
    evaluator = Engine(eval_step)
    
    Accuracy().attach(evaluator, 'accuracy')
    Loss(criterion).attach(evaluator, 'loss')
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    pbar = ProgressBar()
    pbar.attach(trainer, ['loss'])

    # Tracking metrics
    epoch_losses = []
    epoch_accuracies = []
    learning_rates = []
    best_accuracy = [0.0]  # Use list to allow mutation in nested function
    best_model_state = [None]

    @trainer.on(Events.EPOCH_STARTED)
    def record_lr(engine):
        learning_rates.append(optimizer.param_groups[0]['lr'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        epoch = engine.state.epoch
        avg_loss = engine.state.metrics['loss']
        epoch_losses.append(avg_loss)
        
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        epoch_accuracies.append(avg_accuracy)
        
        # Save best model
        if avg_accuracy > best_accuracy[0]:
            best_accuracy[0] = avg_accuracy
            best_model_state[0] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        current_lr = learning_rates[-1]
        pbar.log_message(
            f"Epoch [{epoch}/{options.epochs}] - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, LR: {current_lr:.6f}"
        )
        
        scheduler.step()

    # Training
    print(f"\n{'='*60}")
    print(f"Training {options.model.upper()} on CIFAR-100")
    print(f"Epochs: {options.epochs}, Batch Size: {options.bsize}, LR: {options.lr}")
    print(f"{'='*60}\n")
    
    t0 = time.time()
    trainer.run(train_loader, max_epochs=options.epochs)
    t1 = time.time()
    
    # Save best model checkpoint
    model_checkpoint_file = f"{options.output_name.replace('.json', '')}_best.pth"
    if best_model_state[0] is not None:
        torch.save(best_model_state[0], model_checkpoint_file)
        print(f"Best model saved to: {model_checkpoint_file}")
    
    # Save results
    results_dict = {
        'model': options.model,
        'config': {
            'epochs': options.epochs,
            'batch_size': options.bsize,
            'initial_lr': options.lr,
            'total_params': total_params,
            'trainable_params': trainable_params,
        },
        'training_time': t1 - t0,
        'epoch_losses': epoch_losses,
        'epoch_accuracies': epoch_accuracies,
        'learning_rates': learning_rates,
        'best_accuracy': max(epoch_accuracies),
        'final_accuracy': epoch_accuracies[-1],
        'final_loss': epoch_losses[-1],
        'model_checkpoint': model_checkpoint_file,
    }
    
    output_file = f"{options.output_name}.json"
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training Complete - {options.model.upper()}")
    print(f"  Best Accuracy: {max(epoch_accuracies):.4f}")
    print(f"  Final Accuracy: {epoch_accuracies[-1]:.4f}")
    print(f"  Final Loss: {epoch_losses[-1]:.4f}")
    print(f"  Total Time: {(t1-t0)/60:.2f} min")
    print(f"  Results saved to: {output_file}")
    print(f"{'='*60}\n")
