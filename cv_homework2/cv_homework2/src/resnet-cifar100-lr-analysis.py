import time
import configargparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import BasicBlock, ResNet
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import ProgressBar
import matplotlib.pyplot as plt
import json
import os

results = []
train_losses = []  # Track training loss per epoch


class ResNetCustom(ResNet):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
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


def resnet10(**kwargs):
    return ResNetCustom(BasicBlock, [1, 1, 1, 1], **kwargs)


def logger(engine, model, evaluator, loader, pbar, train_loss_history):
    evaluator.run(loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    
    # Get average loss from training history
    avg_loss = train_loss_history[-1] if train_loss_history else 0
    
    pbar.log_message(
        "Test Results - Avg accuracy: {:.2f}, Avg loss: {:.4f}".format(avg_accuracy, avg_loss)
    )
    results.append(avg_accuracy)


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()

    parser.add_argument('-c', '--config', required=False,
                        is_config_file=True, help='config file')
    parser.add_argument('--root', required=False, type=str, default='./data',
                        help='data root path')
    parser.add_argument('--workers', required=False, type=int, default=4,
                        help='number of workers for data loader')
    parser.add_argument('--bsize', required=False, type=int, default=256,
                        help='batch size')
    parser.add_argument('--epochs', required=False, type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--lr', required=False, type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('--device', required=False, default=None, type=str,
                        help='CUDA device ids for GPU training')
    parser.add_argument('--lr-schedule', required=False, action='store_true',
                        help='Enable learning rate schedule (decay at epoch 30, 40)')
    parser.add_argument('--output-name', required=False, type=str, default='training_result',
                        help='Output filename prefix for results')
    options = parser.parse_args()

    root = options.root
    bsize = options.bsize
    workers = options.workers
    epochs = options.epochs
    initial_lr = options.lr
    
    # Device selection
    if options.device is None:
        device = torch.device('cpu')
        gpu_ids = []
    else:
        if options.device.lower() == 'all':
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            gpu_ids = [int(d.strip()) for d in options.device.split(',')]
        
        device = torch.device(f'cuda:{gpu_ids[0]}')
        if len(gpu_ids) > 1:
            print(f'Using multi-GPU training on devices: {gpu_ids}')
        else:
            print(f'Using single GPU: cuda:{gpu_ids[0]}')

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.CIFAR100(root=root, train=True,
                                             download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bsize,
                                               shuffle=True, num_workers=workers)

    test_set = torchvision.datasets.CIFAR100(root=root, train=False,
                                            download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=bsize,
                                              shuffle=False, num_workers=workers)

    model = resnet10(num_classes=100)
    model = model.to(device)
    
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=0.0001)
    
    # Learning rate scheduler: step decay at epoch 30 and 40
    if options.lr_schedule:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
        print(f'Using LR schedule: initial={initial_lr}, decay at epoch 30 and 40 by 10x')
    else:
        scheduler = None
        print(f'Using fixed learning rate: {initial_lr}')

    # Custom training function to track loss
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

    # Custom evaluation function
    def eval_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            return y_pred, y

    trainer = Engine(train_step)
    evaluator = Engine(eval_step)
    
    # Metrics
    from ignite.metrics import Accuracy, Loss
    Accuracy().attach(evaluator, 'accuracy')
    Loss(criterion).attach(evaluator, 'loss')
    
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    pbar = ProgressBar()
    pbar.attach(trainer, ['loss'])

    # Track loss per epoch
    epoch_losses = []
    epoch_accuracies = []
    learning_rates = []

    @trainer.on(Events.EPOCH_STARTED)
    def record_lr_before_epoch(engine):
        # Record learning rate BEFORE training this epoch
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        epoch = engine.state.epoch
        
        # Record average loss for this epoch
        avg_loss = engine.state.metrics['loss']
        epoch_losses.append(avg_loss)
        
        # Get the learning rate that was used for this epoch
        current_lr = learning_rates[-1] if learning_rates else optimizer.param_groups[0]['lr']
        
        # Evaluate
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        epoch_accuracies.append(avg_accuracy)
        
        pbar.log_message(
            f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, LR: {current_lr:.6f}"
        )
        
        # Step scheduler AFTER epoch completes (for next epoch)
        if scheduler:
            scheduler.step()

    # Start training
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Initial LR: {initial_lr}")
    print(f"  LR Schedule: {'Enabled (decay at 30, 40)' if options.lr_schedule else 'Disabled'}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {bsize}")
    print(f"{'='*60}\n")
    
    t0 = time.time()
    trainer.run(train_loader, max_epochs=epochs)
    t1 = time.time()
    
    # Save results to JSON
    results_dict = {
        'config': {
            'initial_lr': initial_lr,
            'lr_schedule': options.lr_schedule,
            'epochs': epochs,
            'batch_size': bsize,
        },
        'training_time': t1 - t0,
        'epoch_losses': epoch_losses,
        'epoch_accuracies': epoch_accuracies,
        'learning_rates': learning_rates,
        'best_accuracy': max(epoch_accuracies),
    }
    
    output_file = f"{options.output_name}.json"
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Best Accuracy: {max(epoch_accuracies):.4f}")
    print(f"  Final Loss: {epoch_losses[-1]:.4f}")
    print(f"  Total Time: {t1 - t0:.2f}s")
    print(f"  Results saved to: {output_file}")
    print(f"{'='*60}\n")
