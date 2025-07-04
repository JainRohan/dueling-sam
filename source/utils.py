import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn as nn

def get_cifar_data(dataset='cifar10', batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform_test)
        num_classes = 10
    else:
        trainset = torchvision.datasets.CIFAR100(root='/home/rohan/narval_checkpoints_march_2025/rjain/cifar100/cifar-100-python/', train=True,
                                               download=False, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='/home/rohan/narval_checkpoints_march_2025/rjain/cifar100/cifar-100-python/', train=False,
                                              download=False, transform=transform_test)
        num_classes = 100

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    return trainloader, testloader, num_classes

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        if optimizer.__class__.__name__ == 'DuelingSAM':
            # For DuelingSAM, the optimizer steps do not use gradients.
            with torch.no_grad():
                pred = model(X)
                loss = loss_fn(pred, y)
                running_loss += loss.item()
            
            optimizer.first_step(zero_grad=True, model=model, inputs=X, targets=y)
            optimizer.second_step(zero_grad=True, model=model, inputs=X, targets=y)
        else:
            # Standard training for other optimizers
            loss = loss_fn(model(), y)
            running_loss += loss.item()
            loss.backward()
            
            # SAM requires two forward-backward passes.
            if optimizer.__class__.__name__ == 'SAM':
                optimizer.first_step(zero_grad=True)
                loss_fn(model(X), y).backward()
                optimizer.second_step(zero_grad=True)
            else:
                # Regular SGD training
                optimizer.step()
                optimizer.zero_grad()

    return running_loss / len(dataloader)

def test(model, testloader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return test_loss/len(testloader), 100.*correct/total

# Dueling feedback estimator: estimates normalized gradient direction using function comparisons
def dueling_feedback_estimate(f, x, rho=0.05, gamma=1e-2, num_samples=50, device=None):
    """
    Estimate the normalized gradient direction at x using Dueling Feedback for SAM.
    1. Sample u, compute h_k using dueling feedback (inner step)
    2. Compute x_adv = x + rho * h_k * u
    3. Sample u', compute h_k' using dueling feedback at x_adv (outer step)
    4. Accumulate h_k' * u' for grad_est, average over num_samples
    Args:
        f: function that takes a torch tensor and returns a scalar (loss)
        x: torch tensor (current point)
        rho: float, adversarial perturbation scale
        gamma: float, dueling feedback step size
        num_samples: int, number of random directions to average
        device: torch device (optional)
    Returns:
        torch tensor, estimated normalized gradient direction (same shape as x)
    """
    x = x.detach()
    if device is None:
        device = x.device
    grad_est = torch.zeros_like(x)
    for _ in range(num_samples):
        # Inner dueling feedback
        u = torch.randn_like(x)
        u = u / (u.norm() + 1e-12)
        f_plus = f(x + gamma * u)
        f_minus = f(x - gamma * u)
        h_k = 2 * (1.0 if f_plus > f_minus else 0.0) - 1  # +1 or -1
        # Adversarial point
        x_adv = x + rho * h_k * u
        # Outer dueling feedback
        u_prime = torch.randn_like(x)
        u_prime = u_prime / (u_prime.norm() + 1e-12)
        f_plus_outer = f(x_adv + gamma * u_prime)
        f_minus_outer = f(x_adv - gamma * u_prime)
        h_k_prime = 2 * (1.0 if f_plus_outer > f_minus_outer else 0.0) - 1  # +1 or -1
        grad_est += h_k_prime * u_prime
    grad_est = grad_est / num_samples 
    return grad_est