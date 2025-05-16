import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

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

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader, num_classes

def train(model, trainloader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # First forward-backward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        
        # Second forward-backward pass
        criterion(model(inputs), targets).backward()
        optimizer.second_step(zero_grad=True)
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return train_loss/len(trainloader), 100.*correct/total

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